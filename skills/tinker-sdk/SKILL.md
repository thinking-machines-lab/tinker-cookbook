---
name: tinker-sdk
description: Guide for using the Tinker Python SDK APIs — ServiceClient, TrainingClient, SamplingClient, RestClient, forward_backward, optim_step, sampling, and async patterns. Use when the user asks about Tinker API basics, how to call training/sampling, or how the SDK works.
---

# Tinker Python SDK

Help the user understand and use the core Tinker SDK APIs.

## Reference docs

Read these for authoritative API documentation:
- `docs/api-reference/serviceclient.md` — ServiceClient API
- `docs/api-reference/trainingclient.md` — TrainingClient API
- `docs/api-reference/samplingclient.md` — SamplingClient API
- `docs/api-reference/restclient.md` — RestClient API
- `docs/api-reference/types.md` — All SDK types
- `docs/training-sampling.mdx` — Starter walkthrough
- `docs/async.mdx` — Sync/async patterns, futures
- `docs/losses.mdx` — Loss functions
- `docs/under-the-hood.mdx` — Clock cycles, worker pools

## ServiceClient (entry point)

`ServiceClient` is the main entry point. All other clients are created from it.

```python
from tinker import ServiceClient

svc = ServiceClient(user_metadata={"experiment": "v1"}, project_id="my-project")

# Create a new LoRA training client
tc = svc.create_lora_training_client(
    base_model="Qwen/Qwen3-8B",
    rank=32,
    seed=None,
    train_mlp=True,
    train_attn=True,
    train_unembed=True,
)

# Resume from a training checkpoint
tc = svc.create_training_client_from_state(path="tinker://...")              # weights only
tc = svc.create_training_client_from_state_with_optimizer(path="tinker://...") # weights + optimizer

# Create a sampling client
sc = svc.create_sampling_client(model_path="tinker://...", base_model=None, retry_config=None)

# Create a REST client for checkpoint/run management
rest = svc.create_rest_client()

# Query available models
caps = svc.get_server_capabilities()  # returns GetServerCapabilitiesResponse
```

All creation methods have `_async` variants.

## TrainingClient

```python
# Forward/backward pass (compute loss + gradients)
result = tc.forward_backward(data=[datum1, datum2], loss_fn="cross_entropy")

# Forward-only pass (compute loss, no gradients — useful for eval)
result = tc.forward(data=[datum1, datum2], loss_fn="cross_entropy")

# Custom loss function
result = tc.forward_backward_custom(data=[datum1, datum2], loss_fn=my_custom_loss_fn)

# Optimizer step
tc.optim_step(adam_params=AdamParams(learning_rate=2e-4))

# Checkpointing
tc.save_state(name="step_100", ttl_seconds=None)                # Full state (resumable)
tc.save_weights_for_sampler(name="step_100_sampler", ttl_seconds=None)  # Sampler-only

# Save + get SamplingClient in one call (ephemeral, not persistently saved)
# NOTE: This does NOT persist the checkpoint. For persistent saves, use
# save_weights_for_sampler(name=...) + service_client.create_sampling_client(model_path=...)
sc = tc.save_weights_and_get_sampling_client()

# Load checkpoint
tc.load_state(path="tinker://...")
tc.load_state_with_optimizer(path="tinker://...")

# Metadata
info = tc.get_info()          # GetInfoResponse (model name, LoRA rank, tokenizer)
tokenizer = tc.get_tokenizer()  # HuggingFace tokenizer
```

### Loss functions
- `"cross_entropy"` — Standard SL loss
- `"importance_sampling"` — On-policy RL (default for GRPO)
- `"ppo"` — Proximal Policy Optimization
- `"cispo"` — Conservative Importance Sampling PPO
- `"dro"` — Distributionally Robust Optimization

See `docs/losses.mdx` for details and `loss_fn_config` parameters.

### Async variants

All methods have `_async` variants that return `APIFuture`:
```python
fb_future = tc.forward_backward_async(data=data, loss_fn="cross_entropy")
optim_future = tc.optim_step_async(adam_params=adam_params)
# Do other work...
fb_result = fb_future.result()
optim_result = optim_future.result()
```

**Key pattern:** Submit `forward_backward_async` and `optim_step_async` back-to-back before awaiting — this overlaps GPU computation with data preparation.

## SamplingClient

```python
from tinker import SamplingParams

sc = tc.save_weights_and_get_sampling_client()

response = sc.sample(
    prompt=model_input,
    num_samples=4,
    sampling_params=SamplingParams(max_tokens=256, temperature=1.0),
    include_prompt_logprobs=False,   # Set True to get per-token prompt logprobs
    topk_prompt_logprobs=0,          # Top-K logprobs per prompt token (0 = disabled)
)

for seq in response.sequences:
    print(seq.tokens, seq.logprobs, seq.stop_reason)

# Get logprobs for existing tokens (no generation)
logprobs_response = sc.compute_logprobs(prompt=model_input)

# Metadata
base_model = sc.get_base_model()    # Base model name string
tokenizer = sc.get_tokenizer()      # HuggingFace tokenizer
```

SamplingClient is picklable for multiprocessing use.

**Important:** Always create a **new** SamplingClient after saving weights. A stale client points at old weights.

## RestClient

For managing training runs and checkpoints. See also the `/tinker-cli` skill for CLI equivalents.

```python
rest = svc.create_rest_client()

# Training runs
runs = rest.list_training_runs(limit=20, offset=0, access_scope="owned")
run = rest.get_training_run(training_run_id="...")
run = rest.get_training_run_by_tinker_path(tinker_path="tinker://...")

# Checkpoints
checkpoints = rest.list_checkpoints(training_run_id="...")
all_checkpoints = rest.list_user_checkpoints(limit=100, offset=0)
rest.delete_checkpoint(training_run_id="...", checkpoint_id="...")
rest.delete_checkpoint_from_tinker_path(tinker_path="tinker://...")

# Checkpoint visibility
rest.publish_checkpoint_from_tinker_path(tinker_path="tinker://...")    # Make public
rest.unpublish_checkpoint_from_tinker_path(tinker_path="tinker://...")  # Make private

# Checkpoint TTL
rest.set_checkpoint_ttl_from_tinker_path(tinker_path="tinker://...", ttl_seconds=86400)

# Download URL
url_resp = rest.get_checkpoint_archive_url_from_tinker_path(tinker_path="tinker://...")

# Checkpoint metadata
info = rest.get_weights_info_by_tinker_path(tinker_path="tinker://...")
```

All RestClient methods have `_async` variants.

## Retry behavior

The Tinker SDK retries **all** HTTP API calls automatically (10 attempts, exponential backoff with jitter). Retried request types: timeouts (408), lock conflicts (409), rate limits (429), server errors (500+), and connection failures. The SDK respects `Retry-After` headers and attaches idempotency keys to non-GET requests.

Client errors (400, 401, 403, 404, 422) are **not** retried — these raise immediately (e.g., `tinker.BadRequestError`, `tinker.AuthenticationError`).

Override via `max_retries` on client creation:
```python
svc = tinker.ServiceClient(max_retries=3)   # reduce retries
svc = tinker.ServiceClient(max_retries=0)   # disable retries
```

**Do not** add retry wrappers around Tinker API calls in training loops — the SDK handles this. Enable retry logging with `logging.getLogger("tinker").setLevel(logging.DEBUG)`.

## Common pitfalls
- **Use ServiceClient** to create clients — `TrainingClient` and `SamplingClient` cannot be constructed directly
- Always await futures before submitting new forward_backward calls
- Submit `forward_backward_async` + `optim_step_async` back-to-back before awaiting
- Create a **new** SamplingClient after saving weights (sampler desync)
- Use `save_state` for resumable checkpoints, `save_weights_for_sampler` for sampling-only
- `forward()` computes loss without gradients — use for eval, not training
