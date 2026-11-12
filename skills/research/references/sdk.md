# Tinker SDK & Types Reference

Complete reference for the Tinker SDK clients and all SDK types.

---

## SDK API

### ServiceClient (entry point)

```python
from tinker import RestClient, ServiceClient

svc = ServiceClient(user_metadata={"experiment": "v1"}, project_id="my-project")

# Create a new LoRA training client
tc = svc.create_lora_training_client(
    base_model="Qwen/Qwen3-8B", rank=32,
    seed=None, train_mlp=True, train_attn=True, train_unembed=True,
).result()

# Resume from a training checkpoint
tc = svc.create_training_client_from_state(path="tinker://...").result()
tc = svc.create_training_client_from_state_with_optimizer(path="tinker://...").result()

# Create a sampling client
sc = svc.create_sampling_client(model_path="tinker://...", base_model=None, retry_config=None).result()

# Create a REST client
rest = RestClient()

# Query available models
caps = rest.get_server_capabilities().result()
```

Client factories return `APIFuture`s: use `await` in async code or `.result()` in synchronous code.

### TrainingClient

```python
# Forward/backward pass (compute loss + gradients)
future = tc.forward_backward(data=[datum1, datum2], loss_fn="cross_entropy")

# Forward-only pass (compute loss, no gradients — useful for eval)
future = tc.forward(data=[datum1, datum2], loss_fn="cross_entropy")

# Custom loss function
future = tc.forward_backward_custom(data=[datum1, datum2], loss_fn=my_custom_loss_fn)

# Optimizer step
tc.optim_step(adam_params=AdamParams(learning_rate=2e-4))

# Checkpointing
tc.save_state(name="step_100", ttl_seconds=None)
tc.save_weights_for_sampler(name="step_100_sampler", ttl_seconds=None)
sc = tc.save_weights_and_get_sampling_client().result()  # Ephemeral, not persistently saved

# Metadata
info = tc.get_info()
tokenizer = tc.get_tokenizer()
```

### Loss functions

- `"cross_entropy"` — Standard SL loss
- `"importance_sampling"` — On-policy RL
- `"stable_reinforce"` — REINFORCE variant that stays stable when the sampling policy differs from the one being trained (default for async runs in `rl/train.py`, via `AsyncConfig.loss_fn`; needs provenance spans, see `Datum.with_provenance`)
- `"ppo"` — Proximal Policy Optimization
- `"cispo"` — Conservative Importance Sampling PPO
- `"dro"` — Distributionally Robust Optimization

See the loss function source or the [Tinker cookbook repo](https://github.com/thinking-machines-lab/tinker-cookbook) for details and `loss_fn_config` parameters.

### Futures and pipelining

Network operations return `APIFuture`s. Submit training operations before waiting:
```python
fb_future = tc.forward_backward(data=data, loss_fn="cross_entropy")
optim_future = tc.optim_step(adam_params=adam_params)
fb_result = fb_future.result()
optim_result = optim_future.result()
```

Submit `forward_backward` and `optim_step` back-to-back before waiting to overlap GPU computation with data preparation. In async code, await each future directly; pass `.coroutine()` to `asyncio.gather` or `asyncio.create_task`.

### SamplingClient

```python
from tinker import SamplingParams

response = sc.sample(
    prompt=model_input, num_samples=4,
    sampling_params=SamplingParams(max_tokens=256, temperature=1.0),
    include_prompt_logprobs=False, topk_prompt_logprobs=0,
).result()

for seq in response.sequences:
    print(seq.tokens, seq.logprobs, seq.stop_reason)

# Get logprobs for existing tokens (no generation)
logprobs_response = sc.compute_logprobs(prompt=model_input).result()

# Metadata
base_model = sc.get_base_model()
tokenizer = sc.get_tokenizer()
```

SamplingClient is picklable for multiprocessing. Always create a new SamplingClient after saving weights.

### RestClient

```python
rest = RestClient()

# Training runs
runs = rest.list_training_runs(limit=20, offset=0, access_scope="owned").result()
run = rest.get_training_run(training_run_id="...").result()
run = rest.get_training_run_by_tinker_path(tinker_path="tinker://...").result()

# Checkpoints
checkpoints = rest.list_checkpoints(training_run_id="...").result()
all_checkpoints = rest.list_user_checkpoints(limit=100, offset=0).result()
rest.delete_checkpoint_from_tinker_path(tinker_path="tinker://...").result()

# Visibility & TTL
rest.publish_checkpoint_from_tinker_path(tinker_path="tinker://...").result()
rest.unpublish_checkpoint_from_tinker_path(tinker_path="tinker://...").result()
rest.set_checkpoint_ttl_from_tinker_path(tinker_path="tinker://...", ttl_seconds=86400).result()

# Download URL & metadata
export = tc.save_weights_external(name="export").result()
url_resp = rest.get_external_weights_urls(tinker_path=export.path).result()
info = rest.get_weights_info_by_tinker_path(tinker_path="tinker://...").result()
```

RestClient methods return `APIFuture`s: await them in async code or call `.result()` in synchronous code.

### Retry behavior

The SDK retries all HTTP API calls automatically (10 attempts, exponential backoff with jitter). Retried: timeouts (408), lock conflicts (409), rate limits (429), server errors (500+), connection failures. Client errors (400, 401, 403, 404, 422) raise immediately.

Override via `max_retries` on client creation:
```python
svc = tinker.ServiceClient(max_retries=3)   # reduce retries
svc = tinker.ServiceClient(max_retries=0)   # disable retries
```

Do not add retry wrappers around Tinker API calls — the SDK handles this.

---

## Types

### Core data types

#### ModelInput

```python
from tinker import ModelInput

mi = ModelInput.from_ints([1, 2, 3, 4, 5])
tokens = mi.to_ints()
length = mi.length           # Property, not method
mi2 = mi.append(chunk)
mi3 = mi.append_int(42)
mi_empty = ModelInput.empty()
```

#### TensorData

```python
from tinker import TensorData

td = TensorData.from_numpy(np.array([1.0, 0.0, 1.0]))
td = TensorData.from_torch(torch.tensor([1.0, 0.0]))
arr = td.to_numpy()
tensor = td.to_torch()
lst = td.tolist()
# Fields: data (flat list), dtype ("int64"|"float32"), shape (optional)
```

#### Datum

```python
from tinker import Datum, ModelInput, TensorData

datum = Datum(
    model_input=ModelInput.from_ints(tokens),
    loss_fn_inputs={"weights": TensorData.from_numpy(weights_array)},
)
```

### Configuration types

#### SamplingParams

```python
from tinker import SamplingParams

params = SamplingParams(
    max_tokens=256, temperature=1.0, top_k=50, top_p=0.95,
    stop=["<|eot_id|>"], seed=42,
)
```

#### AdamParams

```python
from tinker import AdamParams

adam = AdamParams(
    learning_rate=2e-4, beta1=0.9, beta2=0.95, eps=1e-12,
    weight_decay=0.0, grad_clip_norm=1.0,
)
```

#### LoraConfig

```python
from tinker import LoraConfig

config = LoraConfig(rank=32, seed=None, train_mlp=True, train_attn=True, train_unembed=True)
```

### Response types

#### ForwardBackwardOutput

```python
result = tc.forward_backward(data=batch, loss_fn="cross_entropy").result()
result.metrics              # dict[str, float]
result.loss_fn_outputs      # list[LossFnOutput]
result.loss_fn_output_type  # str
```

#### SampleResponse / SampledSequence

```python
response = sc.sample(prompt=mi, num_samples=4, sampling_params=params).result()
response.sequences                # list[SampledSequence]
response.prompt_logprobs          # Optional[list[Optional[float]]]
response.topk_prompt_logprobs     # Optional[list[Optional[list[tuple[int, float]]]]]

for seq in response.sequences:
    seq.tokens       # list[int]
    seq.logprobs     # Optional[list[float]]
    seq.stop_reason  # StopReason: "length" | "stop"
```

#### Other response types

- `OptimStepResponse` — confirms parameter update
- `SaveWeightsResponse` — `path: str` (tinker:// path)
- `LoadWeightsResponse` — confirms loaded weights
- `GetInfoResponse` — `model_data: ModelData` (model_name, lora_rank, tokenizer_id)
- `GetServerCapabilitiesResponse` — `supported_models: list[SupportedModel]`
- `WeightsInfoResponse` — `base_model`, `lora_rank`, `is_lora`, `train_mlp`, `train_attn`, `train_unembed`

### Checkpoint and run types

```python
from tinker import TrainingRun, Checkpoint, CheckpointType, ParsedCheckpointTinkerPath

# TrainingRun
run.training_run_id, run.base_model, run.is_lora, run.lora_rank
run.last_checkpoint, run.user_metadata

# Checkpoint
ckpt.checkpoint_id, ckpt.checkpoint_type, ckpt.tinker_path
ckpt.size_bytes, ckpt.public, ckpt.expires_at

# Parse a tinker:// path
parsed = ParsedCheckpointTinkerPath.from_tinker_path("tinker://run-id/weights/ckpt-id")
parsed.training_run_id, parsed.checkpoint_type, parsed.checkpoint_id
```

### Error types

All exceptions inherit from `tinker.TinkerError`:
- **`APIStatusError`**: `BadRequestError` (400), `AuthenticationError` (401), `PermissionDeniedError` (403), `NotFoundError` (404), `ConflictError` (409), `UnprocessableEntityError` (422), `RateLimitError` (429), `InternalServerError` (500+)
- **`APIConnectionError`**, **`APITimeoutError`**, **`APIResponseValidationError`**
- **`RequestFailedError`** — async request failure with error category

### Cookbook helper functions

Use these instead of manual Datum construction:
- `tinker_cookbook.supervised.data.conversation_to_datum(messages, renderer, max_length, train_on_what)` — full SL pipeline
- `tinker_cookbook.supervised.common.datum_from_model_input_weights(model_input, weights, max_length)` — from ModelInput + weights
- `renderer.build_supervised_example(messages)` — returns `(ModelInput, weights)`
