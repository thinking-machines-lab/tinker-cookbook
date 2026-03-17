---
name: tinker-sdk
description: Guide for using the Tinker Python SDK APIs — TrainingClient, SamplingClient, forward_backward, optim_step, sampling, and async patterns. Use when the user asks about Tinker API basics, how to call training/sampling, or how the SDK works.
---

# Tinker Python SDK

Help the user understand and use the core Tinker SDK APIs.

## Overview

The Tinker SDK provides two main clients:
- **TrainingClient** — runs forward/backward passes, optimizer steps, checkpointing
- **SamplingClient** — generates text from a model

Both run on remote GPU workers. Your code runs on a CPU machine and calls the SDK over the network.

## Reference docs

Read these for authoritative API documentation:
- `docs/api-reference/trainingclient.md` — TrainingClient API
- `docs/api-reference/samplingclient.md` — SamplingClient API
- `docs/api-reference/types.md` — All SDK types
- `docs/training-sampling.mdx` — Starter walkthrough
- `docs/async.mdx` — Sync/async patterns, futures
- `docs/losses.mdx` — Loss functions
- `docs/under-the-hood.mdx` — Clock cycles, worker pools

## TrainingClient

```python
from tinker import TrainingClient

tc = TrainingClient(model_name="meta-llama/Llama-3.1-8B")

# Forward/backward pass
result = tc.forward_backward(data=[datum1, datum2], loss_fn="cross_entropy")

# Optimizer step
tc.optim_step(adam_params=AdamParams(learning_rate=2e-4))

# Checkpointing
tc.save_state(name="step_100")                        # Full state (resumable)
tc.save_weights_for_sampler(name="step_100_sampler")   # Sampler-only weights

# Load checkpoint
tc.load_state(path="tinker://...")
```

### Key methods
- `forward_backward(data, loss_fn, loss_fn_config)` — Compute loss and gradients
- `optim_step(adam_params)` — Apply gradients
- `save_state(name, ttl_seconds)` — Save full state (weights + optimizer) for resumption
- `save_weights_for_sampler(name, ttl_seconds)` — Save weights for sampling
- `save_weights_and_get_sampling_client(name)` — Save + create SamplingClient in one call
- `load_state(path)` / `load_state_with_optimizer(path)` — Resume from checkpoint
- `get_tokenizer()` — Get the model's tokenizer
- `get_info()` — Model metadata

### Async variants
All methods have `_async` variants that return `APIFuture`:
```python
fb_future = tc.forward_backward_async(data=data, loss_fn="cross_entropy")
optim_future = tc.optim_step_async(adam_params=adam_params)
# Do other work...
fb_result = fb_future.result()
optim_result = optim_future.result()
```

**Key pattern:** Submit forward_backward_async and optim_step_async back-to-back before awaiting — this overlaps GPU computation with data preparation.

### Loss functions
- `"cross_entropy"` — Standard SL loss
- `"importance_sampling"` — On-policy RL (default for GRPO)
- `"ppo"` — Proximal Policy Optimization
- `"cispo"` — Conservative Importance Sampling PPO
- `"dro"` — Distributionally Robust Optimization
- `"dpo"` — Direct Preference Optimization
- `"forward_backward_custom"` — Custom loss via CustomLossFnV1

See `docs/losses.mdx` for details and `loss_fn_config` parameters.

## SamplingClient

```python
from tinker import SamplingClient, SamplingParams

sc = tc.save_weights_and_get_sampling_client(name="step_100")
# OR
sc = SamplingClient(model_path="tinker://...")

response = sc.sample(
    prompt=model_input,
    num_samples=4,
    sampling_params=SamplingParams(max_tokens=256, temperature=1.0),
)

for seq in response.sequences:
    print(seq.tokens, seq.logprobs, seq.stop_reason)
```

### Key methods
- `sample(prompt, num_samples, sampling_params)` — Generate completions
- `compute_logprobs(prompt)` — Get logprobs for existing tokens
- `get_tokenizer()` — Get the model's tokenizer

**Important:** Always create a **new** SamplingClient after saving weights. A stale client points at old weights.

## Common patterns

### Pipelined training loop
```python
fb_future = tc.forward_backward_async(data=batch, loss_fn="cross_entropy")
# Prepare next batch while GPU works...
next_batch = prepare_batch(...)
fb_result = fb_future.result()

optim_future = tc.optim_step_async(adam_params=adam_params)
# Prepare more data...
optim_result = optim_future.result()
```

### Save and sample
```python
sc = tc.save_weights_and_get_sampling_client(name=f"step_{step}")
response = sc.sample(prompt=prompt, num_samples=4, sampling_params=params)
```

## Common pitfalls
- Always await futures before submitting new forward_backward calls
- Submit forward_backward_async + optim_step_async back-to-back before awaiting
- Create a **new** SamplingClient after saving weights (sampler desync)
- Use `save_state` for resumable checkpoints, `save_weights_for_sampler` for sampling-only
