# Checkpoint Resume and Optimizer State in Tinker

A guide to saving, loading, and resuming training in Tinker with correct optimizer state handling. Written in response to a customer question about training instability in the first ~10 steps after resuming from a checkpoint.

## The problem

When you restart training from saved weights but **don't restore optimizer state**, Adam's momentum (`m_t`) and variance (`v_t`) buffers start from zero. This means:

1. **Adam effectively becomes SGD for the first few steps** - the moving averages haven't accumulated yet, so the adaptive learning rate estimates are noisy
2. **The bias correction terms amplify early gradients** - Adam's bias correction divides by `(1 - beta^t)`, which is small when `t` is small, producing large effective step sizes
3. **The LR schedule may jump** - if you recompute the LR schedule from step 0 instead of the resumed step, you get the wrong learning rate

The fix is simple: use `create_training_client_from_state_with_optimizer_async` instead of `create_training_client_from_state_async` when resuming an interrupted run.

## Two ways to load a checkpoint

Tinker exposes two distinct checkpoint-loading APIs. Picking the wrong one is the most common cause of post-resume instability.

### 1. Resume training (weights + optimizer state)

```python
training_client = await service_client.create_training_client_from_state_with_optimizer_async(
    state_path,                    # tinker:// URI from save_state()
    user_metadata=user_metadata,
)
```

This restores:
- Model weights (LoRA adapter parameters)
- Adam optimizer state: first-moment estimate `m_t` and second-moment estimate `v_t` for every parameter
- Optimizer step counter `t` (critical for bias correction)

Use this when: **resuming an interrupted run** - same hyperparameters, same dataset, same schedule. The training should continue as if the interruption never happened.

Source: SFT training does this at

https://github.com/thinking-machines-lab/tinker-cookbook/blob/2856586/tinker_cookbook/supervised/train.py#L295-L305

RL training does this at

https://github.com/thinking-machines-lab/tinker-cookbook/blob/2856586/tinker_cookbook/rl/train.py#L1913-L1923

### 2. Transfer learning (weights only, fresh optimizer)

```python
training_client = await service_client.create_training_client_from_state_async(
    checkpoint_path,               # tinker:// URI from save_state() or save_weights_for_sampler()
    user_metadata=user_metadata,
)
```

This restores:
- Model weights only

The optimizer starts fresh (zero momentum, zero variance, step counter `t=0`).

Use this when: **starting a new training run from a pretrained checkpoint** - different LR, different dataset, different task. You *want* the optimizer to adapt to the new loss landscape from scratch.

Source: SFT training does this at

https://github.com/thinking-machines-lab/tinker-cookbook/blob/2856586/tinker_cookbook/supervised/train.py#L306-L314

RL training does this at

https://github.com/thinking-machines-lab/tinker-cookbook/blob/2856586/tinker_cookbook/rl/train.py#L1924-L1932

## Saving checkpoints

### Two types of saves

| Method | What it saves | Returns | When to use |
|--------|--------------|---------|-------------|
| `save_state_async(name)` | Weights + optimizer state | `tinker://` state path | Resumable training checkpoints |
| `save_weights_for_sampler_async(name)` | Weights only | `tinker://` sampler path | Inference, export, weight merging |

The cookbook's `save_checkpoint_async()` helper wraps both. The `kind` parameter controls which types are saved:

```python
from tinker_cookbook import checkpoint_utils

# Save both types (standard for periodic checkpoints)
paths = await checkpoint_utils.save_checkpoint_async(
    training_client=training_client,
    name=f"{step:06d}",           # e.g. "000100"
    log_path="./logs/my-run",
    loop_state={"epoch": 0, "batch": 100},
    kind="both",                   # "state", "sampler", or "both"
    ttl_seconds=604800,            # 7 days; None = keep forever
)
# paths = {"state_path": "tinker://...", "sampler_path": "tinker://..."}
```

Source:

https://github.com/thinking-machines-lab/tinker-cookbook/blob/2856586/tinker_cookbook/checkpoint_utils.py#L431-L475

### Checkpoint metadata

Every save appends a record to `checkpoints.jsonl` in your `log_path`:

```json
{"name": "000100", "batch": 100, "epoch": 0, "state_path": "tinker://run-abc/state/000100", "sampler_path": "tinker://run-abc/sampler/000100"}
```

On resume, `get_last_checkpoint()` reads this file and returns the most recent record that has a `state_path`:

```python
resume_info = checkpoint_utils.get_last_checkpoint(log_path)
# resume_info.state_path -> "tinker://run-abc/state/000100"
# resume_info.batch -> 100
# resume_info.epoch -> 0
```

Source:

https://github.com/thinking-machines-lab/tinker-cookbook/blob/2856586/tinker_cookbook/checkpoint_utils.py#L404-L427

## Minimal end-to-end example

Here's a self-contained snippet showing the correct save/resume pattern:

```python
import asyncio
import tinker
from tinker_cookbook import checkpoint_utils


async def train_with_checkpoints():
    service_client = tinker.ServiceClient()

    # --- Check for existing checkpoint ---
    log_path = "./logs/my-run"
    resume_info = checkpoint_utils.get_last_checkpoint(log_path)

    if resume_info:
        # RESUME: load weights + optimizer state
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info.state_path
            )
        )
        start_step = resume_info.batch
        print(f"Resumed from step {start_step}, state: {resume_info.state_path}")
    else:
        # FRESH: create new training client
        training_client = await service_client.create_lora_training_client_async(
            base_model="Qwen/Qwen3-8B", rank=32
        )
        start_step = 0

    # --- Training loop ---
    total_steps = 1000
    save_every = 100

    for step in range(start_step, total_steps):
        # Compute LR from the ACTUAL step (not from 0)
        lr_multiplier = 1 - step / total_steps  # linear decay
        learning_rate = 1e-4 * lr_multiplier

        adam_params = tinker.AdamParams(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )

        # ... build your batch of Datum objects here ...
        fb_future = await training_client.forward_backward_async(batch, loss_fn="cross_entropy")
        optim_future = await training_client.optim_step_async(adam_params)

        # Await results
        fb_result = await fb_future.result_async()
        await optim_future.result_async()

        # --- Periodic checkpoint ---
        if save_every > 0 and step > 0 and step % save_every == 0:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=log_path,
                loop_state={"batch": step},
                kind="both",
                ttl_seconds=604800,  # 7 days
            )
            print(f"Checkpoint saved at step {step}")

    # --- Final checkpoint (kept forever) ---
    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=log_path,
        loop_state={"batch": total_steps, "final": True},
        kind="both",
        ttl_seconds=None,  # keep forever
    )


asyncio.run(train_with_checkpoints())
```

## Rolling checkpoints (cheap resume points)

For long training runs, periodic checkpoints may be too infrequent for comfortable resume. The `RollingCheckpointManager` saves lightweight state-only checkpoints at a finer cadence and automatically deletes the previous one to bound storage:

```python
rolling_mgr = checkpoint_utils.RollingCheckpointManager(
    training_client=training_client,
    service_client=service_client,
    log_path=log_path,
    rolling_save_every=10,          # save every 10 steps
    save_every=100,                 # skip when periodic checkpoint fires on same step
    rolling_ttl_seconds=7200,       # 2-hour safety TTL
)

for step in range(start_step, total_steps):
    # ... training step ...

    await rolling_mgr.maybe_save_async(
        step=step,
        loop_state={"batch": step},
    )

# Clean up after final periodic checkpoint
await rolling_mgr.finalize_async()
```

Rolling checkpoints save training state only (no sampler export), so they're cheaper but can only be used for resumption - not for inference.

Source:

https://github.com/thinking-machines-lab/tinker-cookbook/blob/2856586/tinker_cookbook/checkpoint_utils.py#L524-L676

## LR schedule on resume

The cookbook's LR schedule is computed purely from `(step, total_steps)` - there's no scheduler state to save. The key thing to get right is **computing LR from the actual resumed step**, not from step 0.

The SFT training loop does this correctly because it passes the real `step` (which accounts for `start_batch` from the checkpoint):

```python
# From supervised/train.py
step = epoch_idx * n_batches + batch_idx  # batch_idx starts from start_batch on resume

learning_rate = config.learning_rate * compute_schedule_lr_multiplier(
    lr_schedule=config.lr_schedule,
    step=step,                   # actual step, not relative to resume
    total_steps=total_steps,
)
```

If you're writing a custom loop, make sure your LR function receives the global step.

Source:

https://github.com/thinking-machines-lab/tinker-cookbook/blob/2856586/tinker_cookbook/supervised/train.py#L357-L362

https://github.com/thinking-machines-lab/tinker-cookbook/blob/2856586/tinker_cookbook/utils/lr_scheduling.py#L14-L44

## The resume test

The cookbook includes an integration test that validates resume correctness by checking loss reproducibility:

1. Train to step 8, interrupt (checkpoint saved at step 5)
2. Resume from step 5, train to step 8 again
3. Assert losses at steps 5-7 match within 1% between runs

This test confirms optimizer state is properly restored - if momentum/variance were reset, the losses would diverge.

Source:

https://github.com/thinking-machines-lab/tinker-cookbook/blob/2856586/tinker_cookbook/supervised/resume_test.py#L53-L157

## Diagnosing post-resume instability

If you see instability in the first ~10 steps after resuming, check these things in order:

### 1. Are you loading optimizer state? (most common cause)

Wrong (fresh optimizer):
```python
training_client = await service_client.create_training_client_from_state_async(state_path)
```

Right (with optimizer):
```python
training_client = await service_client.create_training_client_from_state_with_optimizer_async(state_path)
```

### 2. Is your checkpoint a `state` checkpoint or a `sampler` checkpoint?

Only `save_state_async()` saves optimizer state. If you saved with `save_weights_for_sampler_async()`, there's no optimizer state to restore - you'll get a fresh optimizer regardless of which load API you call.

Check your `checkpoints.jsonl` - records with a `state_path` field have optimizer state. Records with only `sampler_path` don't.

### 3. Is your LR schedule computed from the correct step?

If you're computing LR from step 0 instead of the resumed step, you'll get a different (likely higher) learning rate for the first few steps, which looks like instability.

### 4. Is your data order the same?

If your dataset is shuffled differently on resume, the first few batches will be different data - the loss will look different even if the optimizer is correct. The cookbook's SFT loop uses deterministic batch ordering (indexed by `batch_idx`) so resume produces identical data.

### 5. Are you using the same adam hyperparameters?

If `beta1`, `beta2`, or `eps` changed between the original run and the resumed run, the optimizer state from the checkpoint is paired with different hyperparameters, which can cause a transient.

### 6. (RL-specific) Is your sampler using the right weights?

After resuming an RL run, the sampling client must also be updated to use the checkpoint's weights. If the sampler still has old/base weights while the trainer has resumed weights, the rollouts will be off-distribution and the first few gradient steps will be noisy.

## Summary table

| Scenario | Load API | Optimizer | LR schedule | Expected behavior |
|----------|----------|-----------|-------------|-------------------|
| Resume interrupted SFT | `create_training_client_from_state_with_optimizer_async` | Restored | From actual step | Seamless, matches pre-interrupt trajectory |
| Resume interrupted RL | `create_training_client_from_state_with_optimizer_async` | Restored | From actual step | Seamless |
| Fine-tune from SFT checkpoint | `create_training_client_from_state_async` | Fresh | From step 0 | Normal warmup behavior |
| New RL from SFT checkpoint | `create_training_client_from_state_async` | Fresh | From step 0 | Normal warmup behavior |
| Load sampler weights for eval | `create_sampling_client_async(model_path=...)` | N/A | N/A | Inference only |

## Key source files

- Checkpoint utilities: `tinker_cookbook/checkpoint_utils.py`
- SFT training loop: `tinker_cookbook/supervised/train.py`
- RL training loop: `tinker_cookbook/rl/train.py`
- DPO training loop: `tinker_cookbook/preference/train_dpo.py`
- LR scheduling: `tinker_cookbook/utils/lr_scheduling.py`
- Resume test: `tinker_cookbook/supervised/resume_test.py`
- Weights tutorial (204): `tutorials/204_weights.py`
