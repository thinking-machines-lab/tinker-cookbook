---
name: checkpoints
description: Guide for checkpointing — saving, loading, and resuming training with CheckpointRecord. Use when the user asks about saving weights, resuming training, checkpoint management, or the checkpoint lifecycle.
---

# Checkpointing

Tinker supports two types of checkpoints and provides utilities for managing them during training.

## Reference

Read these for details:
- `tinker_cookbook/checkpoint_utils.py` — CheckpointRecord, save/load helpers
- `docs/save-load.mdx` — Checkpointing guide (save_weights_for_sampler vs save_state)

## Two checkpoint types

| Type | Method | Purpose | Contains |
|------|--------|---------|----------|
| **State** | `save_state()` | Resume training | Weights + optimizer state |
| **Sampler** | `save_weights_for_sampler()` | Sampling / export | Weights only |

```python
# Save full state (for resumption)
tc.save_state(name="step_100", ttl_seconds=None)

# Save sampler weights (for sampling/export)
tc.save_weights_for_sampler(name="step_100_sampler", ttl_seconds=None)

# Save both + get a SamplingClient
sc = tc.save_weights_and_get_sampling_client(name="step_100")
```

`ttl_seconds=None` means indefinite retention. Set a TTL for intermediate checkpoints to avoid storage bloat.

## CheckpointRecord

Typed dataclass for checkpoint bookkeeping:

```python
from tinker_cookbook.checkpoint_utils import CheckpointRecord

record = CheckpointRecord(
    name="step_100",
    batch=100,
    epoch=1,
    final=False,
    state_path="tinker://...",
    sampler_path="tinker://...",
    extra={"eval_loss": 0.5},  # User metadata
)

# Serialize
d = record.to_dict()

# Deserialize
record = CheckpointRecord.from_dict(d)

# Check if a field is set
record.has("state_path")  # True
```

## Save/load helpers

```python
from tinker_cookbook import checkpoint_utils

# Save checkpoint (async)
paths = await checkpoint_utils.save_checkpoint_async(
    training_client=tc,
    name="step_100",
    log_path="/tmp/my_run",
    loop_state={"batch": 100, "epoch": 1},
    kind="both",           # "state", "sampler", or "both"
    ttl_seconds=None,
)
# paths = {"state_path": "tinker://...", "sampler_path": "tinker://..."}

# Load checkpoint list
records = checkpoint_utils.load_checkpoints_file("/tmp/my_run")

# Get last checkpoint
record = checkpoint_utils.get_last_checkpoint(
    "/tmp/my_run",
    required_key="state_path",  # Only return records with this field
)
```

## Resuming training

The standard pattern (used by `supervised/train.py` and `rl/train.py`):

```python
# In CLIConfig
behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"  # "ask", "delete", "resume"

# In training loop
if config.load_checkpoint_path:
    tc.load_state_with_optimizer(config.load_checkpoint_path)
```

Set `behavior_if_log_dir_exists=resume` to continue from the last checkpoint in an existing log directory.

## Common pitfalls
- Use `save_state` for resumable checkpoints, `save_weights_for_sampler` for sampling/export
- `get_last_checkpoint()` returns `None` if no matching checkpoint exists — always check
- Checkpoint paths start with `tinker://` — they reference remote storage, not local files
- Set `ttl_seconds` on intermediate checkpoints to avoid accumulating old weights
- For RLHF pipelines, the SFT stage saves `state_path` (for RL init) and the RM stage saves `sampler_path` (for reward scoring)
