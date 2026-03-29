# Checkpointing

Complete reference for checkpoint management.

## Reference

- `tinker_cookbook/checkpoint_utils.py` — CheckpointRecord, save/load helpers

## Two checkpoint types

| Type | Method | Purpose | Contains |
|------|--------|---------|----------|
| **State** | `save_state()` | Resume training | Weights + optimizer state |
| **Sampler** | `save_weights_for_sampler()` | Sampling / export | Weights only |

```python
tc.save_state(name="step_100", ttl_seconds=None)
tc.save_weights_for_sampler(name="step_100_sampler", ttl_seconds=None)
sc = tc.save_weights_and_get_sampling_client()  # Ephemeral, not persistently saved
```

## CheckpointRecord

```python
from tinker_cookbook.checkpoint_utils import CheckpointRecord

record = CheckpointRecord(
    name="step_100", batch=100, epoch=1, final=False,
    state_path="tinker://...", sampler_path="tinker://...",
    extra={"eval_loss": 0.5},
)
d = record.to_dict()
record = CheckpointRecord.from_dict(d)
record.has("state_path")  # True
```

## Save/load helpers

```python
from tinker_cookbook import checkpoint_utils

# Save (async)
paths = await checkpoint_utils.save_checkpoint_async(
    training_client=tc, name="step_100", log_path="/tmp/my_run",
    loop_state={"batch": 100, "epoch": 1},
    kind="both",  # "state", "sampler", or "both"
    ttl_seconds=None,
)

# Load checkpoint list
records = checkpoint_utils.load_checkpoints_file("/tmp/my_run")

# Get last checkpoint
record = checkpoint_utils.get_last_checkpoint("/tmp/my_run", required_key="state_path")
```

## Resuming training

```python
behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"  # "ask", "delete", "resume"

if config.load_checkpoint_path:
    tc.load_state_with_optimizer(config.load_checkpoint_path)
```

## REST API / CLI management

```python
rest = ServiceClient().create_rest_client()
checkpoints = rest.list_user_checkpoints(limit=100)
rest.publish_checkpoint_from_tinker_path("tinker://...")
rest.set_checkpoint_ttl_from_tinker_path("tinker://...", ttl_seconds=86400)
rest.delete_checkpoint_from_tinker_path("tinker://...")
```

```bash
tinker checkpoint list
tinker checkpoint publish <TINKER_PATH>
tinker checkpoint set-ttl <TINKER_PATH> --ttl 86400
tinker checkpoint delete <TINKER_PATH>
```
