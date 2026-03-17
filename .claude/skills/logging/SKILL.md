---
name: logging
description: Guide for training outputs, metrics logging, logtree reports, tracing/profiling, and debugging training runs. Use when the user asks about training logs, metrics, debugging, tracing, profiling, timing, Gantt charts, or understanding training output files.
---

# Logging & Debugging

Every training run writes structured outputs to `log_path`. This skill covers what's produced and how to use it.

## Reference

- `docs/rl/rl-logging.mdx` — Complete file reference for RL training outputs
- `tinker_cookbook/utils/ml_log.py` — Metrics logging API
- `tinker_cookbook/utils/logtree.py` — Logtree (structured rollout transcripts)
- `tinker_cookbook/utils/trace.py` — Tracing/profiling (`@scope`, `trace_iteration`, Gantt charts)

## Output files

Each training run writes to its `log_path` directory:

| File | Format | Contents |
|------|--------|----------|
| `metrics.jsonl` | JSONL | Scalar metrics per training iteration |
| `config.json` | JSON | Full serialized training config (reproducibility) |
| `checkpoints.jsonl` | JSONL | Checkpoint metadata (paths, loop state for resume) |
| `code.diff` | text | Git diff at training start |
| `train_iteration_NNNNNN.html` | HTML | Human-readable logtree report |
| `train_iteration_NNNNNN_logtree.json` | JSON | Machine-readable rollout transcripts |
| `train_iteration_NNNNNN_rollout_summaries.jsonl` | JSONL | Per-trajectory rewards and metrics |
| `eval_<name>_iteration_NNNNNN.*` | mixed | Same formats for eval rollouts |
| `timing_spans.jsonl` | JSONL | Per-iteration span timing data (from `trace_iteration`) |
| `trace_events.jsonl` | JSONL | Perfetto/Chrome Trace format events (from `trace_init`) |
| `gantt_NNNNNN.html` | HTML | Plotly Gantt chart of span timeline (optional) |

Iteration numbers are zero-padded to 6 digits.

## Analyzing metrics

```python
import pandas as pd

df = pd.read_json("path/to/log_path/metrics.jsonl", lines=True)
df.plot(x="progress/batch", y="env/all/reward/total")
```

### Common metric keys

**Progress:**
- `progress/batch` — iteration index
- `progress/done_frac` — completion fraction

**RL rewards:**
- `env/all/reward/total` — mean total reward
- `env/all/<metric>` — env-emitted metrics (e.g., `correct`, `format_parse`)

**Training health:**
- `entropy` — per-token entropy
- `kl_sample_train_v1`, `kl_sample_train_v2` — KL divergence (should stay < 0.01)
- `optim/lr` — current learning rate
- `ac_tokens_per_turn` — mean generated tokens per turn

**Timing** (from `trace_iteration`):
- `time/total` — iteration wall-clock duration
- `time/<name>` — single-call duration (e.g., `time/train_step`)
- `time/<name>:total`, `time/<name>:count`, `time/<name>:mean`, `time/<name>:max` — aggregates for functions called multiple times (e.g., `time/sample_async:total`)

## Analyzing rollouts

### Rollout summaries (aggregate)

```python
import json

with open("train_iteration_000010_rollout_summaries.jsonl") as f:
    trajectories = [json.loads(line) for line in f]

for traj in trajectories:
    print(f"reward={traj['total_reward']:.2f}, metrics={traj['trajectory_metrics']}")
    # Each trajectory has: total_reward, final_reward, trajectory_metrics,
    # steps (list of {ob_len, ac_len, reward, episode_done, metrics})
```

### Logtree JSON (full transcripts)

Contains full text of prompts, model responses, grading details. Walk the tree recursively looking for nodes with `data.type == "conversation"` to extract conversations. See `docs/rl/rl-logging.mdx` for the full schema.

### HTML reports

Open `train_iteration_NNNNNN.html` in a browser for a human-readable view of rollouts with collapsible sections. `num_groups_to_log` (default: 4) controls how many trajectory groups get detailed logging.

## Logging in your own code

### Scalar metrics

```python
from tinker_cookbook.utils import ml_log

# Set up logging (done once in training scripts)
ml_logger = ml_log.setup_logging(log_path="/tmp/my_run", wandb_project=None, wandb_name=None)

# Log scalar metrics
ml_logger.log_metrics({"train/loss": 0.5, "eval/accuracy": 0.85}, step=100)
```

### Logtree (structured transcripts)

```python
from tinker_cookbook.utils import logtree

with logtree.scope_header("my_section"):
    # Nested logging of rollouts, grading, etc.
    ...
```

## Weights & Biases integration

Pass `wandb_project` and `wandb_name` in your config to enable W&B logging:

```python
config = train.Config(
    wandb_project="my-project",
    wandb_name="my-experiment",
    ...
)
```

## Tracing & profiling

The `tinker_cookbook/utils/trace` module provides per-iteration profiling across all training modules (RL, SL, DPO, distillation).

### Core API

```python
from tinker_cookbook.utils import trace

# Initialize Perfetto trace collector (optional — writes trace_events.jsonl)
trace.trace_init()

# In training loop — collect per-iteration timing
for i_batch in range(n_batches):
    with trace.trace_iteration(step=i_batch) as window:
        # All @scope-decorated calls are automatically recorded
        await gather_rollouts(...)
        await train_step(...)

    # Get timing metrics for this iteration
    metrics.update(window.get_timing_metrics())

    # Persist span data for post-hoc analysis
    window.write_spans_jsonl(log_path / "timing_spans.jsonl", step=i_batch)

    # Optional: Gantt chart visualization (requires plotly)
    trace.save_gantt_chart_html(window, i_batch, log_path / f"gantt_{i_batch}.html")
```

### Instrumenting your code

```python
from tinker_cookbook.utils import trace

# Decorator — automatically traces function calls
@trace.scope
async def my_training_step(tc, batch):
    result = await tc.forward_backward_async(data=batch, loss_fn="cross_entropy")
    return result

# Inline span — for timing a code block without a dedicated function
async with trace.scope_span("data_prep"):
    batch = prepare_next_batch(...)

# Sync variant
with trace.scope_span_sync("data_prep"):
    batch = prepare_next_batch(...)
```

`@scope` and `scope_span` are no-ops when called outside `trace_iteration` — safe to leave in production.

### Viewing Perfetto traces

```bash
# Convert JSONL to JSON for visualization
uv run python -m tinker_cookbook.utils.trace trace_events.jsonl trace.json
# Open trace.json in chrome://tracing or https://ui.perfetto.dev/
```

## Debugging tips

1. **Training not improving**: Check `metrics.jsonl` — is loss decreasing? Are rewards increasing?
2. **KL divergence spiking**: KL > 0.01 indicates instability. Lower the learning rate.
3. **Reward stuck at 0**: Check rollout summaries — are responses being parsed correctly?
4. **OOM / timeout**: Reduce `batch_size`, `group_size`, or `max_tokens`
5. **Shrink workloads for debugging**: Set small `batch_size`, `group_size`, and `max_steps`
6. **Compare runs**: Load multiple `metrics.jsonl` into a DataFrame and overlay plots
