# Logging & Debugging

Complete reference for training outputs, metrics, logtree parsing, and tracing.

## Reference

- `tinker_cookbook/utils/ml_log.py` — Metrics logging API
- `tinker_cookbook/utils/logtree.py` — Logtree structured transcripts
- `tinker_cookbook/utils/trace.py` — Tracing/profiling

## Output files

**Top-level:**

| File | Format | Contents |
|------|--------|----------|
| `metrics.jsonl` | JSONL | Scalar metrics per iteration |
| `config.json` | JSON | Full training config |
| `checkpoints.jsonl` | JSONL | Checkpoint metadata |
| `code.diff` | text | Git diff at training start |
| `timing_spans.jsonl` | JSONL | Per-iteration span timing |
| `trace_events.jsonl` | JSONL | Perfetto/Chrome Trace events |

**Per-iteration** (inside `iteration_NNNNNN/`):

| File | Format | Contents |
|------|--------|----------|
| `train.html` | HTML | Human-readable logtree report |
| `train_logtree.json` | JSON | Machine-readable rollout transcripts |
| `train_rollout_summaries.jsonl` | JSONL | Per-trajectory rewards |
| `eval_<name>.html` | HTML | Eval rollout report |
| `eval_<name>_logtree.json` | JSON | Eval rollout transcripts |
| `timing_gantt.html` | HTML | Plotly Gantt chart |

## Common metric keys

**Progress:** `progress/batch`, `progress/done_frac`
**RL rewards:** `env/all/reward/total`, `env/all/<metric>`
**Health:** `entropy`, `kl_sample_train_v1` (< 0.01), `optim/lr`
**Timing:** `time/total`, `time/<name>`, `time/<name>:total`, `time/<name>:mean`

## Analyzing rollouts

### Rollout summaries

```python
import json
with open("iteration_000010/train_rollout_summaries.jsonl") as f:
    trajectories = [json.loads(line) for line in f]
for traj in trajectories:
    print(f"reward={traj['total_reward']:.2f}, metrics={traj['trajectory_metrics']}")
```

### Logtree JSON (full transcripts)

```python
import json
with open("iteration_000060/train_logtree.json") as f:
    data = json.load(f)

groups = [c for c in data["root"]["children"]
          if isinstance(c, dict) and c.get("tag") == "section"]
```

### Extracting conversations

```python
def find_conversations(node):
    results = []
    if isinstance(node, dict):
        data = node.get("data", {})
        if isinstance(data, dict) and data.get("type") == "conversation":
            results.append(data)
        for child in node.get("children", []):
            results.extend(find_conversations(child))
    return results
```

### Extracting tables

```python
def find_tables(node):
    results = []
    if isinstance(node, dict):
        if node.get("tag") == "table":
            results.append(node)
        for c in node.get("children", []):
            results.extend(find_tables(c))
    return results

def parse_table_rows(table_node):
    rows = []
    for part in table_node.get("children", []):
        if not isinstance(part, dict): continue
        if part.get("tag") in ("tbody", "thead"):
            for row in part.get("children", []):
                if isinstance(row, dict) and row.get("tag") == "tr":
                    cells = []
                    for cell in row.get("children", []):
                        if isinstance(cell, dict) and cell.get("tag") in ("td", "th"):
                            cells.append(get_text(cell).strip())
                    rows.append(cells)
    return rows

def get_text(node):
    if isinstance(node, str): return node
    return "".join(get_text(c) for c in node.get("children", []))
```

## Logging in your own code

```python
from tinker_cookbook.utils import ml_log
ml_logger = ml_log.setup_logging(log_path="/tmp/my_run", wandb_project=None, wandb_name=None)
ml_logger.log_metrics({"train/loss": 0.5, "eval/accuracy": 0.85}, step=100)
```

## Tracing & profiling

```python
from tinker_cookbook.utils import trace

trace.trace_init()

for i_batch in range(n_batches):
    with trace.trace_iteration(step=i_batch) as window:
        await gather_rollouts(...)
        await train_step(...)
    metrics.update(window.get_timing_metrics())
    window.write_spans_jsonl(log_path / "timing_spans.jsonl", step=i_batch)
```

### Instrumenting code

```python
@trace.scope
async def my_training_step(tc, batch):
    result = await tc.forward_backward_async(data=batch, loss_fn="cross_entropy")
    return result

async with trace.scope_span("data_prep"):
    batch = prepare_next_batch(...)
```

### Viewing Perfetto traces

```bash
uv run python -m tinker_cookbook.utils.trace trace_events.jsonl trace.json
# Open in chrome://tracing or https://ui.perfetto.dev/
```

## Weights & Biases

```python
config = train.Config(wandb_project="my-project", wandb_name="my-experiment", ...)
```
