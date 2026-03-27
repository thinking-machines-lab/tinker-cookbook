---
name: tinker-logging
description: Guide for training outputs, metrics logging, logtree reports, tracing/profiling, parsing RL logs, and debugging training runs. Use when the user asks about training logs, metrics, reading/parsing RL logs, logtree JSON structure, extracting rollout data, debugging, tracing, profiling, timing, Gantt charts, or understanding training output files.
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

**Top-level files:**

| File | Format | Contents |
|------|--------|----------|
| `metrics.jsonl` | JSONL | Scalar metrics per training iteration |
| `config.json` | JSON | Full serialized training config (reproducibility) |
| `checkpoints.jsonl` | JSONL | Checkpoint metadata (paths, loop state for resume) |
| `code.diff` | text | Git diff at training start |
| `timing_spans.jsonl` | JSONL | Per-iteration span timing data (from `trace_iteration`) |
| `trace_events.jsonl` | JSONL | Perfetto/Chrome Trace format events (from `trace_init`) |

**Per-iteration files** (inside `iteration_NNNNNN/` subdirectories):

| File | Format | Contents |
|------|--------|----------|
| `train.html` | HTML | Human-readable logtree report |
| `train_logtree.json` | JSON | Machine-readable rollout transcripts |
| `train_rollout_summaries.jsonl` | JSONL | Per-trajectory rewards and metrics |
| `eval_<name>.html` | HTML | Logtree report for eval rollouts |
| `eval_<name>_logtree.json` | JSON | Machine-readable eval rollout transcripts |
| `eval_<name>_rollout_summaries.jsonl` | JSONL | Per-trajectory eval data |
| `timing_gantt.html` | HTML | Plotly Gantt chart of span timeline (optional) |

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

with open("iteration_000010/train_rollout_summaries.jsonl") as f:
    trajectories = [json.loads(line) for line in f]

for traj in trajectories:
    print(f"reward={traj['total_reward']:.2f}, metrics={traj['trajectory_metrics']}")
    # Each trajectory has: total_reward, final_reward, trajectory_metrics,
    # steps (list of {ob_len, ac_len, reward, episode_done, metrics})
```

### Logtree JSON (full transcripts)

Contains full text of prompts, model responses, grading details. Top level:
```json
{"title": "...", "started_at": "...", "path": "...", "root": {node}}
```

Each node: `{tag, attrs, children, data?}` where children are either text strings or nested nodes.

#### Groups

The root's children include title/subtitle elements plus **group sections** (one per trajectory group):

```python
import json

with open("iteration_000060/train_logtree.json") as f:
    data = json.load(f)

groups = [c for c in data["root"]["children"]
          if isinstance(c, dict) and c.get("tag") == "section"]
```

`num_groups_to_log` (default 4) controls how many groups get full rollout content. Groups beyond this limit only have `Trajectory Details` (numeric stats). The specific sections within each group depend on the environment — see your project's skill docs for the layout.

#### Extracting conversations

Conversations are stored in `data` fields on nodes:
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

# Each conversation: {"type": "conversation", "messages": [...]}
# Each message: {"role": "user"|"assistant"|"system", "content": str | list[part]}
# Content parts: {"type": "text", "text": "..."} or {"type": "thinking", "thinking": "..."}
```

#### Extracting section content by title

```python
def get_section_title(section_node):
    h = section_node.get("children", [{}])[0]
    if isinstance(h, dict) and h.get("children"):
        return h["children"][0]
    return ""

def get_section_body(section_node):
    children = section_node.get("children", [])
    return children[1] if len(children) > 1 else {"children": []}
```

#### Extracting tables (rubric criteria, grading scores)

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
    """Returns list of rows, each row a list of cell text strings."""
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
    """Recursively extract text from a node tree."""
    if isinstance(node, str): return node
    return "".join(get_text(c) for c in node.get("children", []))
```

#### Tips

- Eval logtrees (`eval_*_logtree.json` inside `iteration_NNNNNN/`) have the same structure. Eval prompts are fixed across iterations, making them ideal for tracking policy changes over time.
- Training prompts are randomly sampled per iteration but use the same dataset and seed ordering, so they match across runs at the same iteration.

### HTML reports

Open `iteration_NNNNNN/train.html` in a browser for a human-readable view of rollouts with collapsible sections. `num_groups_to_log` (default: 4) controls how many trajectory groups get detailed logging.

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
    iter_dir = iteration_dir(log_path, i_batch)  # from tinker_cookbook.utils.misc_utils
    if iter_dir is not None:
        trace.save_gantt_chart_html(window, i_batch, iter_dir / "timing_gantt.html")
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
