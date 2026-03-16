---
name: logging
description: Guide for training outputs, metrics logging, logtree reports, and debugging training runs. Use when the user asks about training logs, metrics, debugging, analyzing training runs, or understanding training output files.
---

# Logging & Debugging

Every training run writes structured outputs to `log_path`. This skill covers what's produced and how to use it.

## Reference

- `docs/rl/rl-logging.mdx` — Complete file reference for RL training outputs
- `tinker_cookbook/utils/ml_log.py` — Metrics logging API
- `tinker_cookbook/utils/logtree.py` — Logtree (structured rollout transcripts)

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

**Timing:**
- `time/...` — wall-clock timings for different stages

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

Contains full text of prompts, model responses, grading details:

```python
import json

def find_conversations(node):
    results = []
    if isinstance(node, dict):
        if node.get("data", {}).get("type") == "conversation":
            results.append(node["data"])
        for child in node.get("children", []):
            if isinstance(child, dict):
                results.extend(find_conversations(child))
    return results

with open("train_iteration_000010_logtree.json") as f:
    trace = json.load(f)

for conv in find_conversations(trace["root"]):
    for msg in conv["messages"]:
        print(f"{msg['role']}: {msg['content'][:100]}")
```

### HTML reports

Open `train_iteration_NNNNNN.html` in a browser for a human-readable view of rollouts with collapsible sections.

`num_groups_to_log` (default: 4) controls how many trajectory groups get detailed logging.

## Logging in your own code

### Scalar metrics

```python
from tinker_cookbook.utils.ml_log import log_metrics

log_metrics({"train/loss": 0.5, "eval/accuracy": 0.85}, step=100)
```

### Logtree (structured transcripts)

```python
from tinker_cookbook.utils.logtree import scope

with scope("my_section"):
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

## Debugging tips

1. **Training not improving**: Check `metrics.jsonl` — is loss decreasing? Are rewards increasing?
2. **KL divergence spiking**: KL > 0.01 indicates instability. Lower the learning rate.
3. **Reward stuck at 0**: Check rollout summaries — are responses being parsed correctly?
4. **OOM / timeout**: Reduce `batch_size`, `group_size`, or `max_tokens`
5. **Shrink workloads for debugging**: Set small `batch_size`, `group_size`, and `max_steps`
6. **Compare runs**: Load multiple `metrics.jsonl` into a DataFrame and overlay plots
