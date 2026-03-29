---
name: tinker-ops
description: Training lifecycle operations — checkpointing, weight export, logging/metrics analysis, and evaluation. Use when the user asks about saving/loading checkpoints, exporting weights to HuggingFace, reading training logs, parsing metrics, debugging training runs, evaluating models, or understanding training outputs.
---

# Training Operations

Checkpointing, weight export, logging, and evaluation.

## Checkpointing

Two checkpoint types:

| Type | Method | Purpose | Contains |
|------|--------|---------|----------|
| **State** | `save_state()` | Resume training | Weights + optimizer |
| **Sampler** | `save_weights_for_sampler()` | Sampling / export | Weights only |

```python
# Save
tc.save_state(name="step_100", ttl_seconds=None)
tc.save_weights_for_sampler(name="step_100_sampler", ttl_seconds=None)
sc = tc.save_weights_and_get_sampling_client()  # Ephemeral, not persisted

# Resume
tc.load_state_with_optimizer(path="tinker://...")
```

### CheckpointRecord

```python
from tinker_cookbook import checkpoint_utils

# Save with helpers
paths = await checkpoint_utils.save_checkpoint_async(
    training_client=tc, name="step_100", log_path="/tmp/my_run",
    loop_state={"batch": 100, "epoch": 1}, kind="both",
)

# Load
record = checkpoint_utils.get_last_checkpoint("/tmp/my_run", required_key="state_path")
```

For the full checkpoint API and resume patterns, read `references/checkpoints.md`.

## Weight export

Download, merge LoRA, and publish to HuggingFace:

```python
from tinker_cookbook import weights

# Download adapter
adapter_dir = weights.download(tinker_path="tinker://run-id/sampler_weights/final", output_dir="./adapter")

# Merge LoRA into base model
weights.build_hf_model(base_model="Qwen/Qwen3-8B", adapter_path=adapter_dir,
                       output_path="./model", dtype="bfloat16")

# Or build a PEFT adapter (for vLLM/SGLang serving)
weights.build_lora_adapter(base_model="Qwen/Qwen3-8B", adapter_path=adapter_dir,
                           output_path="./peft_adapter")

# Publish to HuggingFace Hub
url = weights.publish_to_hf_hub(model_path="./model", repo_id="user/my-model", private=True)
```

For the complete weight lifecycle API, read `references/weights.md`.

## Logging & metrics

Every training run writes to `log_path`:

| File | Contents |
|------|----------|
| `metrics.jsonl` | Scalar metrics per iteration |
| `config.json` | Full training config |
| `checkpoints.jsonl` | Checkpoint metadata |
| `iteration_NNNNNN/train.html` | Human-readable rollout report |
| `iteration_NNNNNN/train_logtree.json` | Machine-readable rollout transcripts |
| `iteration_NNNNNN/train_rollout_summaries.jsonl` | Per-trajectory rewards |

### Analyzing metrics

```python
import pandas as pd
df = pd.read_json("path/to/metrics.jsonl", lines=True)
df.plot(x="progress/batch", y="env/all/reward/total")
```

**Common metric keys:**
- `progress/batch`, `progress/done_frac` — progress
- `env/all/reward/total` — mean reward (RL)
- `entropy`, `kl_sample_train_v1` — training health (KL should stay < 0.01)
- `optim/lr` — current learning rate

### Debugging tips

1. **Not improving**: Check `metrics.jsonl` — is loss decreasing? Rewards increasing?
2. **KL spiking**: KL > 0.01 = instability. Lower learning rate.
3. **Reward stuck at 0**: Check rollout summaries — are responses parsed correctly?
4. **OOM / timeout**: Reduce `batch_size`, `group_size`, or `max_tokens`

For the full logging reference (logtree parsing, Gantt charts, tracing/profiling), read `references/logging.md`.

## Evaluation

Training scripts support inline evaluation at configurable intervals.

### SL evaluators

```python
config = supervised_train.Config(
    evaluator_builders=[...],              # Every eval_every steps
    infrequent_evaluator_builders=[...],   # Every infrequent_eval_every steps
    eval_every=8, infrequent_eval_every=50,
)
```

### RL evaluators

```python
async def my_evaluator(sampling_client) -> dict[str, float]:
    return {"accuracy": 0.85}

config = rl_train.Config(evaluator_builders=[my_evaluator], eval_every=20)
```

### Inspect AI integration

```python
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

evaluator = InspectAPIFromTinkerSampling(
    task="gsm8k", renderer_name=renderer_name,
    model_name=model_name, include_reasoning=True,
)
```

For custom evaluator patterns, read `references/evals.md`.

## Async patterns (important for throughput)

Checkpoint saves and evaluations should use async to avoid blocking the training loop:

```python
# CORRECT: async checkpoint save — training continues while saving
save_future = checkpoint_utils.save_checkpoint_async(
    training_client=tc, name="step_100", log_path=log_path,
    loop_state={"batch": 100}, kind="both",
)
# ... continue training or prepare next batch ...
paths = await save_future

# For evaluation, run test samples concurrently:
import asyncio
tasks = [evaluate_problem(sc, p) for p in test_problems]
results = await asyncio.gather(*tasks)
# NOT: sequential one-by-one evaluation
```

## Common pitfalls

- **Sequential evaluation**: Run eval samples concurrently with `asyncio.gather()`, not in a sequential loop. Sequential evaluation wastes the parallelism Tinker provides.
- **Sampler desync**: After saving weights, create a **new** SamplingClient. A stale client silently samples from old weights.
- Use `save_state` for resuming, `save_weights_for_sampler` for export
- `download()` expects sampler weights, not state checkpoints
- Set `HF_TOKEN` for private models and publishing
- Checkpoint paths start with `tinker://` — remote storage, not local
- Set `ttl_seconds` on intermediate checkpoints to avoid storage bloat
- Keep evaluators fast to avoid stalling the training loop

## Code references

- `tinker_cookbook/checkpoint_utils.py` — CheckpointRecord, save/load helpers
- `tinker_cookbook/weights/` — Download, merge, publish pipeline
- `tinker_cookbook/utils/ml_log.py` — Metrics logging
- `tinker_cookbook/utils/logtree.py` — Structured rollout transcripts
- `tinker_cookbook/utils/trace.py` — Tracing/profiling
- `tinker_cookbook/eval/evaluators.py` — Evaluator types
