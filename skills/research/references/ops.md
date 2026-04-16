# Operations Reference — Checkpoints, Weights, Logging, Evaluation

Consolidated reference for training lifecycle operations: checkpoint management, weight export, logging/metrics analysis, and evaluation.

---

## Checkpoints

### Reference

- `tinker_cookbook/checkpoint_utils.py` — CheckpointRecord, save/load helpers

### Two checkpoint types

| Type | Method | Purpose | Contains |
|------|--------|---------|----------|
| **State** | `save_state()` | Resume training | Weights + optimizer state |
| **Sampler** | `save_weights_for_sampler()` | Sampling / export | Weights only |

```python
tc.save_state(name="step_100", ttl_seconds=None)
tc.save_weights_for_sampler(name="step_100_sampler", ttl_seconds=None)
sc = tc.save_weights_and_get_sampling_client()  # Ephemeral, not persistently saved
```

### CheckpointRecord

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

### Save/load helpers

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

### Resuming training

```python
behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"  # "ask", "delete", "resume"

if config.load_checkpoint_path:
    tc.load_state_with_optimizer(config.load_checkpoint_path)
```

### REST API / CLI management

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

---

## Weights

### Reference

- `tinker_cookbook/weights/__init__.py` — API overview
- `tinker_cookbook/weights/_download.py` — Download implementation
- `tinker_cookbook/weights/_export/` — LoRA merge (full, quantized, sharded)
- `tinker_cookbook/weights/_publish.py` — HuggingFace Hub publish

### Download

```python
from tinker_cookbook import weights

adapter_dir = weights.download(
    tinker_path="tinker://run-id/sampler_weights/final",
    output_dir="./adapter",
    base_url=None,
)
```

### Merge LoRA into base model

```python
weights.build_hf_model(
    base_model="Qwen/Qwen3-8B",
    adapter_path="./adapter",
    output_path="./model",
    dtype="bfloat16",
    trust_remote_code=None,
)
```

### PEFT adapter (no merge)

Convert to PEFT format for vLLM/SGLang serving:

```python
weights.build_lora_adapter(
    base_model="Qwen/Qwen3-8B",
    adapter_path="./adapter",
    output_path="./peft_adapter",
    trust_remote_code=None,
)
```

### Publish to HuggingFace Hub

```python
url = weights.publish_to_hf_hub(
    model_path="./model",
    repo_id="user/my-finetuned-model",
    private=True,
    token=None,  # Uses HF_TOKEN env var
)
```

### Full workflow

```python
from tinker_cookbook import weights

# Step 1: Download adapter
adapter_dir = weights.download(
    tinker_path="tinker://run-id/sampler_weights/final",
    output_dir="./adapter",
)

# Step 2: Merge LoRA into base model
weights.build_hf_model(
    base_model="Qwen/Qwen3.5-35B-A3B",
    adapter_path=adapter_dir,
    output_path="./model",
    dtype="bfloat16",
)

# Step 3: Publish to HuggingFace Hub
url = weights.publish_to_hf_hub(
    model_path="./model",
    repo_id="user/my-finetuned-model",
    private=True,
)
```

### Pitfalls

- `download()` expects `tinker://` path from `save_weights_for_sampler`, not `save_state`
- `build_hf_model()` requires the base model to be downloadable from HuggingFace
- Set `HF_TOKEN` for private models and publishing
- `dtype="bfloat16"` is recommended for most models

---

## Logging

### Reference

- `tinker_cookbook/utils/ml_log.py` — Metrics logging API
- `tinker_cookbook/utils/logtree.py` — Logtree structured transcripts
- `tinker_cookbook/utils/trace.py` — Tracing/profiling

### Output files

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

### Common metric keys

**Progress:** `progress/batch`, `progress/done_frac`
**RL rewards:** `env/all/reward/total`, `env/all/<metric>`
**Health:** `entropy`, `kl_sample_train_v1` (< 0.01), `optim/lr`
**Timing:** `time/total`, `time/<name>`, `time/<name>:total`, `time/<name>:mean`

### Rollout analysis

#### Rollout summaries

```python
import json
with open("iteration_000010/train_rollout_summaries.jsonl") as f:
    trajectories = [json.loads(line) for line in f]
for traj in trajectories:
    print(f"reward={traj['total_reward']:.2f}, metrics={traj['trajectory_metrics']}")
```

#### Logtree JSON (full transcripts)

```python
import json
with open("iteration_000060/train_logtree.json") as f:
    data = json.load(f)

groups = [c for c in data["root"]["children"]
          if isinstance(c, dict) and c.get("tag") == "section"]
```

#### Extracting conversations

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

#### Extracting tables

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

### Custom logging

```python
from tinker_cookbook.utils import ml_log
ml_logger = ml_log.setup_logging(log_path="/tmp/my_run", wandb_project=None, wandb_name=None)
ml_logger.log_metrics({"train/loss": 0.5, "eval/accuracy": 0.85}, step=100)
```

### Tracing & profiling

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

#### Instrumenting code

```python
@trace.scope
async def my_training_step(tc, batch):
    result = await tc.forward_backward_async(data=batch, loss_fn="cross_entropy")
    return result

async with trace.scope_span("data_prep"):
    batch = prepare_next_batch(...)
```

#### Viewing Perfetto traces

```bash
uv run python -m tinker_cookbook.utils.trace trace_events.jsonl trace.json
# Open in chrome://tracing or https://ui.perfetto.dev/
```

### Weights & Biases

```python
config = train.Config(wandb_project="my-project", wandb_name="my-experiment", ...)
```

---

## Evaluation

### Reference

- `tinker_cookbook/eval/evaluators.py` — Evaluator types
- `tinker_cookbook/eval/inspect_evaluators.py` — Inspect-based evaluators
- `tinker_cookbook/eval/custom_evaluators.py` — Custom evaluator implementations
- `tinker_cookbook/supervised/nll_evaluator.py` — NLL evaluator
- `tinker_cookbook/supervised/train.py` — SL evaluator integration
- `tinker_cookbook/rl/train.py` — RL evaluator integration

### SL evaluators

Two tiers:
```python
config = supervised_train.Config(
    evaluator_builders=[...],              # Every eval_every steps
    infrequent_evaluator_builders=[...],   # Every infrequent_eval_every steps
    eval_every=8,
    infrequent_eval_every=50,
)
```

### RL evaluators

Uses `SamplingClientEvaluator`:
```python
async def my_evaluator(sampling_client: SamplingClient) -> dict[str, float]:
    return {"accuracy": 0.85, "avg_length": 150}

config = rl_train.Config(evaluator_builders=[my_evaluator], eval_every=20)
```

### Test set evaluator

Built into `rl/train.py` via the test dataset from `RLDatasetBuilder.__call__()`:
```python
# RLDatasetBuilder.__call__() returns (train_dataset, test_dataset)
```

### Inspect AI integration

```python
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

evaluator = InspectAPIFromTinkerSampling(
    task="gsm8k", renderer_name=renderer_name,
    model_name=model_name, include_reasoning=True,
)
```

See `tinker_cookbook/recipes/chat_sl/train.py` for a working example with GSM8K and IFEval.

### Custom evaluators

#### Sampling-based

```python
async def eval_math(sampling_client: SamplingClient) -> dict[str, float]:
    async def evaluate_one(problem):
        response = await sampling_client.sample_async(
            prompt=problem.prompt, num_samples=1,
            sampling_params=SamplingParams(max_tokens=256, temperature=0.0),
        )
        return parse_answer(response.sequences[0].tokens) == problem.expected

    # Evaluate all problems concurrently — sequential loops waste throughput
    results = await asyncio.gather(*[evaluate_one(p) for p in test_problems])
    return {"math_accuracy": sum(results) / len(results)}
```

#### NLL-based

Compute NLL on a held-out dataset without generating text. See the built-in evaluator in `tinker_cookbook/supervised/train.py`.

### Metrics logging

```python
from tinker_cookbook.utils.ml_log import log_metrics
log_metrics({"train/loss": 0.5, "eval/accuracy": 0.85}, step=100)
```
