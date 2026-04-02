# Evaluation Framework

Three layers for evaluating models trained with Tinker.

## 1. Evaluators — Inline Training Eval

Lightweight interfaces called every N training steps. Return `dict[str, float]` metrics.

```python
from tinker_cookbook.eval import SamplingClientEvaluator

class MyEval(SamplingClientEvaluator):
    async def __call__(self, sampling_client):
        # generate, grade, return metrics
        return {"eval/accuracy": 0.85}
```

Pass evaluators to your training loop via `evaluator_builders`.

## 2. Benchmarks — Standalone Evaluation

Full benchmark framework reusing the RL `Env` abstraction. Each benchmark creates `Env` instances; the runner handles concurrency, trajectory storage, and aggregation.

### Run benchmarks

```python
from tinker_cookbook.eval.benchmarks import run_benchmark, run_benchmarks

# Single benchmark
result = await run_benchmark("gsm8k", sampling_client, renderer)
print(f"GSM8K: {result.score:.1%}")  # GSM8K: 78.3%

# When save_dir is set, the runner automatically resumes from previously completed examples.

# Multiple benchmarks (parallel by default)
results = await run_benchmarks(
    ["gsm8k", "mmlu_pro", "ifeval"],
    sampling_client, renderer,
    BenchmarkConfig(save_dir="evals/step500", max_examples=200),
)
```

### Available benchmarks

**Stable benchmarks** — verified against published scores:

| Benchmark | Type | Grading | Prerequisites |
|-----------|------|---------|---------------|
| gsm8k | Single-turn | Programmatic (numeric) | — |
| math500 | Single-turn | Programmatic (numeric) | — |
| aime_2025 | Single-turn | Programmatic (numeric) | — |
| aime_2026 | Single-turn | Programmatic (numeric) | — |
| mmlu_pro | Single-turn | Programmatic (MCQA) | — |
| mmlu_redux | Single-turn | Programmatic (MCQA) | — |
| gpqa | Single-turn | Programmatic (MCQA) | HF auth (gated) |
| ifeval | Single-turn | Programmatic (IF constraints) | — |
| arena_hard | Single-turn | LLM-as-judge | `judge_sampling_client` in config |
| mbpp | Single-turn | Code execution | Modal |

**Experimental benchmarks** (``_``-prefixed modules) — functional but need further validation:

| Benchmark | Type | Grading | Status |
|-----------|------|---------|--------|
| longbench | Single-turn | Programmatic | Limited by 65K context window |
| livecodebench | Single-turn | Code execution (Modal) | Score gap vs public (23% vs 75%) |
| ifbench | Single-turn | IF constraints | Verifier doesn't cover IFBench instruction types |
| bfcl | Single-turn | Function call AST | Ground truth format mismatch |
| terminal_bench | Multi-turn | Sandbox + tests (Modal) | Multi-turn works, test grading needs work |
| swe_bench | Multi-turn | Sandbox + pytest (Modal) | Not yet tested end-to-end |
| tau2_bench | Multi-turn | Tool dispatch + user sim | Needs ``judge_sampling_client`` |

**Prerequisites:**

- **HF auth (gated)**: Set `HF_TOKEN` or run `huggingface-cli login`.
- **Modal**: Install with `pip install 'tinker-cookbook[modal]'` and authenticate with `modal token new`.
- **Modal**: Benchmarks that execute code in a sandbox require Modal. Install with `pip install 'tinker-cookbook[modal]'` and authenticate with `modal token new`.
- **`judge_sampling_client`**: Benchmarks using LLM-as-judge or user simulation require a separate Tinker sampling client for the judge model. Pass via `BenchmarkConfig(judge_sampling_client=..., judge_renderer=...)`.

### Browse results

```python
from tinker_cookbook.eval.benchmarks import load_result, load_trajectories, print_trajectory

# Load aggregated score
result = load_result("evals/step500", "gsm8k")
print(f"{result.name}: {result.score:.1%} ({result.num_correct}/{result.num_examples})")

# Browse incorrect examples
wrong = load_trajectories("evals/step500", "gsm8k", incorrect_only=True)
for t in wrong[:5]:
    print(f"Expected: {t.logs['expected']}, Got: {t.logs['extracted']}")
    print_trajectory(t)
```

### Pass@k evaluation

When `num_samples > 1`, the runner evaluates each example multiple times and computes unbiased pass@k estimates (per the Codex paper):

```python
config = BenchmarkConfig(num_samples=10, save_dir="evals/pass_at_k")
result = await run_benchmark("mbpp", sampling_client, renderer, config)
print(result.pass_at_k)  # {1: 0.45, 5: 0.72, 10: 0.85}
```

### Use benchmarks as inline training evaluators

`BenchmarkEvaluator` bridges any benchmark into the `SamplingClientEvaluator` interface:

```python
from tinker_cookbook.eval.benchmark_evaluator import BenchmarkEvaluator

evaluator_builders = [
    lambda: BenchmarkEvaluator("gsm8k", renderer, max_examples=100),
    lambda: BenchmarkEvaluator("ifeval", renderer, max_examples=50),
]
```

### Add a new benchmark

1. Create `tinker_cookbook/eval/benchmarks/my_benchmark.py`
2. Implement an `Env` subclass (same as RL envs):

```python
from tinker_cookbook.rl.types import Env, StepResult

class MyEnv(Env):
    async def initial_observation(self):
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        response = self.renderer.tokenizer.decode(action)
        correct = grade(response, self.expected)
        return StepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={"correct": float(correct)},
            logs={"example_id": self.example_id, "expected": self.expected},
        )
```

3. Implement a `BenchmarkBuilder`:

```python
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.eval.benchmarks import register

class MyBenchmarkBuilder(BenchmarkBuilder):
    name = "my_benchmark"

    def make_envs(self, renderer, config):
        ds = load_dataset("my/dataset", split="test")
        if config.max_examples is not None:
            ds = ds.select(range(min(config.max_examples, len(ds))))
        return [MyEnv(row, renderer) for row in ds]

register(MyBenchmarkBuilder())
```

Key points:
- `Env` objects are single-use (no reset)
- Set `multi_turn = True` on the builder for agent/sandbox benchmarks (uses lower concurrency)
- Include a stable `example_id` in `logs` for cross-run comparison (e.g., hash of the question)
- The runner handles concurrency, timeouts, resumability, and JSONL storage automatically
- For sandbox-based benchmarks, define an async `cleanup()` method on your Env. The runner calls it on timeout or error to prevent resource leaks.

## 3. EvalStore — Cross-Checkpoint Comparison

Persistent, file-based storage for tracking evaluation across checkpoints. Matches examples by `example_id` to identify regressions and improvements.

```python
from tinker_cookbook.eval.store import EvalStore
from tinker_cookbook.eval.benchmarks import run_benchmarks, BenchmarkConfig

store = EvalStore("~/experiments/evals")

# Run evals for a checkpoint
run_id = store.create_run(
    model_name="nvidia/...",
    checkpoint_name="sft_step500",
    benchmarks=["gsm8k", "ifeval"],
)
await run_benchmarks(
    ["gsm8k", "ifeval"], sampling_client, renderer,
    BenchmarkConfig(save_dir=store.run_dir(run_id)),
)
store.finalize_run(run_id)

# Compare two checkpoints
comp = store.compare_runs("sft_step500_20260327", "ifrl_step30_20260327", "gsm8k")
store.print_comparison(comp)
# === gsm8k: sft_step500 vs ifrl_step30 ===
#   Score: 0.743 -> 0.781 (delta=+0.038)
#   Regressions: 3 (correct in A, wrong in B)
#   Improvements: 18 (wrong in A, correct in B)

# Dashboard across all runs
store.print_dashboard()
```

### Storage layout

```
eval_store/
  runs.jsonl                          # Append-only index
  runs/
    sft_step500_20260327_143022/
      metadata.json                   # Model, checkpoint, config, scores
      gsm8k/
        result.json                   # Aggregated BenchmarkResult
        trajectories.jsonl            # Per-example StoredTrajectory
      ifeval/
        result.json
        trajectories.jsonl
```

## Configuration

`BenchmarkConfig` controls runtime behavior:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_examples` | `None` (all) | Limit number of examples |
| `concurrency` | `64` | Max concurrent rollouts (single-turn) |
| `agent_concurrency` | `8` | Max concurrent rollouts (multi-turn) |
| `timeout_seconds` | `300` | Per-example timeout |
| `max_tokens` | `32768` | Max generation tokens |
| `temperature` | `0.6` | Sampling temperature |
| `num_samples` | `1` | Number of samples per example for pass@k evaluation |
| `save_dir` | `None` | Directory for saving trajectories/results |
| `judge_sampling_client` | `None` | Sampling client for LLM-as-judge benchmarks |

## Verification

We verified the benchmark framework against published scores by running **Qwen3.5-35B-A3B** and comparing with the [official model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B):

| Benchmark | Our Score | Public Score | Settings | Match? |
|-----------|----------|-------------|----------|--------|
| AIME 2026 (pass@4) | **93.3%** | **93.33%** | 64K tokens, 4 samples, 1800s timeout | **Exact match** |
| AIME 2026 (pass@1) | 85.0% | — | 64K tokens, 1 sample | — |
| LongBench v2 | 63.6%* | 59.0% | *on examples fitting 65K context | Consistent** |
| GSM8K | 76.6% | — | 32K tokens, 600s timeout | Not reported |
| MATH-500 | 80.8% | — | 32K tokens, 600s timeout | Not reported |

\* LongBench: 78% of examples exceed our 65K context window. On the 110 examples that fit, accuracy is 63.6% — above the public 59.0% which includes harder long-context examples evaluated with 256K context.

\*\* The public LongBench score includes all examples with a 256K context window. Our higher accuracy on the subset that fits in 65K is expected since shorter examples tend to be easier.

**Key finding:** When evaluation settings match the reference (MathArena protocol: 64K tokens, 4 samples), our framework reproduces published scores exactly. Discrepancies in other benchmarks are explained by setup differences (context window, timeout, number of samples), not framework bugs.

## Testing

```bash
pytest tinker_cookbook/eval/benchmarks/benchmark_test.py
```
