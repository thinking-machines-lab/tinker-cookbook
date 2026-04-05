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
| mbpp | Single-turn | Code execution | Modal |

**Experimental benchmarks** (``_``-prefixed modules) — functional but need further validation:

| Benchmark | Type | Grading | Status |
|-----------|------|---------|--------|
| arena_hard | Single-turn | LLM-as-judge | Works with self-judge, needs cross-model judge |
| longbench | Single-turn | Programmatic | Limited by 65K context window |
| livecodebench | Single-turn | Code execution (Modal) | 47.4% on Qwen3.5-35B-A3B (needs 1800s timeout) |
| ifbench | Single-turn | IF constraints | Verifier doesn't cover IFBench instruction types |
| bfcl | Single-turn | Function call AST | Ground truth format mismatch |
| terminal_bench | Multi-turn | Sandbox + tests (Modal) | 27.7% on Qwen3.5-35B-A3B (ctx overflow on 65K model) |
| swe_bench | Multi-turn | Sandbox + pytest (Modal) | 0% — 65K context too small for multi-turn repo exploration |
| tau2_bench | Multi-turn | Tool dispatch + user sim | 30% (needs separate user simulator model) |

**Prerequisites:**

- **HF auth (gated)**: Set `HF_TOKEN` or run `huggingface-cli login`.
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
2. Implement a `MessageEnv` (recommended) — the renderer handles thinking-token stripping and prompt building automatically:

```python
from tinker_cookbook.eval.benchmarks._common import build_messages, make_example_id
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.message_env import MessageEnv, MessageStepResult

class MyMessageEnv(MessageEnv):
    def __init__(self, question: str, expected: str, example_id: str = ""):
        self.question = question
        self.expected = expected
        self.example_id = example_id

    async def initial_observation(self) -> list[Message]:
        return build_messages(self.question)

    async def step(self, message: Message) -> MessageStepResult:
        response = get_text_content(message)  # thinking already stripped
        correct = self.expected in response
        return MessageStepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_messages=[],
            metrics={"correct": float(correct)},
            logs={"example_id": self.example_id, "expected": self.expected},
        )
```

3. Implement a `BenchmarkBuilder` that creates envs and wraps them with `EnvFromMessageEnv`:

```python
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.eval.benchmarks import register
from tinker_cookbook.rl.message_env import EnvFromMessageEnv

class MyBenchmarkBuilder(BenchmarkBuilder):
    name = "my_benchmark"

    def make_envs(self, renderer, config):
        ds = load_dataset("my/dataset", split="test")
        if config.max_examples is not None:
            ds = ds.select(range(min(config.max_examples, len(ds))))
        envs = []
        for row in ds:
            msg_env = MyMessageEnv(row["question"], row["answer"])
            envs.append(EnvFromMessageEnv(
                renderer=renderer,
                message_env=msg_env,
                failed_parse_reward=0.0,
                context_overflow_reward=0.0,
            ))
        return envs

register(MyBenchmarkBuilder())
```

Key points:
- Use `MessageEnv` + `EnvFromMessageEnv` for automatic thinking-token stripping and context overflow handling
- `Env` objects are single-use (no reset)
- Set `multi_turn = True` on the builder for agent/sandbox benchmarks (uses lower concurrency)
- Include a stable `example_id` in `logs` for cross-run comparison (e.g., hash of the question)
- The runner handles concurrency, timeouts, resumability, and JSONL storage automatically
- For sandbox-based benchmarks, use the `SandboxMixin` from `_common.py` for cleanup. The runner calls `cleanup()` on timeout or error to prevent resource leaks.

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

## Important: scores are setup-dependent

Benchmark scores are highly sensitive to evaluation settings. Small changes in `max_tokens`, `temperature`, `system_prompt`, or `timeout_seconds` can shift scores by 10–30%. Always document your exact configuration when reporting results.

Common pitfalls with thinking models:
- **`max_tokens` truncation**: Thinking models generate long reasoning chains that may fill `max_tokens` before producing an answer. For LiveCodeBench v6, 78/91 wrong answers were truncated at 32K tokens — increasing `max_tokens` to 64K would likely recover most of them.
- **Timeouts**: Thinking models need 1800s+ for code benchmarks. LiveCodeBench went from 20% (600s) to 47.4% (1800s) on Qwen3.5-35B-A3B.
- **Context overflow**: Multi-turn benchmarks (terminal_bench, swe_bench) can exceed the model's context window as conversations grow. The 65K context window of Qwen3.5-35B-A3B is insufficient for SWE-bench.
- **System prompt**: GSM8K improved from 84.7% to 95.6% by instructing the model to use `\boxed{}`.

Treat these scores as reference points for a specific configuration, not definitive model capabilities. The framework's primary value is **consistent, reproducible evaluation** — not producing leaderboard numbers.

## Verification

Reference scores on **[Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)** with `max_tokens=32768`, `temperature=0.6`.
Official scores from the model card (which may use different settings).

**Stable benchmarks:**

| Benchmark | Our Score | Official | Match? | Settings |
|-----------|-----------|----------|--------|----------|
| MMLU-Pro | 85.2%* | 85.3 | **Match** | 32K tokens |
| MMLU-Redux | **93.5%** | 93.3 | **Match** | 32K tokens |
| GPQA Diamond | 91.5%* | 84.2 | Above* | 32K tokens |
| IFEval | 93.6%* | 91.9 | **Match** | 32K tokens |
| GSM8K | 95.6%* | — | — | system_prompt=\boxed{}, 32K tokens |
| MATH-500 | 96.2%* | — | — | system_prompt=\boxed{}, 32K tokens |
| MBPP | 84.4%* | — | — | Modal sandbox, 32K tokens |
| AIME 2026 (pass@4) | 90.0% | 93.33 | Close | system_prompt=\boxed{}, 32K tokens |

\* Excluding context overflow — the thinking model's reasoning chain exceeds context on some examples. These are scored as failures (reward=0).

**Experimental benchmarks (Modal sandbox):**

| Benchmark | Our Score | Official | Notes |
|-----------|-----------|----------|-------|
| LiveCodeBench v6 | **47.4%** (175 ex) | 74.6 | 78/91 wrong due to 32K truncation; excl. truncated: 86.5% |
| Terminal Bench 2 | **27.7%** (112 ex) | 40.5 | 24 ctx overflow + 14 timeout on 65K model |
| SWE-bench Verified | 0% (500 ex) | 69.2 | 65K context too small — all ctx overflow |
| TAU2-Bench | 30.0% (50 ex) | 81.2 | Same-model user sim limits score; official uses GPT-4.1 |

## Testing

```bash
pytest tinker_cookbook/eval/benchmarks/benchmark_test.py
```
