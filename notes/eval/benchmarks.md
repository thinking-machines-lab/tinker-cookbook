# Nemotron-Cascade-2 Benchmark Evaluation Suite

All benchmarks live in `tinker_cookbook/recipes/nemotron_cascade/run_evals.py`.

## Available Benchmarks

| Benchmark | Key | Type | Dataset | Metric |
|-----------|-----|------|---------|--------|
| GSM8K | `gsm8k` | Math | `openai/gsm8k` | Exact numeric match |
| IFEval | `ifeval` | Instruction following | Local JSONL | Loose + strict accuracy |
| MMLU-Pro | `mmlu` | Knowledge | `TIGER-Lab/MMLU-Pro` | Letter match |
| MATH-500 | `math500` | Math | `HuggingFaceH4/MATH-500` | `grade_answer` from math_grading |
| GPQA-Diamond | `gpqa` | Science QA | `Idavidrein/gpqa` (gpqa_diamond) | Letter match (A/B/C/D) |
| AIME 2025 | `aime2025` | Math competition | HF (multiple sources) | Integer match (0-999) |
| MBPP | `mbpp` | Code generation | `google-research-datasets/mbpp` | Execution pass (assertions) |
| LongBench v2 | `longbench` | Long-context | `THUDM/LongBench-v2` | MC letter match or substring |

## Usage

```bash
# Single benchmark
python -m tinker_cookbook.recipes.nemotron_cascade.run_evals \
    --benchmarks gpqa --limit 50

# Multiple benchmarks
python -m tinker_cookbook.recipes.nemotron_cascade.run_evals \
    --benchmarks gsm8k,gpqa,aime2025,mbpp,longbench --limit 100

# All benchmarks with checkpoint comparison
python -m tinker_cookbook.recipes.nemotron_cascade.run_evals \
    --compare \
    --sft-checkpoint /path/to/sft \
    --ifrl-checkpoint /path/to/ifrl \
    --benchmarks gsm8k,ifeval,mmlu,math500,gpqa,aime2025,mbpp,longbench
```

## Benchmark Details

### GPQA-Diamond
- 198 hard graduate-level science questions (physics, chemistry, biology)
- Multiple choice (A/B/C/D) with chain-of-thought prompting
- Dataset: `Idavidrein/gpqa` config `gpqa_diamond`

### AIME 2025
- ~30 problems from the 2025 American Invitational Mathematics Examination
- Integer answers in range 0-999
- Tries multiple HuggingFace dataset sources (auto-fallback)
- Uses `\boxed{}` extraction with fallback to last-number heuristic

### MBPP
- 257 (sanitized) basic Python programming tasks
- Execution-based grading: generates a function, runs assertion tests in subprocess
- No Modal dependency (uses local subprocess with 15s timeout)
- Reuses the same dataset as the Code RL training environment

### LongBench v2
- Long-context comprehension across multiple domains
- Prefers `THUDM/LongBench-v2` (multiple choice), falls back to v1 (open-ended)
- Reports per-subtask accuracy breakdowns in addition to overall accuracy
- Context lengths can be very long; uses the completer's default max_tokens

## Implementation Pattern

Every benchmark follows the same async signature:

```python
async def eval_BENCHMARK(
    completer: TinkerMessageCompleter,
    limit: int | None = None,
    concurrency: int = 128,
) -> dict[str, float]:
```

Concurrency is controlled via `asyncio.Semaphore(concurrency)`. All benchmarks
are registered in the `BENCHMARKS` dict and work with `--benchmarks` flag.
