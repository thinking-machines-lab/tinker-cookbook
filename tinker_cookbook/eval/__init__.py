"""Evaluation: evaluator interfaces, benchmark framework, and builders.

Two evaluation systems:

1. **Evaluators** (:class:`SamplingClientEvaluator`, :class:`TrainingClientEvaluator`):
   Lightweight interfaces for inline evaluation during training. Called every N
   steps with a training or sampling client, return ``dict[str, float]`` metrics.

2. **Benchmarks** (:mod:`tinker_cookbook.eval.benchmarks`):
   Standalone benchmark framework using the RL Env abstraction. Each benchmark
   creates Env instances; the runner handles concurrency, trajectory storage,
   and aggregation. Use :class:`BenchmarkEvaluator` to bridge benchmarks into
   the training evaluator interface.

Example — standalone benchmark::

    from tinker_cookbook.eval.benchmarks import run_benchmarks, BenchmarkConfig

    results = await run_benchmarks(
        ["gsm8k", "mmlu_pro", "ifeval"],
        sampling_client, renderer,
        BenchmarkConfig(save_dir="evals/checkpoint_001"),
    )

Example — inline training eval::

    from tinker_cookbook.eval.benchmark_evaluator import BenchmarkEvaluator

    config = Config(
        evaluator_builders=[
            lambda: BenchmarkEvaluator("gsm8k", renderer, max_examples=100),
        ],
        ...
    )
"""

from tinker_cookbook.eval.evaluators import (
    Evaluator,
    EvaluatorBuilder,
    SamplingClientEvaluator,
    SamplingClientEvaluatorBuilder,
    TrainingClientEvaluator,
)

__all__ = [
    # Evaluator interfaces (for training loops)
    "Evaluator",
    "EvaluatorBuilder",
    "SamplingClientEvaluator",
    "SamplingClientEvaluatorBuilder",
    "TrainingClientEvaluator",
    # Benchmark sub-package accessible via tinker_cookbook.eval.benchmarks
    # BenchmarkEvaluator accessible via tinker_cookbook.eval.benchmark_evaluator
]
