"""Evaluation: evaluator interfaces, benchmark framework, and eval store.

Three layers:

1. **Evaluators** (:class:`SamplingClientEvaluator`, :class:`TrainingClientEvaluator`):
   Lightweight interfaces for inline evaluation during training. Called every N
   steps with a training or sampling client, return ``dict[str, float]`` metrics.

2. **Benchmarks** (:mod:`tinker_cookbook.eval.benchmarks`):
   Standalone benchmark framework using the RL Env abstraction. Each benchmark
   creates Env instances; the runner handles concurrency, trajectory storage,
   and aggregation. Use :class:`BenchmarkEvaluator` to bridge benchmarks into
   the training evaluator interface.

3. **EvalStore** (:class:`tinker_cookbook.eval.store.EvalStore`):
   Persistent storage for evaluation runs. Tracks metadata, scores, and
   trajectories across checkpoints. Supports cross-run comparison to identify
   regressions and improvements.

Example — run and store::

    from tinker_cookbook.eval.store import EvalStore
    from tinker_cookbook.eval.benchmarks import run_benchmarks, BenchmarkConfig

    store = EvalStore("~/experiments/evals")
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

Example — compare checkpoints::

    comp = store.compare_runs("sft_step500", "sft_step500_ifrl", "ifeval")
    store.print_comparison(comp)
    # Regressions: 3, Improvements: 18
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
    # Sub-modules:
    #   tinker_cookbook.eval.benchmarks — benchmark framework
    #   tinker_cookbook.eval.store — persistent eval storage
    #   tinker_cookbook.eval.benchmark_evaluator — bridge to training evaluators
]
