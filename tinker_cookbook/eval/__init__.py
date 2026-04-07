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

3. **EvalStore** (:class:`tinker_cookbook.stores.eval_store.EvalStore`):
   Persistent storage for evaluation runs. Tracks metadata, scores, and
   trajectories across checkpoints. Supports cross-run comparison to identify
   regressions and improvements.

Example — quick eval (no persistence)::

    result = await run_benchmark("gsm8k", sampling_client, renderer)
    print(f"GSM8K: {result.score:.1%}")

Example — persistent eval with EvalStore::

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

    # Query results
    result = store.read_result(run_id, "gsm8k")
    wrong = store.read_trajectories(run_id, "gsm8k", incorrect_only=True)

"""

from tinker_cookbook.eval.benchmark_evaluator import BenchmarkEvaluator
from tinker_cookbook.eval.benchmarks import (
    REGISTRY,
    BenchmarkBuilder,
    BenchmarkConfig,
    BenchmarkResult,
    StoredTrajectory,
    print_trajectory,
    register,
    run_benchmark,
    run_benchmarks,
)
from tinker_cookbook.eval.evaluators import (
    Evaluator,
    EvaluatorBuilder,
    SamplingClientEvaluator,
    SamplingClientEvaluatorBuilder,
    TrainingClientEvaluator,
)
from tinker_cookbook.eval.rl_benchmark import (
    RLTestSetBenchmarkBuilder,
    RLTestSetBenchmarkEvaluator,
)
from tinker_cookbook.stores.eval_store import EvalStore, RunMetadata

__all__ = [
    # Evaluator interfaces (for training loops)
    "Evaluator",
    "EvaluatorBuilder",
    "SamplingClientEvaluator",
    "SamplingClientEvaluatorBuilder",
    "TrainingClientEvaluator",
    # Benchmark framework — running
    "BenchmarkBuilder",
    "BenchmarkConfig",
    "BenchmarkResult",
    "StoredTrajectory",
    "REGISTRY",
    "register",
    "run_benchmark",
    "run_benchmarks",
    "print_trajectory",
    # Benchmark-to-evaluator bridge
    "BenchmarkEvaluator",
    # RL test set → benchmark adapter
    "RLTestSetBenchmarkBuilder",
    "RLTestSetBenchmarkEvaluator",
    # Eval store — persistent storage and querying
    "EvalStore",
    "RunMetadata",
]
