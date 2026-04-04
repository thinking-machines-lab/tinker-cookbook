"""
Benchmark evaluation framework for Tinker.

Provides a unified way to run evaluation benchmarks using the same Env
abstraction as RL training. Each benchmark is a BenchmarkBuilder that creates
Env instances — one per eval example. The runner handles concurrency, trajectory
storage, and result aggregation.

Usage::

    from tinker_cookbook.eval.benchmarks import run_benchmark, run_benchmarks, REGISTRY

    # Run a single benchmark
    result = await run_benchmark("gsm8k", sampling_client, renderer)
    print(f"GSM8K: {result.score:.1%}")

    # Run multiple benchmarks (parallel by default)
    results = await run_benchmarks(
        ["gsm8k", "mmlu_pro", "ifeval"],
        sampling_client, renderer,
    )

    # Compare across checkpoints (each gets its own save_dir)
    for name, path in checkpoints.items():
        client = sc.create_sampling_client(model_path=path)
        results[name] = await run_benchmarks(
            ["gsm8k", "ifeval"], client, renderer,
            BenchmarkConfig(save_dir=f"evals/{name}"),
        )

Design:
    - Benchmarks reuse the RL ``Env`` protocol — same step/reward interface
    - Single-turn evals: Env.step() grades the response and returns episode_done=True
    - Multi-turn evals: Env runs multiple turns (sandbox, tool use) then grades
    - All sampling requests run concurrently with a semaphore
    - Trajectories saved to disk as JSONL for visualization and resumability
"""

from tinker_cookbook.eval.benchmarks._runner import (
    BenchmarkConfig,
    BenchmarkResult,
    load_result,
    load_summary,
    load_trajectories,
    print_trajectory,
    regrade_trajectories,
    run_benchmark,
    run_benchmarks,
)
from tinker_cookbook.eval.benchmarks._types import (
    BenchmarkBuilder,
    BenchmarkResultDict,
    Logs,
    Metrics,
    StoredTrajectory,
    StoredTrajectoryDict,
    StoredTurnDict,
)

REGISTRY: dict[str, BenchmarkBuilder] = {}
"""Global registry of available benchmarks. Populated by benchmark modules."""


def register(builder: BenchmarkBuilder) -> BenchmarkBuilder:
    """Register a benchmark builder in the global registry."""
    REGISTRY[builder.name] = builder
    return builder


__all__ = [
    # Types
    "BenchmarkBuilder",
    "BenchmarkConfig",
    "BenchmarkResult",
    "StoredTrajectory",
    # Type aliases
    "Logs",
    "Metrics",
    # Serialization TypedDicts
    "BenchmarkResultDict",
    "StoredTrajectoryDict",
    "StoredTurnDict",
    # Registry
    "REGISTRY",
    "register",
    # Running
    "run_benchmark",
    "run_benchmarks",
    # Loading / viewing / regrading results
    "load_result",
    "load_trajectories",
    "load_summary",
    "print_trajectory",
    "regrade_trajectories",
]
