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

    # Run multiple benchmarks
    results = await run_benchmarks(
        ["gsm8k", "mmlu_pro", "ifeval"],
        sampling_client, renderer,
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
    run_benchmark,
    run_benchmarks,
)
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder

REGISTRY: dict[str, BenchmarkBuilder] = {}
"""Global registry of available benchmarks. Populated by benchmark modules."""


def register(builder: BenchmarkBuilder) -> BenchmarkBuilder:
    """Register a benchmark builder in the global registry."""
    REGISTRY[builder.name] = builder
    return builder


__all__ = [
    "BenchmarkBuilder",
    "BenchmarkConfig",
    "BenchmarkResult",
    "REGISTRY",
    "register",
    "run_benchmark",
    "run_benchmarks",
]
