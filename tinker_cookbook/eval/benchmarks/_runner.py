"""Benchmark runner — handles concurrency, storage, resumability.

Key design decisions:
- Uses ``do_single_rollout`` from RL infrastructure (same Env protocol)
- Trajectories stored as JSONL with decoded text for visualization
- Resumability via idx-based deduplication
- Thread-safe saving via asyncio.Lock
- Multi-turn benchmarks get lower concurrency (agent_concurrency)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import tinker

from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.eval.benchmarks._types import (
    BenchmarkBuilder,
    BenchmarkConfig,
    BenchmarkResult,
    StoredTrajectory,
    StoredTurn,
)
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.types import Trajectory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory conversion — decode tokens to text for storage
# ---------------------------------------------------------------------------


def _trajectory_to_stored(
    idx: int,
    trajectory: Trajectory,
    benchmark_name: str,
    tokenizer,
    time_seconds: float,
) -> StoredTrajectory:
    """Convert a Trajectory to a StoredTrajectory with decoded text."""
    turns: list[StoredTurn] = []
    total_reward = 0.0
    all_metrics: dict = {}
    all_logs: dict = {}

    for t in trajectory.transitions:
        total_reward += t.reward

        # Decode observation (the prompt/environment response for this turn)
        ob_text = tokenizer.decode(list(t.ob.to_ints())) if t.ob.length > 0 else ""
        if ob_text:
            # First turn's observation is the initial prompt (role=user)
            # Subsequent turns are environment responses
            role = "user" if len(turns) == 0 else "environment"
            turns.append(StoredTurn(
                role=role,
                content=ob_text,
                token_count=t.ob.length,
            ))

        # Decode action (the model's response)
        ac_text = tokenizer.decode(t.ac.tokens) if t.ac.tokens else ""
        turns.append(StoredTurn(
            role="assistant",
            content=ac_text,
            token_count=len(t.ac.tokens),
        ))

        all_metrics.update(t.metrics)
        all_logs.update(t.logs)

    return StoredTrajectory(
        idx=idx,
        benchmark=benchmark_name,
        turns=turns,
        reward=total_reward,
        metrics=all_metrics,
        logs=all_logs,
        time_seconds=time_seconds,
    )


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _load_completed(save_dir: str, benchmark_name: str) -> dict[int, float]:
    """Load completed trajectory indices and their rewards from disk."""
    path = Path(save_dir) / benchmark_name / "trajectories.jsonl"
    if not path.exists():
        return {}
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                results[d["idx"]] = d["reward"]
    return results


_save_lock = asyncio.Lock()


async def _save_trajectory(save_dir: str, benchmark_name: str, traj: StoredTrajectory) -> None:
    """Append one trajectory to the JSONL file (thread-safe)."""
    dir_path = Path(save_dir) / benchmark_name
    dir_path.mkdir(parents=True, exist_ok=True)
    async with _save_lock:
        with open(dir_path / "trajectories.jsonl", "a") as f:
            f.write(json.dumps(traj.to_dict()) + "\n")


def _save_result(save_dir: str, result: BenchmarkResult) -> None:
    """Save the aggregated result as JSON."""
    dir_path = Path(save_dir) / result.name
    dir_path.mkdir(parents=True, exist_ok=True)
    with open(dir_path / "result.json", "w") as f:
        json.dump(
            {
                "name": result.name,
                "score": result.score,
                "num_examples": result.num_examples,
                "num_correct": result.num_correct,
                "num_errors": result.num_errors,
                "metrics": result.metrics,
                "time_seconds": result.time_seconds,
            },
            f,
            indent=2,
        )


def _save_summary(save_dir: str, results: dict[str, BenchmarkResult]) -> None:
    """Save a combined summary across all benchmarks."""
    summary = {}
    for name, r in results.items():
        summary[name] = {"score": r.score, "num_examples": r.num_examples, "num_correct": r.num_correct}
    with open(Path(save_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_benchmark(
    benchmark: BenchmarkBuilder | str,
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    config: BenchmarkConfig = BenchmarkConfig(),
) -> BenchmarkResult:
    """Run a single benchmark end-to-end.

    Handles:
    - Concurrent rollouts with semaphore (adjusts for multi-turn)
    - Trajectory storage with decoded text (JSONL)
    - Resumability (skips already-completed examples by idx)
    - Error isolation (failed examples logged, not fatal)
    - Progress logging

    Args:
        benchmark: A BenchmarkBuilder instance or a name from REGISTRY.
        sampling_client: Tinker sampling client for model generation.
        renderer: Renderer for tokenization and prompt building.
        config: Runtime configuration.

    Returns:
        Aggregated BenchmarkResult.
    """
    from tinker_cookbook.eval.benchmarks import REGISTRY

    if isinstance(benchmark, str):
        benchmark = REGISTRY[benchmark]

    t0 = time.monotonic()
    logger.info(f"Running benchmark: {benchmark.name}")

    # Create envs
    envs = list(benchmark.make_envs(renderer, config))
    logger.info(f"  {len(envs)} examples loaded")

    # Resume: load completed examples from disk
    completed_rewards: dict[int, float] = {}
    if config.save_dir:
        completed_rewards = _load_completed(config.save_dir, benchmark.name)
        if completed_rewards:
            logger.info(
                f"  Resuming: {len(completed_rewards)} already completed, "
                f"{len(envs) - len(completed_rewards)} remaining"
            )

    # Build policy
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        context_window=config.context_window,
    )

    # Choose concurrency based on benchmark type
    max_concurrent = config.agent_concurrency if benchmark.multi_turn else config.concurrency
    sem = asyncio.Semaphore(max_concurrent)

    # Per-example results (None = not yet run or errored)
    rewards: list[float | None] = [None] * len(envs)
    metrics_list: list[dict] = [{}] * len(envs)
    num_errors = 0
    num_completed = 0
    total_to_run = len(envs) - len(completed_rewards)

    # Pre-fill from resumed results
    for idx, reward in completed_rewards.items():
        if idx < len(rewards):
            rewards[idx] = reward

    tokenizer = renderer.tokenizer

    async def run_one(idx: int, env) -> None:
        nonlocal num_completed, num_errors

        # Skip already-completed
        if idx in completed_rewards:
            return

        async with sem:
            t_start = time.monotonic()
            try:
                trajectory = await do_single_rollout(policy, env)
                elapsed = time.monotonic() - t_start

                total_reward = sum(t.reward for t in trajectory.transitions)
                rewards[idx] = total_reward

                # Collect metrics from all transitions
                step_metrics: dict = {}
                for t in trajectory.transitions:
                    step_metrics.update(t.metrics)
                metrics_list[idx] = step_metrics

                # Save trajectory with decoded text
                if config.save_dir:
                    stored = _trajectory_to_stored(
                        idx, trajectory, benchmark.name, tokenizer, elapsed
                    )
                    await _save_trajectory(config.save_dir, benchmark.name, stored)

            except Exception as e:
                elapsed = time.monotonic() - t_start
                logger.warning(f"  {benchmark.name}[{idx}] failed ({elapsed:.1f}s): {e}")
                num_errors += 1

                # Save error trajectory
                if config.save_dir:
                    error_traj = StoredTrajectory(
                        idx=idx,
                        benchmark=benchmark.name,
                        turns=[],
                        reward=0.0,
                        error=str(e),
                        time_seconds=elapsed,
                    )
                    await _save_trajectory(config.save_dir, benchmark.name, error_traj)

            num_completed += 1
            if num_completed % max(1, total_to_run // 10) == 0 or num_completed == total_to_run:
                total_elapsed = time.monotonic() - t0
                logger.info(
                    f"  {benchmark.name}: {num_completed}/{total_to_run} done "
                    f"({total_elapsed:.0f}s elapsed, {num_errors} errors)"
                )

    await asyncio.gather(*[run_one(i, env) for i, env in enumerate(envs)])

    # Aggregate
    valid_rewards = [r for r in rewards if r is not None]
    valid_metrics = [m for r, m in zip(rewards, metrics_list) if r is not None]

    result = benchmark.aggregate(valid_rewards, valid_metrics)
    result.num_errors = num_errors
    result.time_seconds = time.monotonic() - t0

    logger.info(
        f"  {benchmark.name}: score={result.score:.3f} "
        f"({result.num_correct}/{result.num_examples}, {num_errors} errors) "
        f"in {result.time_seconds:.0f}s"
    )

    if config.save_dir:
        _save_result(config.save_dir, result)

    return result


async def run_benchmarks(
    names: list[str],
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    config: BenchmarkConfig = BenchmarkConfig(),
) -> dict[str, BenchmarkResult]:
    """Run multiple benchmarks sequentially, saving a combined summary.

    Args:
        names: List of benchmark names from REGISTRY.
        sampling_client: Tinker sampling client.
        renderer: Renderer for tokenization.
        config: Shared runtime configuration.

    Returns:
        Dict mapping benchmark name to BenchmarkResult.
    """
    results = {}
    for name in names:
        results[name] = await run_benchmark(name, sampling_client, renderer, config)

    # Save combined summary
    if config.save_dir:
        _save_summary(config.save_dir, results)

    # Log summary table
    logger.info("\n=== Benchmark Summary ===")
    for name, r in results.items():
        logger.info(f"  {name:20s} {r.score:.3f} ({r.num_correct}/{r.num_examples})")

    return results
