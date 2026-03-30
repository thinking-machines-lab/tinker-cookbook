"""Benchmark runner — handles concurrency, storage, resumability."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import tinker

from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig, BenchmarkResult
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.types import Trajectory

logger = logging.getLogger(__name__)


def _trajectory_to_dict(idx: int, trajectory: Trajectory, benchmark_name: str) -> dict:
    """Convert a Trajectory to a JSON-serializable dict for storage."""
    total_reward = sum(t.reward for t in trajectory.transitions)
    turns = []
    for t in trajectory.transitions:
        turns.append({
            "ob_len": t.ob.length,
            "ac_len": len(t.ac.tokens),
            "reward": t.reward,
            "episode_done": t.episode_done,
            "metrics": t.metrics,
            "logs": t.logs,
        })
    return {
        "idx": idx,
        "benchmark": benchmark_name,
        "total_reward": total_reward,
        "num_turns": len(turns),
        "turns": turns,
        "timestamp": time.time(),
    }


def _load_completed_ids(save_dir: str, benchmark_name: str) -> set[int]:
    """Load indices of already-completed trajectories from disk."""
    path = Path(save_dir) / benchmark_name / "trajectories.jsonl"
    if not path.exists():
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["idx"])
    return ids


def _append_trajectory(save_dir: str, benchmark_name: str, traj_dict: dict) -> None:
    """Append one trajectory to the JSONL file."""
    dir_path = Path(save_dir) / benchmark_name
    dir_path.mkdir(parents=True, exist_ok=True)
    with open(dir_path / "trajectories.jsonl", "a") as f:
        f.write(json.dumps(traj_dict) + "\n")


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
                "metrics": result.metrics,
            },
            f,
            indent=2,
        )


async def run_benchmark(
    benchmark: BenchmarkBuilder | str,
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    config: BenchmarkConfig = BenchmarkConfig(),
) -> BenchmarkResult:
    """Run a single benchmark end-to-end.

    Handles concurrency (semaphore-bounded), trajectory storage (JSONL),
    and resumability (skips already-completed examples).

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

    logger.info(f"Running benchmark: {benchmark.name}")

    # Create envs
    envs = list(benchmark.make_envs(renderer, config))
    logger.info(f"  {len(envs)} examples loaded")

    # Resume: skip already-completed
    if config.save_dir:
        done_ids = _load_completed_ids(config.save_dir, benchmark.name)
        if done_ids:
            logger.info(f"  Resuming: {len(done_ids)} already completed, {len(envs) - len(done_ids)} remaining")
    else:
        done_ids = set()

    # Build policy (no logprobs needed for eval)
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    # Run rollouts concurrently
    sem = asyncio.Semaphore(config.concurrency)
    completed = 0
    total = len(envs) - len(done_ids)
    t0 = time.monotonic()

    rewards: list[float | None] = [None] * len(envs)
    metrics_list: list[dict] = [{}] * len(envs)
    batch_buffer: list[dict] = []

    async def run_one(idx: int, env) -> None:
        nonlocal completed
        if idx in done_ids:
            return

        async with sem:
            try:
                trajectory = await do_single_rollout(policy, env)
                total_reward = sum(t.reward for t in trajectory.transitions)
                rewards[idx] = total_reward

                # Collect metrics from all transitions
                step_metrics: dict = {}
                for t in trajectory.transitions:
                    step_metrics.update(t.metrics)
                metrics_list[idx] = step_metrics

                # Buffer for batch saving
                if config.save_dir:
                    traj_dict = _trajectory_to_dict(idx, trajectory, benchmark.name)
                    batch_buffer.append(traj_dict)
                    if len(batch_buffer) >= config.save_every:
                        for td in batch_buffer:
                            _append_trajectory(config.save_dir, benchmark.name, td)
                        batch_buffer.clear()

            except Exception as e:
                logger.warning(f"  {benchmark.name}[{idx}] failed: {e}")
                rewards[idx] = None

            completed += 1
            if completed % max(1, total // 10) == 0 or completed == total:
                elapsed = time.monotonic() - t0
                logger.info(
                    f"  {benchmark.name}: {completed}/{total} done "
                    f"({elapsed:.0f}s elapsed, {elapsed / completed:.1f}s/example)"
                )

    await asyncio.gather(*[run_one(i, env) for i, env in enumerate(envs)])

    # Flush remaining buffer
    if config.save_dir and batch_buffer:
        for td in batch_buffer:
            _append_trajectory(config.save_dir, benchmark.name, td)
        batch_buffer.clear()

    # Load rewards from disk for resumed examples
    if done_ids and config.save_dir:
        path = Path(config.save_dir) / benchmark.name / "trajectories.jsonl"
        with open(path) as f:
            for line in f:
                d = json.loads(line.strip())
                idx = d["idx"]
                if idx < len(rewards) and rewards[idx] is None:
                    rewards[idx] = d["total_reward"]

    # Filter out failures
    valid_rewards = [r for r in rewards if r is not None]
    valid_metrics = [m for r, m in zip(rewards, metrics_list) if r is not None]

    result = benchmark.aggregate(valid_rewards, valid_metrics)

    elapsed = time.monotonic() - t0
    logger.info(
        f"  {benchmark.name}: score={result.score:.3f} "
        f"({result.num_correct}/{result.num_examples}) in {elapsed:.0f}s"
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
    """Run multiple benchmarks sequentially.

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
    return results
