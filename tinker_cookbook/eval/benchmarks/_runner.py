"""Benchmark runner — handles concurrency, storage, resumability.

Key design decisions:
- Uses ``do_single_rollout`` from RL infrastructure (same Env protocol)
- Trajectories stored as JSONL with decoded text for visualization
- Resumability via idx-based deduplication
- Coroutine-safe saving via asyncio.Lock
- Multi-turn benchmarks get lower concurrency (agent_concurrency)

**Future plan — Storage migration:**

File I/O currently uses ``Path``/``open()`` directly. The write paths below
should be migrated to the ``Storage`` protocol (``tinker_cookbook.stores``)
to enable cloud-backed eval persistence (S3/GCS/Azure):

1. ``_save_trajectory`` → ``EvalStore.write_trajectory``
2. ``_save_result`` → ``EvalStore.write_result``
3. ``_save_summary`` → ``EvalStore.write_summary``
4. ``_load_completed`` → ``EvalStore.read_trajectories`` (for resumability)

``EvalStore`` already provides these methods with Storage-backed I/O.
The main challenge is threading the ``EvalStore`` instance through the
runner's async concurrency model (``asyncio.Lock``, ``asyncio.Semaphore``).
See ``stores/storage.py`` for the protocol contract.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path

import tinker

from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.eval.benchmarks._types import (
    METRIC_CONTEXT_OVERFLOW,
    METRIC_MAX_TOKENS_REACHED,
    BenchmarkBuilder,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkResultDict,
    Logs,
    Metrics,
    PassAtKScores,
    StoredTrajectory,
    StoredTurn,
    TurnRole,
)
from tinker_cookbook.exceptions import BenchmarkNotFoundError, EvalTimeoutError
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.types import Env, Trajectory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory conversion — decode tokens to text for storage
# ---------------------------------------------------------------------------


_warned_no_example_id: set[str] = set()


def _get_example_id(env: Env, idx: int, benchmark_name: str = "") -> str:
    """Get the stable ``example_id`` from an env, falling back to ``str(idx)``.

    ``example_id`` is set on ``EnvFromMessageEnv`` (forwarded from the inner
    ``MessageEnv``). For raw ``Env`` implementations that don't set it,
    falls back to the positional index and logs a one-time warning.
    """
    eid = getattr(env, "example_id", None)
    if eid:
        return eid
    if benchmark_name and benchmark_name not in _warned_no_example_id:
        _warned_no_example_id.add(benchmark_name)
        logger.warning(
            f"  {benchmark_name} envs have no example_id. Using positional index "
            f"which is unstable across shuffles and pass@k rounds. Set example_id "
            f"on your MessageEnv (use make_example_id from _common.py)."
        )
    return str(idx)


def _trajectory_to_stored(
    idx: int,
    trajectory: Trajectory,
    benchmark_name: str,
    tokenizer,
    time_seconds: float,
    example_id: str,
) -> StoredTrajectory:
    """Convert a Trajectory to a StoredTrajectory with decoded text."""
    turns: list[StoredTurn] = []
    total_reward = 0.0
    all_metrics: Metrics = {}
    all_logs: Logs = {}

    for t in trajectory.transitions:
        total_reward += t.reward

        # Decode observation (the prompt/environment response for this turn)
        ob_text = tokenizer.decode(list(t.ob.to_ints())) if t.ob.length > 0 else ""
        if ob_text:
            # First turn's observation is the initial prompt (role=user)
            # Subsequent turns are environment responses
            role: TurnRole = "user" if len(turns) == 0 else "environment"
            turns.append(
                StoredTurn(
                    role=role,
                    content=ob_text,
                    token_count=t.ob.length,
                )
            )

        # Decode action (the model's response)
        ac_text = tokenizer.decode(t.ac.tokens) if t.ac.tokens else ""
        turns.append(
            StoredTurn(
                role="assistant",
                content=ac_text,
                token_count=len(t.ac.tokens),
            )
        )

        all_metrics.update(t.metrics)
        all_logs.update(t.logs)

    # Remove example_id from logs (already stored as a top-level field)
    all_logs.pop("example_id", None)

    return StoredTrajectory(
        idx=idx,
        benchmark=benchmark_name,
        example_id=example_id,
        turns=turns,
        reward=total_reward,
        metrics=all_metrics,
        logs=all_logs,
        time_seconds=time_seconds,
    )


def _last_assistant_content(turns: list[StoredTurn]) -> str:
    """Return the content of the last assistant turn, or empty string."""
    for turn in reversed(turns):
        if turn.role == "assistant":
            return turn.content
    return ""


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


class _ResumedExample:
    """Data restored from a previously completed trajectory for resumability."""

    __slots__ = ("is_error", "metrics", "reward")

    def __init__(self, reward: float, metrics: Metrics, is_error: bool):
        self.reward = reward
        self.metrics = metrics
        self.is_error = is_error


def _load_completed(save_dir: str, benchmark_name: str) -> dict[str, _ResumedExample]:
    """Load completed trajectories keyed by ``example_id``.

    Uses ``example_id`` (content-hash) instead of positional ``idx``,
    making resumability robust to dataset shuffling and ``max_examples``
    changes between runs.
    """
    path = Path(save_dir) / benchmark_name / "trajectories.jsonl"
    if not path.exists():
        return {}
    results: dict[str, _ResumedExample] = {}
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                eid = d.get("example_id") or str(d["idx"])
                results[eid] = _ResumedExample(
                    reward=d.get("reward", 0.0),
                    metrics=d.get("metrics", {}),
                    is_error=d.get("error") is not None,
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    f"Skipping malformed trajectory line {line_num} in "
                    f"{benchmark_name}/trajectories.jsonl: {e}"
                )
    return results


_save_locks: dict[str, asyncio.Lock] = {}


def _get_save_lock(key: str) -> asyncio.Lock:
    """Get or create a per-file lock. Supports parallel benchmarks and checkpoints."""
    return _save_locks.setdefault(key, asyncio.Lock())


_created_dirs: set[str] = set()


async def _save_trajectory(save_dir: str, benchmark_name: str, traj: StoredTrajectory) -> None:
    """Append one trajectory to the JSONL file (per-file lock for parallel safety)."""
    dir_path = Path(save_dir) / benchmark_name
    dir_key = str(dir_path)
    if dir_key not in _created_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        _created_dirs.add(dir_key)
    filepath = str(dir_path / "trajectories.jsonl")
    async with _get_save_lock(filepath):
        with open(filepath, "a") as f:
            f.write(json.dumps(traj.to_dict()) + "\n")


def _save_result(save_dir: str, result: BenchmarkResult) -> None:
    """Save the aggregated result as JSON."""
    dir_path = Path(save_dir) / result.name
    dir_path.mkdir(parents=True, exist_ok=True)
    with open(dir_path / "result.json", "w") as f:
        d = BenchmarkResultDict(
            name=result.name,
            score=result.score,
            num_examples=result.num_examples,
            num_correct=result.num_correct,
            num_errors=result.num_errors,
            num_truncated=result.num_truncated,
            metrics=result.metrics,
            time_seconds=result.time_seconds,
            # JSON keys must be strings; int keys converted back in load_result
            pass_at_k={str(k): v for k, v in result.pass_at_k.items()},
        )
        json.dump(d, f, indent=2)


def _save_summary(save_dir: str, results: dict[str, BenchmarkResult]) -> None:
    """Save a combined summary across all benchmarks."""
    summary = {}
    for name, r in results.items():
        entry: dict = {
            "score": r.score,
            "num_examples": r.num_examples,
            "num_correct": r.num_correct,
        }
        if r.pass_at_k:
            entry["pass_at_k"] = {str(k): v for k, v in r.pass_at_k.items()}
        summary[name] = entry
    with open(Path(save_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Pass@k computation
# ---------------------------------------------------------------------------


def _pass_at_k_single(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k for a single example (Codex paper).

    Args:
        n: Total number of samples.
        c: Number of correct samples.
        k: k value for pass@k.

    Returns:
        Probability that at least one of k random samples is correct.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _compute_pass_at_k(
    per_example_results: dict[str, list[float]],
    k_values: list[int],
) -> PassAtKScores:
    """Compute pass@k across all examples for given k values.

    Args:
        per_example_results: Maps example_id to list of rewards from each sample.
        k_values: List of k values to compute (e.g. [1, 5, 10]).

    Returns:
        :data:`PassAtKScores` mapping k to mean pass@k across all examples.
    """
    if not per_example_results:
        return {}

    result: PassAtKScores = {}
    for k in k_values:
        scores: list[float] = []
        for rewards in per_example_results.values():
            n = len(rewards)
            if k > n:
                continue  # Can't compute pass@k when k > n
            c = sum(1 for r in rewards if r > 0)
            scores.append(_pass_at_k_single(n, c, k))
        if scores:
            result[k] = sum(scores) / len(scores)
    return result


def _choose_k_values(num_samples: int) -> list[int]:
    """Choose which k values to report for a given num_samples.

    Always includes 1 and num_samples. Also includes standard intermediate
    values (5, 10, 25, 50, 100) if they fit.
    """
    candidates = [1, 5, 10, 25, 50, 100]
    k_values = [k for k in candidates if k <= num_samples]
    if num_samples not in k_values:
        k_values.append(num_samples)
    return sorted(k_values)


# ---------------------------------------------------------------------------
# Requirement validation
# ---------------------------------------------------------------------------


def _validate_requirements(benchmark: BenchmarkBuilder, config: BenchmarkConfig) -> None:
    """Check that config satisfies the benchmark's declared requirements.

    Called before make_envs() so misconfigurations fail fast instead of
    mid-evaluation.
    """
    if benchmark.requires_judge and config.judge_sampling_client is None:
        raise ValueError(
            f"Benchmark '{benchmark.name}' requires an LLM judge. "
            f"Set config.judge_sampling_client to a Tinker SamplingClient for the judge model."
        )
    if benchmark.requires_sandbox and config.sandbox_factory is None:
        try:
            from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox  # noqa: F401
        except ImportError:
            raise ValueError(
                f"Benchmark '{benchmark.name}' requires a sandbox for code execution. "
                f"Either install Modal (`pip install 'tinker-cookbook[modal]'`) "
                f"or provide a custom sandbox_factory in BenchmarkConfig."
            ) from None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _ensure_registered(name: str) -> None:
    """Auto-import a benchmark module to populate REGISTRY if needed."""
    from tinker_cookbook.eval.benchmarks import REGISTRY

    if name in REGISTRY:
        return
    # Try importing the benchmark module by name.
    # Also try the base name (e.g., "aime" for "aime_2026") since a single
    # module may register multiple benchmark variants.
    import importlib

    candidates = [name]
    # Strip trailing _YYYY suffix to find parent module (e.g., "aime_2026" -> "aime")
    if len(name) > 5 and name[-4:].isdigit() and name[-5] == "_":
        candidates.append(name[:-5])
    # Also try first segment as module name (e.g., "hmmt_feb_2025" -> "hmmt")
    first_segment = name.split("_")[0]
    if first_segment != name and first_segment not in candidates:
        candidates.append(first_segment)

    for module_name in candidates:
        # Try public module, then _-prefixed (experimental) module
        for prefix in ("", "_"):
            with contextlib.suppress(ImportError):
                importlib.import_module(f"tinker_cookbook.eval.benchmarks.{prefix}{module_name}")
            if name in REGISTRY:
                return


async def run_benchmark(
    benchmark: BenchmarkBuilder | str,
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    config: BenchmarkConfig = BenchmarkConfig(),
) -> BenchmarkResult:
    """Run a single benchmark end-to-end.

    Creates one :class:`Env` per dataset example, runs them concurrently with
    the model, grades responses, and returns aggregated results. Trajectories
    are saved to disk as JSONL for later inspection via :func:`load_trajectories`.

    If ``config.save_dir`` is set and a previous run's trajectories exist,
    completed examples are skipped automatically (resumability).

    Args:
        benchmark: A :class:`BenchmarkBuilder` instance, or a benchmark name
            (e.g. ``"gsm8k"``, ``"mmlu_pro"``) that will be looked up in the
            global :data:`REGISTRY`.
        sampling_client: Tinker sampling client for model generation.
        renderer: Renderer for tokenization and prompt building. Must match
            the model family (e.g. ``"qwen3_5"`` for Qwen3.5 models).
        config: Runtime configuration controlling concurrency, timeouts,
            generation parameters, and optional customization hooks
            (``system_prompt``, ``grade_fn``). See :class:`BenchmarkConfig`.

    Returns:
        A :class:`BenchmarkResult` with the aggregated score, per-example
        metrics, and optional pass@k estimates (when ``config.num_samples > 1``).

    Raises:
        BenchmarkNotFoundError: If ``benchmark`` is a string not in REGISTRY.

    Example::

        result = await run_benchmark("gsm8k", sampling_client, renderer)
        print(f"GSM8K: {result.score:.1%}")  # e.g. "GSM8K: 84.7%"

        # With custom grading and trajectory storage:
        config = BenchmarkConfig(
            save_dir="evals/step500",
            timeout_seconds=1800,
            grade_fn=my_custom_grader,
        )
        result = await run_benchmark("gsm8k", sampling_client, renderer, config)
    """
    from tinker_cookbook.eval.benchmarks import REGISTRY

    if isinstance(benchmark, str):
        _ensure_registered(benchmark)
        if benchmark not in REGISTRY:
            raise BenchmarkNotFoundError(
                f"Unknown benchmark '{benchmark}'. Available: {sorted(REGISTRY.keys())}. "
                f"Make sure the benchmark module exists at tinker_cookbook.eval.benchmarks.{benchmark}"
            )
        benchmark = REGISTRY[benchmark]

    # Validate requirements before doing any work
    _validate_requirements(benchmark, config)

    if benchmark.experimental:
        logger.warning(
            f"Benchmark '{benchmark.name}' is experimental and may not match "
            f"published scores. See the benchmark module docstring for known "
            f"limitations and status."
        )

    # Apply benchmark's recommended system_prompt if user didn't set one
    if config.system_prompt is None and benchmark.recommended_system_prompt is not None:
        config = dataclasses.replace(config, system_prompt=benchmark.recommended_system_prompt)
        logger.info("  Using benchmark's recommended system_prompt")

    t0 = time.monotonic()
    num_samples = config.num_samples

    if num_samples > 1:
        logger.info(f"Running benchmark: {benchmark.name} (pass@k mode, {num_samples} samples)")
    else:
        logger.info(f"Running benchmark: {benchmark.name}")

    # Create envs (first batch — used directly for single-sample, or as the
    # first sample round for multi-sample)
    envs = list(benchmark.make_envs(renderer, config))
    logger.info(f"  {len(envs)} examples loaded")

    # Resume: load completed examples from disk (only for single-sample mode;
    # pass@k mode always runs fresh to get exactly num_samples per example)
    completed: dict[str, _ResumedExample] = {}
    if config.save_dir and num_samples == 1:
        completed = _load_completed(config.save_dir, benchmark.name)
        if completed:
            logger.info(
                f"  Resuming: {len(completed)} already completed, "
                f"{len(envs) - len(completed)} remaining"
            )

    # Build policy
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    # Choose concurrency based on benchmark type
    max_concurrent = config.agent_concurrency if benchmark.multi_turn else config.concurrency
    sem = asyncio.Semaphore(max_concurrent)

    # Per-example results (None = not yet run or errored)
    rewards: list[float | None] = [None] * len(envs)
    metrics_list: list[Metrics] = [{} for _ in range(len(envs))]
    num_errors = 0
    num_completed = 0

    # Build idx -> example_id mapping and pre-fill from resumed results
    env_example_ids: list[str] = []
    for idx, env in enumerate(envs):
        eid = _get_example_id(env, idx, benchmark.name)
        env_example_ids.append(eid)
        if eid in completed:
            ex = completed[eid]
            rewards[idx] = ex.reward
            metrics_list[idx] = ex.metrics
            if ex.is_error:
                num_errors += 1

    total_to_run = sum(1 for eid in env_example_ids if eid not in completed)

    tokenizer = renderer.tokenizer

    async def run_one(idx: int, env) -> None:
        nonlocal num_completed, num_errors

        # Skip already-completed (matched by example_id, not positional idx)
        if env_example_ids[idx] in completed:
            return

        async with sem:
            t_start = time.monotonic()
            try:
                # Apply per-example timeout
                trajectory = await asyncio.wait_for(
                    do_single_rollout(policy, env),
                    timeout=config.timeout_seconds,
                )
                elapsed = time.monotonic() - t_start

                total_reward = sum(t.reward for t in trajectory.transitions)

                # Collect metrics from all transitions
                step_metrics: Metrics = {}
                for t in trajectory.transitions:
                    step_metrics.update(t.metrics)

                # Apply custom grade_fn override if provided
                if config.grade_fn is not None:
                    step_logs: Logs = {}
                    for t in trajectory.transitions:
                        step_logs.update(t.logs)
                    # Decode only the last action (assistant response)
                    last_action_tokens = trajectory.transitions[-1].ac.tokens
                    response = (
                        str(tokenizer.decode(last_action_tokens)) if last_action_tokens else ""
                    )
                    total_reward = config.grade_fn(response, step_logs)
                    step_metrics["custom_graded"] = 1.0

                rewards[idx] = total_reward
                metrics_list[idx] = step_metrics

                # Save trajectory with decoded text
                if config.save_dir:
                    stored = _trajectory_to_stored(
                        idx,
                        trajectory,
                        benchmark.name,
                        tokenizer,
                        elapsed,
                        example_id=_get_example_id(env, idx),
                    )
                    if config.grade_fn is not None:
                        stored.reward = total_reward
                    await _save_trajectory(config.save_dir, benchmark.name, stored)

            except (TimeoutError, EvalTimeoutError):
                elapsed = time.monotonic() - t_start
                logger.warning(
                    f"  {benchmark.name}[{idx}] timed out after {elapsed:.0f}s "
                    f"(limit={config.timeout_seconds}s)"
                )
                num_errors += 1
                rewards[idx] = 0.0  # Timeout = scored failure, reward=0

                if config.save_dir:
                    await _save_trajectory(
                        config.save_dir,
                        benchmark.name,
                        StoredTrajectory(
                            idx=idx,
                            benchmark=benchmark.name,
                            turns=[],
                            reward=0.0,
                            error=f"timeout ({config.timeout_seconds}s)",
                            time_seconds=elapsed,
                            example_id=_get_example_id(env, idx),
                        ),
                    )

            except Exception as e:
                elapsed = time.monotonic() - t_start
                logger.warning(f"  {benchmark.name}[{idx}] failed ({elapsed:.1f}s): {e}")
                num_errors += 1
                rewards[idx] = 0.0  # Error = scored failure, reward=0

                if config.save_dir:
                    await _save_trajectory(
                        config.save_dir,
                        benchmark.name,
                        StoredTrajectory(
                            idx=idx,
                            benchmark=benchmark.name,
                            turns=[],
                            reward=0.0,
                            error=str(e),
                            time_seconds=elapsed,
                            example_id=_get_example_id(env, idx),
                        ),
                    )

            finally:
                # Clean up env resources (e.g., sandboxes) on timeout or error
                if hasattr(env, "cleanup"):
                    with contextlib.suppress(Exception):
                        await env.cleanup()

            num_completed += 1
            if num_completed % max(1, total_to_run // 10) == 0 or num_completed == total_to_run:
                total_elapsed = time.monotonic() - t0
                logger.info(
                    f"  {benchmark.name}: {num_completed}/{total_to_run} done "
                    f"({total_elapsed:.0f}s elapsed, {num_errors} errors)"
                )

    # --- Single-sample mode (default, backward-compatible) ---
    if num_samples == 1:
        await asyncio.gather(*[run_one(i, env) for i, env in enumerate(envs)])

        # Aggregate
        valid_rewards = [r for r in rewards if r is not None]
        valid_metrics = [m for r, m in zip(rewards, metrics_list) if r is not None]

        # Count truncations (max_tokens_reached or context_overflow from EnvFromMessageEnv)
        num_truncated = sum(
            1
            for m in valid_metrics
            if m.get(METRIC_MAX_TOKENS_REACHED) or m.get(METRIC_CONTEXT_OVERFLOW)
        )

        result = benchmark.aggregate(valid_rewards, valid_metrics)
        result.num_errors = num_errors
        result.num_truncated = num_truncated
        result.time_seconds = time.monotonic() - t0

        trunc_str = f", {num_truncated} truncated" if num_truncated else ""
        completed_str = ""
        if num_truncated or num_errors:
            completed_str = (
                f" | completed={result.score_completed:.3f}"
                f" ({result.num_correct}/{result.num_completed})"
            )
        logger.info(
            f"  {benchmark.name}: score={result.score:.3f} "
            f"({result.num_correct}/{result.num_examples}, "
            f"{num_errors} errors{trunc_str}) "
            f"in {result.time_seconds:.0f}s{completed_str}"
        )

        if config.save_dir:
            _save_result(config.save_dir, result)

        return result

    # --- Multi-sample mode (pass@k) ---
    # Run the benchmark num_samples times, each time creating fresh envs.
    # Track per-example rewards keyed by example_id.
    per_example_rewards: dict[str, list[float]] = {}
    all_rewards: list[float] = []
    all_metrics: list[Metrics] = []
    total_errors = 0

    for sample_idx in range(num_samples):
        logger.info(f"  {benchmark.name}: sample {sample_idx + 1}/{num_samples}")

        # Create fresh envs for each sample (envs are single-use)
        if sample_idx == 0:
            sample_envs = envs  # Reuse the already-created first batch
        else:
            sample_envs = list(benchmark.make_envs(renderer, config))

        # Reset per-round state
        rewards = [None] * len(sample_envs)
        metrics_list = [{} for _ in range(len(sample_envs))]
        num_errors = 0
        num_completed = 0
        completed = {}  # No resumability in pass@k mode
        env_example_ids = [_get_example_id(e, i) for i, e in enumerate(sample_envs)]
        total_to_run = len(sample_envs)

        await asyncio.gather(*[run_one(i, env) for i, env in enumerate(sample_envs)])

        total_errors += num_errors

        # Collect per-example rewards using example_id
        for idx, (r, m) in enumerate(zip(rewards, metrics_list)):
            if r is None:
                continue
            per_example_rewards.setdefault(env_example_ids[idx], []).append(r)
            all_rewards.append(r)
            all_metrics.append(m)

    # Compute pass@k
    k_values = _choose_k_values(num_samples)
    pass_at_k = _compute_pass_at_k(per_example_rewards, k_values)

    # Count truncations across all samples
    total_truncated = sum(
        1 for m in all_metrics if m.get(METRIC_MAX_TOKENS_REACHED) or m.get(METRIC_CONTEXT_OVERFLOW)
    )

    # Aggregate using all rewards (gives overall accuracy across all samples)
    result = benchmark.aggregate(all_rewards, all_metrics)
    result.num_errors = total_errors
    result.num_truncated = total_truncated
    result.time_seconds = time.monotonic() - t0
    result.pass_at_k = pass_at_k

    # Log pass@k results
    pass_at_k_str = ", ".join(f"pass@{k}={v:.3f}" for k, v in sorted(pass_at_k.items()))
    logger.info(
        f"  {benchmark.name}: score={result.score:.3f} "
        f"({result.num_correct}/{result.num_examples}, {total_errors} errors) "
        f"in {result.time_seconds:.0f}s"
    )
    logger.info(f"  {benchmark.name}: {pass_at_k_str}")

    if config.save_dir:
        _save_result(config.save_dir, result)

    return result


async def run_benchmarks(
    names: list[str],
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    config: BenchmarkConfig = BenchmarkConfig(),
    parallel: bool = True,
) -> dict[str, BenchmarkResult]:
    """Run multiple benchmarks and save a combined summary.

    Each benchmark runs independently with its own save subdirectory and
    per-file locks. By default all benchmarks run concurrently; set
    ``parallel=False`` for sequential execution (useful for debugging).

    When ``config.save_dir`` is set, a ``summary.json`` is written with
    scores across all benchmarks after completion.

    Args:
        names: List of benchmark names from :data:`REGISTRY`
            (e.g. ``["gsm8k", "mmlu_pro", "ifeval"]``).
        sampling_client: Tinker sampling client for model generation.
        renderer: Renderer for tokenization and prompt building.
        config: Shared runtime configuration. See :class:`BenchmarkConfig`.
        parallel: If ``True`` (default), run all benchmarks concurrently
            via ``asyncio.gather``. If ``False``, run sequentially.

    Returns:
        Dict mapping benchmark name to its :class:`BenchmarkResult`.

    Example::

        results = await run_benchmarks(
            ["gsm8k", "mmlu_pro", "ifeval"],
            sampling_client,
            renderer,
            BenchmarkConfig(save_dir="evals/step500"),
        )
        for name, result in results.items():
            print(f"{name}: {result.score:.1%}")
    """
    if parallel:
        benchmark_results = await asyncio.gather(
            *[run_benchmark(name, sampling_client, renderer, config) for name in names]
        )
        results = dict(zip(names, benchmark_results))
    else:
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


# ---------------------------------------------------------------------------
# Result loading — for post-hoc analysis and trajectory browsing
# ---------------------------------------------------------------------------


def load_result(save_dir: str, benchmark_name: str) -> BenchmarkResult | None:
    """Load a saved BenchmarkResult from disk.

    Args:
        save_dir: Directory passed as ``BenchmarkConfig.save_dir``.
        benchmark_name: Benchmark name (e.g. ``"gsm8k"``).

    Returns:
        BenchmarkResult or None if not found.
    """
    path = Path(save_dir) / benchmark_name / "result.json"
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    # Convert pass_at_k string keys back to ints
    if "pass_at_k" in d:
        d["pass_at_k"] = {int(k): v for k, v in d["pass_at_k"].items()}
    return BenchmarkResult(**d)


def load_trajectories(
    save_dir: str,
    benchmark_name: str,
    correct_only: bool = False,
    incorrect_only: bool = False,
    errors_only: bool = False,
) -> list[StoredTrajectory]:
    """Load stored trajectories from disk with optional filtering.

    Args:
        save_dir: Directory passed as ``BenchmarkConfig.save_dir``.
        benchmark_name: Benchmark name (e.g. ``"gsm8k"``).
        correct_only: If True, return only trajectories with reward > 0.
        incorrect_only: If True, return only trajectories with reward == 0 and no error.
        errors_only: If True, return only trajectories with an error.

    Returns:
        List of StoredTrajectory objects.

    Example::

        # Browse incorrect examples
        wrong = load_trajectories("evals/step500", "gsm8k", incorrect_only=True)
        for t in wrong[:5]:
            print(f"Q: {t.logs.get('input', t.turns[0].content[:100])}")
            print(f"Expected: {t.logs.get('expected')}")
            print(f"Got: {t.logs.get('extracted')}")
            print()
    """
    path = Path(save_dir) / benchmark_name / "trajectories.jsonl"
    if not path.exists():
        return []

    trajectories = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                t = StoredTrajectory.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    f"Skipping malformed trajectory line {line_num} in "
                    f"{benchmark_name}/trajectories.jsonl: {e}"
                )
                continue
            if correct_only and t.reward <= 0:
                continue
            if incorrect_only and (t.reward > 0 or t.error is not None):
                continue
            if errors_only and t.error is None:
                continue
            trajectories.append(t)

    return trajectories


def load_summary(save_dir: str) -> dict[str, dict]:
    """Load the combined summary across all benchmarks.

    Args:
        save_dir: Directory passed as ``BenchmarkConfig.save_dir``.

    Returns:
        Dict mapping benchmark name to score/count info.

    Example::

        summary = load_summary("evals/step500")
        for name, info in summary.items():
            print(f"{name}: {info['score']:.1%}")
    """
    path = Path(save_dir) / "summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def print_trajectory(traj: StoredTrajectory) -> None:
    """Pretty-print a trajectory to the terminal.

    Args:
        traj: A StoredTrajectory to display.
    """
    reward_str = f"reward={traj.reward:.2f}"
    if traj.error:
        reward_str += f" ERROR: {traj.error}"
    print(f"--- [{traj.benchmark}#{traj.idx}] {reward_str} ({traj.time_seconds:.1f}s) ---")

    for turn in traj.turns:
        role_color = {"user": ">>", "assistant": "<<", "environment": "  ", "grader": "##"}.get(
            turn.role, "??"
        )
        content = turn.content
        if len(content) > 500:
            content = content[:500] + f"... ({len(content)} chars)"
        print(f"  {role_color} [{turn.role}] {content}")

    if traj.logs:
        print(f"  Logs: {traj.logs}")
    print()


def regrade_trajectories(
    save_dir: str,
    benchmark_name: str,
    grade_fn: Callable[[str, Logs], float],
    aggregate_fn: Callable[[list[float], list[Metrics]], BenchmarkResult] | None = None,
) -> BenchmarkResult:
    """Re-grade existing trajectories with a new grading function.

    Reads stored trajectories, applies ``grade_fn`` to each one, and
    re-aggregates the results — **without re-running the model**. This is
    useful for iterating on answer extraction logic after a costly eval run.

    Args:
        save_dir: Directory containing saved benchmark results.
        benchmark_name: Name of the benchmark (e.g. ``"gsm8k"``).
        grade_fn: ``(response, logs) -> reward``. ``response`` is the last
            assistant turn (thinking already stripped in stored trajectories).
            ``logs`` contains benchmark-specific fields (``expected``, etc.).
        aggregate_fn: Optional custom aggregation. If ``None``, computes
            simple accuracy (fraction with reward > 0).

    Returns:
        New BenchmarkResult with re-graded scores.

    Example::

        def my_grader(response: str, logs: Logs) -> float:
            import re
            expected = logs["expected"]
            # Look for \\boxed{answer} first, then last number
            match = re.search(r"\\\\boxed\\{(.+?)\\}", response)
            extracted = match.group(1) if match else ""
            return 1.0 if extracted.strip() == expected.strip() else 0.0

        result = regrade_trajectories("evals/step500", "gsm8k", my_grader)
        print(f"Re-graded: {result.score:.1%}")
    """
    trajectories = load_trajectories(save_dir, benchmark_name)

    rewards: list[float] = []
    regraded_metrics: list[Metrics] = []

    for traj in trajectories:
        if traj.error:
            rewards.append(0.0)
            regraded_metrics.append({})
            continue

        # Get the last assistant response
        response = _last_assistant_content(traj.turns)

        reward = grade_fn(response, traj.logs)
        rewards.append(reward)
        regraded_metrics.append(traj.metrics)

    if aggregate_fn is not None:
        return aggregate_fn(rewards, regraded_metrics)

    # Default: simple accuracy
    num_correct = sum(1 for r in rewards if r > 0)
    num_errors = sum(1 for t in trajectories if t.error)
    return BenchmarkResult(
        name=benchmark_name,
        score=num_correct / len(rewards) if rewards else 0.0,
        num_examples=len(rewards),
        num_correct=num_correct,
        num_errors=num_errors,
    )
