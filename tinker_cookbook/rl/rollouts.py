import asyncio
import logging
import numbers
from collections import Counter
from concurrent.futures import Executor
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import tinker

from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter
from tinker_cookbook.exceptions import AllTrajectoriesFailedError
from tinker_cookbook.rl.rollout_strategy import FailFast, RolloutStrategy
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)
from tinker_cookbook.utils import logtree, trace
from tinker_cookbook.utils.misc_utils import all_same

logger = logging.getLogger(__name__)


@dataclass
class RolloutErrorCounter:
    """Accumulates rollout error counts from :class:`TrajectoryGroup` results.

    Lives in the main event loop only — never crosses thread/process boundaries.
    Error information reaches the counter via :attr:`TrajectoryGroup.rollout_errors`,
    which is embedded in the return value (pickleable, safe for any executor).
    """

    _counts: Counter[str] = field(default_factory=Counter)
    _groups_skipped: int = 0

    def ingest(self, result: TrajectoryGroup | None) -> None:
        """Absorb error info from a single rollout result."""
        if result is None:
            self._groups_skipped += 1
            return
        for err in result.rollout_errors:
            self._counts[err.error_type] += 1

    def get_metrics(self, prefix: str = "rollout_errors") -> dict[str, float]:
        """Return cumulative error metrics (monotonically increasing)."""
        out: dict[str, float] = {}
        if self._counts or self._groups_skipped > 0:
            for k, v in self._counts.items():
                out[f"{prefix}/{k}"] = float(v)
            out[f"{prefix}/total"] = float(sum(self._counts.values()))
            out[f"{prefix}/groups_skipped"] = float(self._groups_skipped)
        return out


def _log_transition_logs(logs: dict[str, Any]) -> None:
    """Render transition logs in a readable structure without truncating table cells."""
    if not logs:
        return
    with logtree.scope_header("Diagnostics"):
        for key, value in logs.items():
            text = str(value)
            if "\n" in text or len(text) > 120:
                logtree.details(text, summary=key, pre=True)
            else:
                logtree.log_text(f"{key}: {text}")


def _log_transition_metrics(metrics: dict[str, Any] | None) -> None:
    """Render transition metrics in a compact, always-visible table."""
    if not metrics:
        return
    formatted_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, numbers.Real):
            formatted_metrics[key] = f"{float(value):.3f}"
        else:
            formatted_metrics[key] = str(value)
    with logtree.scope_header("Step Metrics"):
        logtree.table_from_dict(
            formatted_metrics,
            caption="Metrics emitted by env.step",
        )


def _log_single_trajectory_details(traj: Trajectory, final_reward: float) -> None:
    with logtree.scope_header("Episode Details"):
        for turn_idx, transition in enumerate(traj.transitions, start=1):
            with logtree.scope_header(f"Turn {turn_idx}"):
                logtree.table_from_dict(
                    {
                        "ob_len": transition.ob.length,
                        "ac_len": len(transition.ac.tokens),
                        "step_reward": f"{transition.reward:.3f}",
                    },
                    caption="Step stats",
                )
                _log_transition_metrics(transition.metrics)
                _log_transition_logs(transition.logs)

        logtree.table_from_dict(
            {
                "num_turns": len(traj.transitions),
                "final_ob_len": traj.final_ob.length,
                "sum_step_rewards": f"{sum(t.reward for t in traj.transitions):.3f}",
                "final_group_reward": f"{final_reward:.3f}",
                "total_return": f"{sum(t.reward for t in traj.transitions) + final_reward:.3f}",
            },
            caption="Episode totals",
        )


async def do_single_rollout(policy: TokenCompleter, env: Env) -> Trajectory:
    """Run a single rollout (env episode). Env logging (if any) goes into
    whatever logtree scope the caller has set up."""
    transitions = []
    async with trace.scope_span("env_initial_observation"):
        ob, stop_condition = await env.initial_observation()
    while True:
        async with trace.scope_span("policy_sample"):
            ac_with_logprobs = await policy(ob, stop_condition)
        async with trace.scope_span("env_step"):
            step_result = await env.step(ac_with_logprobs.tokens)
        transition = Transition(
            ob=ob,
            ac=ac_with_logprobs,
            reward=step_result.reward,
            episode_done=step_result.episode_done,
            metrics=step_result.metrics,
            logs=step_result.logs,
        )
        transitions.append(transition)
        ob = step_result.next_observation
        stop_condition = step_result.next_stop_condition
        if step_result.episode_done:
            break
    trajectory = Trajectory(transitions=transitions, final_ob=ob)
    if env.trajectory_hook is not None:
        await env.trajectory_hook(trajectory)
    return trajectory


@logtree.scope_header_decorator("Group Rollout")
async def do_group_rollout(
    env_group_builder: EnvGroupBuilder,
    policy: TokenCompleter,
    strategy: RolloutStrategy | None = None,
) -> TrajectoryGroup:
    """Run rollouts for all environments in a group and compute group rewards.

    Args:
        strategy: Controls how trajectories are collected (error handling,
            retries, etc.).  Defaults to :class:`FailFast` which preserves
            the original fail-on-any-error behaviour.
    """
    if strategy is None:
        strategy = FailFast()
    try:
        result = await strategy.execute(env_group_builder, policy)

        async with trace.scope_span("compute_group_rewards"):
            rewards_and_metrics_G = await env_group_builder.compute_group_rewards(
                result.trajectories, result.envs
            )
        rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)

        with logtree.scope_header("Trajectory Details"):
            for traj_idx, (traj, final_reward) in enumerate(
                zip(result.trajectories, rewards_G, strict=True)
            ):
                with logtree.scope_header(f"Trajectory {traj_idx} Episode"):
                    _log_single_trajectory_details(traj, final_reward)

        return TrajectoryGroup(
            result.trajectories, list(rewards_G), list(metrics_G), rollout_errors=result.errors
        )
    finally:
        # cleanup() is not wrapped in try/except; implementations must handle failures
        # internally and not raise, or exceptions here will mask rollout errors.
        await env_group_builder.cleanup()


# ---------------------------------------------------------------------------
# Rollout executor — allows offloading group rollouts to processes/Ray/etc.
# ---------------------------------------------------------------------------

_rollout_executor: ContextVar[Executor | None] = ContextVar("rollout_executor", default=None)


def set_rollout_executor(executor: Executor | None) -> None:
    """Set the executor used for group rollouts.

    When set, ``do_group_rollout_and_filter_constant_reward`` dispatches each
    rollout via ``loop.run_in_executor(executor, ...)`` instead of running it
    as an asyncio coroutine in the current process.

    Pass any ``concurrent.futures.Executor`` — ``ProcessPoolExecutor`` works
    out of the box, or wrap Ray / custom cluster dispatchers as ``Executor``.

    Pass ``None`` to revert to the default in-process async behavior.
    """
    _rollout_executor.set(executor)


def get_rollout_executor() -> Executor | None:
    """Get the current rollout executor (None = in-process async)."""
    return _rollout_executor.get()


@dataclass(frozen=True)
class _RolloutTask:
    """Pickleable bundle of inputs for cross-process rollout dispatch."""

    sampling_client: tinker.SamplingClient
    env_group_builder: EnvGroupBuilder
    max_tokens: int
    temperature: float
    remove_constant_reward_groups: bool
    enable_logging: bool
    strategy: RolloutStrategy = field(default_factory=FailFast)


def _run_rollout_sync(task: _RolloutTask) -> TrajectoryGroup | None:
    """Entry point for executor workers. Runs the async rollout in a fresh event loop.

    Called by ``loop.run_in_executor()`` — must be a module-level sync function
    so it can be pickled for ``ProcessPoolExecutor``.
    """
    return asyncio.run(
        _do_group_rollout_and_filter_constant_reward_impl(
            task.sampling_client,
            task.env_group_builder,
            task.max_tokens,
            task.temperature,
            task.remove_constant_reward_groups,
            task.enable_logging,
            strategy=task.strategy,
        )
    )


@trace.scope
async def do_group_rollout_and_filter_constant_reward(
    sampling_client: tinker.SamplingClient,
    env_group_builder: EnvGroupBuilder,
    max_tokens: int,
    temperature: float,
    do_remove_constant_reward_groups: bool,
    enable_logging: bool = True,
    strategy: RolloutStrategy | None = None,
) -> TrajectoryGroup | None:
    """Run a group rollout, optionally dispatching to an external executor.

    When a rollout executor is set (via ``set_rollout_executor``), inputs are
    bundled into a pickleable ``_RolloutTask`` and dispatched via
    ``loop.run_in_executor()``. Otherwise, runs as an asyncio coroutine
    in the current process (zero overhead).

    Args:
        strategy: Controls how trajectories are collected within the group
            (error handling, retries, etc.).  Defaults to :class:`FailFast`.
    """
    if strategy is None:
        strategy = FailFast()

    executor = get_rollout_executor()
    if executor is not None:
        task = _RolloutTask(
            sampling_client=sampling_client,
            env_group_builder=env_group_builder,
            max_tokens=max_tokens,
            temperature=temperature,
            remove_constant_reward_groups=do_remove_constant_reward_groups,
            enable_logging=enable_logging,
            strategy=strategy,
        )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, _run_rollout_sync, task)

    return await _do_group_rollout_and_filter_constant_reward_impl(
        sampling_client,
        env_group_builder,
        max_tokens,
        temperature,
        do_remove_constant_reward_groups,
        enable_logging,
        strategy=strategy,
    )


async def _do_group_rollout_and_filter_constant_reward_impl(
    sampling_client: tinker.SamplingClient,
    env_group_builder: EnvGroupBuilder,
    max_tokens: int,
    temperature: float,
    do_remove_constant_reward_groups: bool,
    enable_logging: bool = True,
    strategy: RolloutStrategy | None = None,
) -> TrajectoryGroup | None:
    if strategy is None:
        strategy = FailFast()

    policy = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens, temperature=temperature)

    try:
        with logtree.optional_enable_logging(enable_logging):
            trajectory_group = await do_group_rollout(
                env_group_builder,
                policy,
                strategy=strategy,
            )
    except AllTrajectoriesFailedError as e:
        # All retries exhausted — already logged per-trajectory inside the strategy
        logger.warning(str(e))
        return None
    except Exception as e:
        if not strategy.catches_group_errors:
            raise
        logger.warning(f"Rollout error ({type(e).__name__}), skipping group: {e}")
        return None

    # Remove if all trajectories have the same reward
    if do_remove_constant_reward_groups and all_same(trajectory_group.get_total_rewards()):
        return None
    return trajectory_group
