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
from tinker_cookbook.rl.rollout_limits import TerminationRewardPolicy
from tinker_cookbook.rl.rollout_runner import run_rollout
from tinker_cookbook.rl.rollout_strategy import FailFast, RolloutStrategy
from tinker_cookbook.rl.types import (
    DirectEnvGroupBuilder,
    Env,
    EnvGroupBuilder,
    Metrics,
    StopReason,
    Trajectory,
    TrajectoryGroup,
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
        """Absorb error info from a single rollout result.

        If ``result`` is ``None`` (indicating the group was skipped entirely),
        the skipped-group counter is incremented.  Otherwise, each error
        recorded in ``result.rollout_errors`` is added to the cumulative
        error-type counts.

        Args:
            result (TrajectoryGroup | None): A completed trajectory group, or
                ``None`` if the group was skipped (e.g. all trajectories failed
                or rewards were constant).

        Example::

            counter = RolloutErrorCounter()
            result = await do_group_rollout(builder, policy)
            counter.ingest(result)
        """
        if result is None:
            self._groups_skipped += 1
            return
        for err in result.rollout_errors:
            self._counts[err.error_type] += 1

    def get_metrics(self, prefix: str = "rollout_errors") -> dict[str, float]:
        """Return cumulative error metrics as a flat dictionary.

        Counters are monotonically increasing across calls.  The returned
        dictionary is suitable for passing directly to ``ml_log.log_metrics``.

        Args:
            prefix (str): Key prefix for all emitted metric names.  Defaults to
                ``"rollout_errors"``.

        Returns:
            dict[str, float]: Mapping of ``"{prefix}/{error_type}"`` to counts,
                plus ``"{prefix}/total"`` and ``"{prefix}/groups_skipped"``
                summary keys.  Empty dict if no errors have been recorded.

        Example::

            counter = RolloutErrorCounter()
            counter.ingest(result)
            ml_log.log_metrics(counter.get_metrics(), step=step)
        """
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
    """Run a single rollout (one complete episode) in the given environment.

    Repeatedly queries the policy for actions and steps the environment until
    the episode terminates.  Env logging (if any) goes into whatever logtree
    scope the caller has set up.

    This is a thin wrapper over the single rollout-loop implementation,
    :func:`~tinker_cookbook.rl.rollout_runner.run_rollout`, called with no
    limits and no hooks (the unconfigured loop).  Use
    ``run_rollout`` directly to configure per-rollout budgets, timeouts, or
    hooks.

    Args:
        policy (TokenCompleter): The token-level policy used to generate
            actions (token sequences) from observations.
        env (Env): A single-use environment instance.  Must not be reused
            after this call returns.

    Returns:
        Trajectory: The complete sequence of transitions plus the final
            observation after the episode ends.  If the environment reports
            :class:`~tinker_cookbook.rl.types.InitialObservationOverflow`
            (initial prompt already over budget), the trajectory contains a
            single synthetic transition with empty observation/action and
            ``stop_reason="max_tokens"``, and no sampling call is made.

    Example::

        env = my_env_builder.build_one()
        policy = TinkerTokenCompleter(sampling_client, max_tokens=1024)
        trajectory = await do_single_rollout(policy, env)
    """
    return await run_rollout(policy, env)


def _should_skip_group_grading(
    termination: TerminationRewardPolicy | None, trajectories: list[Trajectory]
) -> bool:
    """Whether ``compute_group_rewards`` can be skipped entirely for this group.

    The per-trajectory triple condition for skip-grading is: the trajectory
    stopped with ``rollout_timeout``, AND ``zero_reward_on_limit`` is set, AND
    ``rollout_timeout`` is in ``limit_stop_reasons`` (so the graded reward
    would be clamped to at most 0.0 anyway).  ``compute_group_rewards`` is
    group-simultaneous — it grades all trajectories in one call — so the call
    is skipped only when *every* trajectory in the group satisfies the triple
    condition; if any member needs real grading, the full group is passed
    through unchanged (pairwise/group graders keep seeing the whole group)
    and the timed-out members' rewards are handled by the zero clamp instead.
    """
    return (
        termination is not None
        and termination.skip_grading_on_timeout
        and termination.zero_reward_on_limit
        and StopReason.ROLLOUT_TIMEOUT in termination.limit_stop_reasons
        and len(trajectories) > 0
        and all(traj.stop_reason == StopReason.ROLLOUT_TIMEOUT for traj in trajectories)
    )


def _apply_zero_reward_on_limit(
    termination: TerminationRewardPolicy,
    trajectories: list[Trajectory],
    rewards_G: list[float],
    metrics_G: list[Metrics],
) -> None:
    """Clamp limit-stopped trajectories' total rewards to ``min(reward, 0.0)``.

    Grade-then-clamp: called *after* ``compute_group_rewards``, keyed off
    ``Trajectory.stop_reason``.  The total reward of a trajectory is the sum
    of its per-step rewards plus the group-level reward, so the clamp is
    implemented by lowering the group-level reward until the total is 0.0
    (negative totals pass through unchanged).  Mutates ``rewards_G`` /
    ``metrics_G`` in place; clamped members are marked with
    ``metrics["zero_reward_on_limit"] = 1.0``.
    """
    if not termination.zero_reward_on_limit:
        return
    for i, traj in enumerate(trajectories):
        if traj.stop_reason is None or traj.stop_reason not in termination.limit_stop_reasons:
            continue
        step_sum = sum(transition.reward for transition in traj.transitions)
        if step_sum + rewards_G[i] > 0.0:
            # Setting the group-level reward to -step_sum makes the total
            # exactly 0.0 (avoids floating-point residue from subtraction).
            rewards_G[i] = -step_sum
            metrics_G[i] = {**metrics_G[i], "zero_reward_on_limit": 1.0}


@logtree.scope_header_decorator("Group Rollout")
async def do_group_rollout(
    env_group_builder: EnvGroupBuilder,
    policy: TokenCompleter,
    strategy: RolloutStrategy | None = None,
    termination: TerminationRewardPolicy | None = None,
) -> TrajectoryGroup:
    """Run rollouts for all environments in a group and compute group rewards.

    Creates environments from ``env_group_builder``, executes rollouts using
    the given ``strategy``, computes group-level rewards, and logs per-trajectory
    details via logtree.  Cleanup of the environment group is guaranteed even
    if an error occurs.

    Args:
        env_group_builder (EnvGroupBuilder): Builder that produces the set of
            environments for this group and computes group-level rewards.
        policy (TokenCompleter): The token-level policy used to generate
            actions during each rollout.
        strategy (RolloutStrategy | None): Controls how trajectories are
            collected (error handling, retries, etc.).  Defaults to
            :class:`FailFast` which preserves the original fail-on-any-error
            behaviour.
        termination (TerminationRewardPolicy | None): Stop-reason-keyed reward
            semantics, applied around and after ``compute_group_rewards``:
            the group grading call is bounded by ``grader_timeout_seconds``,
            skipped entirely when every trajectory satisfies the
            skip-grading-on-timeout triple condition, and limit-stopped
            trajectories are clamped to ``min(reward, 0.0)`` afterwards
            (grade-then-clamp).  ``None`` (default) leaves rewards untouched.

    Returns:
        TrajectoryGroup: The collected trajectories, per-trajectory rewards,
            per-trajectory metrics, and any rollout errors.

    Raises:
        AllTrajectoriesFailedError: If the strategy exhausts all retries
            without producing any successful trajectories (propagated from
            the strategy).
        TimeoutError: If ``termination.grader_timeout_seconds`` is set and
            ``compute_group_rewards`` exceeds it (a group-level error, caught
            by strategies with ``catches_group_errors``).

    Example::

        group = await do_group_rollout(env_group_builder, policy)
        rewards = group.get_total_rewards()
    """
    if strategy is None:
        strategy = FailFast()
    try:
        if isinstance(env_group_builder, DirectEnvGroupBuilder):
            return await env_group_builder.rollout_group(policy)

        result = await strategy.execute(env_group_builder, policy)

        if _should_skip_group_grading(termination, result.trajectories):
            # Every trajectory timed out and would be clamped to <= 0.0
            # anyway: skip the (possibly expensive) grading call and use 0.0.
            rewards_and_metrics_G = [(0.0, {}) for _ in result.trajectories]
        else:
            async with trace.scope_span("compute_group_rewards"):
                grader_timeout = (
                    termination.grader_timeout_seconds if termination is not None else None
                )
                async with asyncio.timeout(grader_timeout):
                    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(
                        result.trajectories, result.envs
                    )
        rewards_zip_G, metrics_zip_G = zip(*rewards_and_metrics_G, strict=True)
        rewards_G: list[float] = list(rewards_zip_G)
        metrics_G: list[Metrics] = list(metrics_zip_G)

        if termination is not None:
            _apply_zero_reward_on_limit(termination, result.trajectories, rewards_G, metrics_G)

        with logtree.scope_header("Trajectory Details"):
            for traj_idx, (traj, final_reward) in enumerate(
                zip(result.trajectories, rewards_G, strict=True)
            ):
                with logtree.scope_header(f"Trajectory {traj_idx} Episode"):
                    _log_single_trajectory_details(traj, final_reward)

        return TrajectoryGroup(
            result.trajectories, rewards_G, metrics_G, rollout_errors=result.errors
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

    Pass any ``concurrent.futures.Executor`` -- ``ProcessPoolExecutor`` works
    out of the box, or wrap Ray / custom cluster dispatchers as ``Executor``.

    Pass ``None`` to revert to the default in-process async behavior.

    Args:
        executor (Executor | None): The executor to use for dispatching
            rollout tasks, or ``None`` to run rollouts in-process.

    Example::

        from concurrent.futures import ProcessPoolExecutor

        set_rollout_executor(ProcessPoolExecutor(max_workers=4))
        # ... run training ...
        set_rollout_executor(None)  # revert to in-process
    """
    _rollout_executor.set(executor)


def get_rollout_executor() -> Executor | None:
    """Get the current rollout executor.

    Returns:
        Executor | None: The active executor, or ``None`` if rollouts run
            as in-process async coroutines (the default).
    """
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
    termination: TerminationRewardPolicy | None = None


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
            termination=task.termination,
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
    termination: TerminationRewardPolicy | None = None,
) -> TrajectoryGroup | None:
    """Run a group rollout, optionally dispatching to an external executor.

    When a rollout executor is set (via :func:`set_rollout_executor`), inputs
    are bundled into a pickleable ``_RolloutTask`` and dispatched via
    ``loop.run_in_executor()``.  Otherwise, runs as an asyncio coroutine
    in the current process (zero overhead).

    Groups where all trajectories receive the same reward are optionally
    filtered out (returned as ``None``), since they provide no gradient
    signal for advantage-based RL algorithms.

    Args:
        sampling_client (tinker.SamplingClient): Tinker sampling client used
            to construct the token-level policy.
        env_group_builder (EnvGroupBuilder): Builder that produces the set of
            environments for this group and computes group-level rewards.
        max_tokens (int): Maximum number of tokens the policy may generate
            per action.
        temperature (float): Sampling temperature for the policy.
        do_remove_constant_reward_groups (bool): If ``True``, return ``None``
            when all trajectories in the group receive identical total rewards.
        enable_logging (bool): Whether to enable logtree logging for this
            rollout.  Defaults to ``True``.
        strategy (RolloutStrategy | None): Controls how trajectories are
            collected within the group (error handling, retries, etc.).
            Defaults to :class:`FailFast`.
        termination (TerminationRewardPolicy | None): Stop-reason-keyed
            reward semantics (see :func:`do_group_rollout`).  ``None``
            (default) leaves rewards untouched.

    Returns:
        TrajectoryGroup | None: The completed trajectory group, or ``None``
            if the group was skipped due to constant rewards or a caught
            rollout error.

    Example::

        result = await do_group_rollout_and_filter_constant_reward(
            sampling_client=sampler,
            env_group_builder=builder,
            max_tokens=1024,
            temperature=0.7,
            do_remove_constant_reward_groups=True,
        )
        if result is not None:
            rewards = result.get_total_rewards()
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
            termination=termination,
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
        termination=termination,
    )


async def _do_group_rollout_and_filter_constant_reward_impl(
    sampling_client: tinker.SamplingClient,
    env_group_builder: EnvGroupBuilder,
    max_tokens: int,
    temperature: float,
    do_remove_constant_reward_groups: bool,
    enable_logging: bool = True,
    strategy: RolloutStrategy | None = None,
    termination: TerminationRewardPolicy | None = None,
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
                termination=termination,
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
