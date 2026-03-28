"""Pluggable strategies for collecting trajectories within a rollout group.

A :class:`RolloutStrategy` decides *how* to run N single-rollout coroutines
in parallel — whether to fail fast, retry on failure, etc.  It owns the full
trajectory collection lifecycle including env creation (via
``EnvGroupBuilder.make_envs()``), so strategies like retry can create fresh
envs as needed.

Group reward computation and logging remain in
:func:`~tinker_cookbook.rl.rollouts.do_group_rollout`.

Implementations must be pickleable (frozen dataclasses with primitive fields)
because they are bundled into ``_RolloutTask`` for cross-process dispatch.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from tinker_cookbook.completers import TokenCompleter
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RolloutError, Trajectory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutResult:
    """Output of a :class:`RolloutStrategy`.

    Attributes:
        trajectories (list[Trajectory]): Successfully completed trajectories.
        envs (Sequence[Env]): The environments that produced the successful
            trajectories, in the same order as ``trajectories``.
        errors (list[RolloutError]): Errors encountered during rollout
            collection.  Empty when all rollouts succeed.
    """

    trajectories: list[Trajectory]
    envs: Sequence[Env]
    errors: list[RolloutError]


class RolloutStrategy(ABC):
    """Controls how trajectories are collected from a group of environments.

    Subclasses implement :meth:`execute` which receives the
    :class:`EnvGroupBuilder` and a policy, creates envs, runs rollouts,
    and returns the surviving trajectories plus any error info.

    Implementations must be pickleable — use ``@dataclass(frozen=True)``
    with only primitive fields.
    """

    @property
    def catches_group_errors(self) -> bool:
        """If True, group-level errors (``make_envs``, ``compute_group_rewards``)
        are caught and the group is skipped.  If False, they propagate."""
        return False

    @abstractmethod
    async def execute(
        self,
        env_group_builder: EnvGroupBuilder,
        policy: TokenCompleter,
    ) -> RolloutResult:
        """Create envs, run rollouts, and return results.

        May raise on unrecoverable errors (e.g. retry budget exhausted).
        The caller (:func:`do_group_rollout`) handles group-level error
        recovery based on :attr:`catches_group_errors`.

        Args:
            env_group_builder (EnvGroupBuilder): Builder used to create the
                environments for this rollout group.
            policy (TokenCompleter): The policy (language model) used to
                generate actions during rollouts.

        Returns:
            RolloutResult: The collected trajectories, surviving environments,
                and any errors encountered.
        """
        ...


@dataclass(frozen=True)
class FailFast(RolloutStrategy):
    """Default strategy: any trajectory error crashes the group.

    Produces identical behaviour to the original ``asyncio.gather(...)``
    path — no error tolerance, no overhead.
    """

    async def execute(
        self,
        env_group_builder: EnvGroupBuilder,
        policy: TokenCompleter,
    ) -> RolloutResult:
        """Run all rollouts concurrently, raising immediately on any failure.

        Args:
            env_group_builder (EnvGroupBuilder): Builder used to create the
                environments for this rollout group.
            policy (TokenCompleter): The policy used to generate actions.

        Returns:
            RolloutResult: Result with all trajectories and an empty error list.

        Raises:
            Exception: Any exception raised by a single rollout propagates
                immediately, cancelling the remaining rollouts.
        """
        from tinker_cookbook.rl.rollouts import do_single_rollout

        envs = await env_group_builder.make_envs()
        trajectories: list[Trajectory] = list(
            await asyncio.gather(*[do_single_rollout(policy, env) for env in envs])
        )
        return RolloutResult(trajectories=trajectories, envs=envs, errors=[])


@dataclass(frozen=True)
class RetryOnFailure(RolloutStrategy):
    """Retry failed trajectories with fresh environments.

    When a trajectory fails (container crash, sandbox flake, transient error),
    a fresh env is created via ``make_envs()`` and the rollout is retried.
    This continues until either all trajectories succeed or the retry budget
    is exhausted.

    If the retry budget is exhausted and a failure still occurs, the remaining
    in-flight tasks are cancelled and the exception is re-raised. This avoids
    partial-group bias from training on an incomplete set of trajectories.

    Uses ``asyncio.wait(FIRST_COMPLETED)`` so retries start as soon as a
    failure is detected, without waiting for other in-flight rollouts.

    Args:
        max_retries: Total retry budget across all trajectories in the group.
            For example, with ``max_retries=3`` and a group of 8 envs, up to
            3 individual trajectory failures will be retried.
    """

    max_retries: int = 3

    @property
    def catches_group_errors(self) -> bool:
        return True

    async def execute(
        self,
        env_group_builder: EnvGroupBuilder,
        policy: TokenCompleter,
    ) -> RolloutResult:
        """Run rollouts with automatic retry on individual trajectory failures.

        Creates environments, launches all rollouts concurrently, and retries
        any that fail by creating a fresh environment.  Uses
        ``asyncio.wait(FIRST_COMPLETED)`` so retries begin immediately upon
        detecting a failure.

        Args:
            env_group_builder (EnvGroupBuilder): Builder used to create (and
                re-create on retry) environments for this rollout group.
            policy (TokenCompleter): The policy used to generate actions.

        Returns:
            RolloutResult: Result containing the successfully completed
                trajectories, surviving environments, and a list of any
                errors encountered (including retried ones).

        Raises:
            Exception: Re-raises the failing exception when the retry budget
                is exhausted, after cancelling all remaining in-flight tasks.
        """
        from tinker_cookbook.rl.rollouts import do_single_rollout

        envs = await env_group_builder.make_envs()

        # Map task -> env for tracking
        task_to_env: dict[asyncio.Task[Trajectory], Env] = {}
        for env in envs:
            task = asyncio.create_task(do_single_rollout(policy, env))
            task_to_env[task] = env

        trajectories: list[Trajectory] = []
        surviving_envs: list[Env] = []
        errors: list[RolloutError] = []
        retries_remaining = self.max_retries
        pending: set[asyncio.Task[Trajectory]] = set(task_to_env.keys())

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    traj = task.result()
                    trajectories.append(traj)
                    surviving_envs.append(task_to_env[task])
                except (asyncio.CancelledError, KeyboardInterrupt):
                    # Never swallow cancellation — cancel remaining and propagate
                    for t in pending:
                        t.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    raise
                except Exception as exc:
                    logger.warning(
                        "Trajectory failed (%s): %s (retries_remaining=%d)",
                        type(exc).__name__,
                        exc,
                        retries_remaining,
                    )
                    errors.append(
                        RolloutError(
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                    )
                    if retries_remaining > 0:
                        retries_remaining -= 1
                        # Create a fresh env for retry.
                        # Note: make_envs() creates a full group but we only need one.
                        # The extras are cheap Python objects for most envs; for sandbox
                        # envs the unused containers get GC'd.
                        new_envs = await env_group_builder.make_envs()
                        new_env = new_envs[0]
                        new_task = asyncio.create_task(do_single_rollout(policy, new_env))
                        task_to_env[new_task] = new_env
                        pending.add(new_task)
                    else:
                        # Budget exhausted — cancel remaining and re-raise.
                        # This avoids partial-group bias from training on an
                        # incomplete group of trajectories.
                        logger.error(
                            "Retry budget exhausted (%d retries), cancelling remaining tasks",
                            self.max_retries,
                        )
                        for t in pending:
                            t.cancel()
                        await asyncio.gather(*pending, return_exceptions=True)
                        raise exc

        return RolloutResult(
            trajectories=trajectories,
            envs=surviving_envs,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Config mapping
# ---------------------------------------------------------------------------


def rollout_strategy_from_config(
    rollout_error_tolerance: bool | RolloutStrategy,
) -> RolloutStrategy:
    """Convert a ``Config.rollout_error_tolerance`` value to a :class:`RolloutStrategy`.

    - ``False`` -> :class:`FailFast` (crash on any error, the default)
    - ``True``  -> :class:`RetryOnFailure` with default ``max_retries=3``
    - A :class:`RolloutStrategy` instance -> passed through as-is

    Args:
        rollout_error_tolerance (bool | RolloutStrategy): The config value to
            convert.  ``False`` for fail-fast, ``True`` for retry with
            defaults, or a pre-built :class:`RolloutStrategy` instance.

    Returns:
        RolloutStrategy: The resolved strategy instance.

    Raises:
        ConfigurationError: If the value is not a bool or
            :class:`RolloutStrategy`.
    """
    if isinstance(rollout_error_tolerance, RolloutStrategy):
        return rollout_error_tolerance
    if rollout_error_tolerance is False:
        return FailFast()
    if rollout_error_tolerance is True:
        return RetryOnFailure()
    raise ConfigurationError(f"Invalid rollout_error_tolerance value: {rollout_error_tolerance!r}")
