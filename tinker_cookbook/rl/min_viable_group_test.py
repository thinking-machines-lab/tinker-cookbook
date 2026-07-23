"""Tests for the MinViableGroup rollout strategy.

Covers: partial groups returned once the floor is met (with errors
populated), the below-floor raise feeding the group-drop path, the default
``ceil(0.75 * group_size)`` floor, retry consumption, and the per-rollout
timeout.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
import tinker

from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.exceptions import AllTrajectoriesFailedError
from tinker_cookbook.rl.rollout_strategy import MinViableGroup, RolloutResult
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, StepResult

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _AlwaysSamplePolicy(TokenCompleter):
    async def __call__(
        self, model_input: tinker.ModelInput, stop: Any, *, max_tokens: int | None = None
    ) -> TokensWithLogprobs:
        return TokensWithLogprobs(tokens=[1], maybe_logprobs=[0.0])


class _SucceedEnv(Env):
    async def initial_observation(self) -> tuple[tinker.ModelInput, list[str]]:
        return tinker.ModelInput.from_ints([1]), ["<stop>"]

    async def step(self, action: list[int], *, extra: Any = None) -> StepResult:
        return StepResult(
            reward=1.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=["<stop>"],
        )


class _SlowSucceedEnv(_SucceedEnv):
    """Succeeds after a short delay — lets scripted failures land first."""

    async def initial_observation(self) -> tuple[tinker.ModelInput, list[str]]:
        await asyncio.sleep(0.1)
        return await super().initial_observation()


class _FailEnv(Env):
    async def initial_observation(self) -> tuple[tinker.ModelInput, list[str]]:
        raise RuntimeError("scripted env failure")

    async def step(self, action: list[int], *, extra: Any = None) -> StepResult:
        raise AssertionError("unreachable")


class _HangEnv(Env):
    async def initial_observation(self) -> tuple[tinker.ModelInput, list[str]]:
        await asyncio.sleep(60.0)
        return tinker.ModelInput.from_ints([1]), ["<stop>"]

    async def step(self, action: list[int], *, extra: Any = None) -> StepResult:
        raise AssertionError("unreachable")


class _ScriptedBuilder(EnvGroupBuilder):
    """Replays scripted env batches: the first ``make_envs`` call returns the
    initial group; each subsequent call (a retry) returns the next
    single-env batch.  Running out of retry batches fails the test."""

    def __init__(self, initial_envs: Sequence[Env], retry_envs: Sequence[Env] | None = None):
        self._batches: list[list[Env]] = [list(initial_envs)] + [[e] for e in (retry_envs or [])]
        self.make_envs_calls = 0

    async def make_envs(self) -> Sequence[Env]:
        self.make_envs_calls += 1
        return self._batches.pop(0)


def _execute(strategy: MinViableGroup, builder: _ScriptedBuilder) -> RolloutResult:
    return asyncio.run(strategy.execute(builder, _AlwaysSamplePolicy()))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFloorMet:
    def test_partial_group_returned_with_errors_populated(self):
        """One member fails, its retry fails too; 3/4 successes meet the
        explicit floor of 3, so the partial group is returned with both
        failures recorded.  The successes are deliberately slow so the
        failure (and its retry) are processed while the floor is still
        unmet, making the retry deterministic."""
        builder = _ScriptedBuilder(
            initial_envs=[_SlowSucceedEnv(), _SlowSucceedEnv(), _SlowSucceedEnv(), _FailEnv()],
            retry_envs=[_FailEnv()],
        )
        strategy = MinViableGroup(min_successful=3, max_retries=1)

        result = _execute(strategy, builder)

        assert len(result.trajectories) == 3
        assert len(result.envs) == 3
        assert len(result.errors) == 2
        assert all(err.error_type == "RuntimeError" for err in result.errors)
        assert builder.make_envs_calls == 2  # initial group + one retry

    def test_full_group_when_nothing_fails(self):
        builder = _ScriptedBuilder(initial_envs=[_SucceedEnv() for _ in range(4)])
        result = _execute(MinViableGroup(min_successful=3), builder)
        assert len(result.trajectories) == 4
        assert result.errors == []

    def test_retry_success_restores_full_group(self):
        builder = _ScriptedBuilder(
            initial_envs=[_SucceedEnv(), _SucceedEnv(), _SucceedEnv(), _FailEnv()],
            retry_envs=[_SucceedEnv()],
        )
        result = _execute(MinViableGroup(min_successful=4, max_retries=3), builder)
        assert len(result.trajectories) == 4
        assert len(result.errors) == 1  # the original failure is still recorded


class TestBelowFloorRaises:
    def test_below_floor_after_budget_raises(self):
        """2/4 successes with floor 3 and no retries: the group is dropped
        via AllTrajectoriesFailedError (the existing group-drop path)."""
        builder = _ScriptedBuilder(
            initial_envs=[_SucceedEnv(), _SucceedEnv(), _FailEnv(), _FailEnv()],
        )
        strategy = MinViableGroup(min_successful=3, max_retries=0)

        with pytest.raises(AllTrajectoriesFailedError, match="2/4"):
            _execute(strategy, builder)

    def test_catches_group_errors_so_drop_not_crash(self):
        assert MinViableGroup().catches_group_errors is True


class TestDefaultFloor:
    """min_successful=None computes ceil(0.75 * group_size) at call time."""

    def test_group_of_4_floor_is_3(self):
        # ceil(0.75 * 4) = 3: exactly 3 successes pass ...
        builder = _ScriptedBuilder(
            initial_envs=[_SucceedEnv(), _SucceedEnv(), _SucceedEnv(), _FailEnv()],
        )
        result = _execute(MinViableGroup(max_retries=0), builder)
        assert len(result.trajectories) == 3

        # ... and 2 do not.
        builder = _ScriptedBuilder(
            initial_envs=[_SucceedEnv(), _SucceedEnv(), _FailEnv(), _FailEnv()],
        )
        with pytest.raises(AllTrajectoriesFailedError, match="minimum viable group size 3"):
            _execute(MinViableGroup(max_retries=0), builder)

    def test_group_of_2_floor_is_2(self):
        # ceil(0.75 * 2) = 2: one failure in a pair already drops the group.
        builder = _ScriptedBuilder(initial_envs=[_SucceedEnv(), _FailEnv()])
        with pytest.raises(AllTrajectoriesFailedError, match="minimum viable group size 2"):
            _execute(MinViableGroup(max_retries=0), builder)

    def test_group_of_8_floor_is_6(self):
        # ceil(0.75 * 8) = 6.
        builder = _ScriptedBuilder(
            initial_envs=[_SucceedEnv() for _ in range(6)] + [_FailEnv(), _FailEnv()],
        )
        result = _execute(MinViableGroup(max_retries=0), builder)
        assert len(result.trajectories) == 6
        assert len(result.errors) == 2


class TestPerRolloutTimeout:
    def test_hanging_rollout_treated_as_failure(self):
        """A rollout exceeding per_rollout_timeout is cancelled and recorded
        as a TimeoutError failure; the floor decides the outcome."""
        builder = _ScriptedBuilder(initial_envs=[_SucceedEnv(), _HangEnv()])
        strategy = MinViableGroup(min_successful=1, max_retries=0, per_rollout_timeout=0.05)

        result = _execute(strategy, builder)

        assert len(result.trajectories) == 1
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "TimeoutError"

    def test_hanging_rollout_below_floor_raises(self):
        builder = _ScriptedBuilder(initial_envs=[_SucceedEnv(), _HangEnv()])
        strategy = MinViableGroup(min_successful=2, max_retries=0, per_rollout_timeout=0.05)
        with pytest.raises(AllTrajectoriesFailedError):
            _execute(strategy, builder)

    def test_no_timeout_by_default(self):
        assert MinViableGroup().per_rollout_timeout is None
