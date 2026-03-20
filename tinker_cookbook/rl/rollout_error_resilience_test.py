"""Tests for rollout error resilience: strategy abstraction, retry, error tracking, and pickling."""

from __future__ import annotations

import asyncio
import pickle
from unittest.mock import MagicMock, patch

import pytest
import tinker

from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.exceptions import AllTrajectoriesFailedError, ConfigurationError
from tinker_cookbook.rl.rollout_strategy import (
    FailFast,
    RetryOnFailure,
    rollout_strategy_from_config,
)
from tinker_cookbook.rl.rollouts import (
    RolloutErrorCounter,
    _do_group_rollout_and_filter_constant_reward_impl,
    do_group_rollout,
)
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    RolloutError,
    StepResult,
    Trajectory,
    TrajectoryGroup,
    Transition,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trajectory() -> Trajectory:
    """Create a minimal valid Trajectory."""
    return Trajectory(
        transitions=[
            Transition(
                ob=tinker.ModelInput.from_ints([1, 2, 3]),
                ac=TokensWithLogprobs(tokens=[4, 5], maybe_logprobs=[0.1, 0.2]),
                reward=1.0,
                episode_done=True,
            )
        ],
        final_ob=tinker.ModelInput.from_ints([]),
    )


class _FakePolicy(TokenCompleter):
    """Policy that returns a fixed result, optionally failing on specific call indices."""

    def __init__(self, fail_indices: set[int] | None = None, error: BaseException | None = None):
        self._call_count = 0
        self.fail_indices = fail_indices or set()
        self.error = error or RuntimeError("fake error")

    async def __call__(self, model_input, stop):
        idx = self._call_count
        self._call_count += 1
        if idx in self.fail_indices:
            raise self.error
        return TokensWithLogprobs(tokens=[4, 5], maybe_logprobs=[0.1, 0.2])


class _FakeEnv(Env):
    async def initial_observation(self):
        return tinker.ModelInput.from_ints([1, 2, 3]), [0]

    async def step(self, action):
        return StepResult(
            reward=1.0,
            episode_done=True,
            next_observation=tinker.ModelInput.from_ints([]),
            next_stop_condition=[0],
        )


class _FakeEnvGroupBuilder(EnvGroupBuilder):
    def __init__(self, n_envs: int = 4):
        self.n_envs = n_envs
        self.make_envs_call_count = 0

    async def make_envs(self):
        self.make_envs_call_count += 1
        return [_FakeEnv() for _ in range(self.n_envs)]


# ---------------------------------------------------------------------------
# rollout_strategy_from_config tests
# ---------------------------------------------------------------------------


class TestRolloutStrategyFromConfig:
    def test_false_returns_fail_fast(self):
        strategy = rollout_strategy_from_config(False)
        assert isinstance(strategy, FailFast)
        assert not strategy.catches_group_errors

    def test_true_returns_retry_on_failure(self):
        strategy = rollout_strategy_from_config(True)
        assert isinstance(strategy, RetryOnFailure)
        assert strategy.max_retries == 3
        assert strategy.catches_group_errors

    def test_strategy_instance_passed_through(self):
        strategy = RetryOnFailure(max_retries=5)
        assert rollout_strategy_from_config(strategy) is strategy

    def test_fail_fast_instance_passed_through(self):
        strategy = FailFast()
        assert rollout_strategy_from_config(strategy) is strategy

    def test_invalid_value_raises(self):
        with pytest.raises(ConfigurationError):
            rollout_strategy_from_config(0.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Strategy pickling tests
# ---------------------------------------------------------------------------


class TestStrategyPickle:
    def test_fail_fast_pickleable(self):
        strategy = FailFast()
        restored = pickle.loads(pickle.dumps(strategy))
        assert isinstance(restored, FailFast)

    def test_retry_on_failure_pickleable(self):
        strategy = RetryOnFailure(max_retries=5)
        restored = pickle.loads(pickle.dumps(strategy))
        assert isinstance(restored, RetryOnFailure)
        assert restored.max_retries == 5


# ---------------------------------------------------------------------------
# RolloutError and TrajectoryGroup tests
# ---------------------------------------------------------------------------


class TestRolloutError:
    def test_pickleable(self):
        err = RolloutError(error_type="BadRequestError", error_message="context overflow")
        restored = pickle.loads(pickle.dumps(err))
        assert restored.error_type == "BadRequestError"
        assert restored.error_message == "context overflow"


class TestTrajectoryGroupErrors:
    def test_default_no_errors(self):
        tg = TrajectoryGroup([_make_trajectory()], [1.0], [{}])
        assert tg.rollout_errors == []

    def test_with_errors(self):
        errors = [RolloutError("BadRequestError", "too long")]
        tg = TrajectoryGroup([_make_trajectory()], [1.0], [{}], rollout_errors=errors)
        assert len(tg.rollout_errors) == 1

    def test_pickleable_with_errors(self):
        errors = [RolloutError("BadRequestError", "too long")]
        tg = TrajectoryGroup([_make_trajectory()], [1.0], [{}], rollout_errors=errors)
        restored = pickle.loads(pickle.dumps(tg))
        assert len(restored.rollout_errors) == 1

    def test_get_total_rewards_unaffected(self):
        errors = [RolloutError("Err", "msg")]
        tg = TrajectoryGroup(
            [_make_trajectory(), _make_trajectory()],
            [0.5, 0.3],
            [{}, {}],
            rollout_errors=errors,
        )
        rewards = tg.get_total_rewards()
        assert rewards[0] == pytest.approx(1.5)
        assert rewards[1] == pytest.approx(1.3)


# ---------------------------------------------------------------------------
# RolloutErrorCounter tests
# ---------------------------------------------------------------------------


class TestRolloutErrorCounter:
    def test_ingest_successful_group(self):
        counter = RolloutErrorCounter()
        tg = TrajectoryGroup([_make_trajectory()], [1.0], [{}])
        counter.ingest(tg)
        assert counter.get_metrics() == {}

    def test_ingest_none_increments_groups_skipped(self):
        counter = RolloutErrorCounter()
        counter.ingest(None)
        counter.ingest(None)
        metrics = counter.get_metrics()
        assert metrics["rollout_errors/groups_skipped"] == 2.0

    def test_ingest_group_with_errors(self):
        counter = RolloutErrorCounter()
        errors = [
            RolloutError("BadRequestError", "msg1"),
            RolloutError("BadRequestError", "msg2"),
            RolloutError("TimeoutError", "msg3"),
        ]
        tg = TrajectoryGroup([_make_trajectory()], [1.0], [{}], rollout_errors=errors)
        counter.ingest(tg)
        metrics = counter.get_metrics()
        assert metrics["rollout_errors/BadRequestError"] == 2.0
        assert metrics["rollout_errors/TimeoutError"] == 1.0
        assert metrics["rollout_errors/total"] == 3.0

    def test_cumulative_across_ingests(self):
        counter = RolloutErrorCounter()
        tg1 = TrajectoryGroup(
            [_make_trajectory()],
            [1.0],
            [{}],
            rollout_errors=[RolloutError("BadRequestError", "a")],
        )
        counter.ingest(tg1)
        counter.ingest(None)
        metrics = counter.get_metrics()
        assert metrics["rollout_errors/BadRequestError"] == 1.0
        assert metrics["rollout_errors/groups_skipped"] == 1.0


# ---------------------------------------------------------------------------
# FailFast strategy tests (via do_group_rollout)
# ---------------------------------------------------------------------------


class TestFailFastStrategy:
    def test_default_strategy_raises_on_error(self):
        """Without strategy (FailFast default), errors propagate."""
        builder = _FakeEnvGroupBuilder(n_envs=2)
        policy = _FakePolicy(fail_indices={1})
        with pytest.raises(RuntimeError, match="fake error"):
            asyncio.run(do_group_rollout(builder, policy))

    def test_success_returns_all_trajectories(self):
        builder = _FakeEnvGroupBuilder(n_envs=3)
        policy = _FakePolicy()
        tg = asyncio.run(do_group_rollout(builder, policy))
        assert len(tg.trajectories_G) == 3
        assert tg.rollout_errors == []

    def test_cancelled_error_propagates(self):
        builder = _FakeEnvGroupBuilder(n_envs=2)
        policy = _FakePolicy(fail_indices={0}, error=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(do_group_rollout(builder, policy))


# ---------------------------------------------------------------------------
# RetryOnFailure strategy tests (via do_group_rollout)
# ---------------------------------------------------------------------------


class TestRetryOnFailureStrategy:
    def test_no_errors_returns_all_trajectories(self):
        builder = _FakeEnvGroupBuilder(n_envs=2)
        policy = _FakePolicy()
        tg = asyncio.run(do_group_rollout(builder, policy, strategy=RetryOnFailure(max_retries=3)))
        assert len(tg.trajectories_G) == 2
        assert tg.rollout_errors == []

    def test_retry_recovers_from_transient_failure(self):
        """One trajectory fails initially, retry succeeds."""
        builder = _FakeEnvGroupBuilder(n_envs=2)
        # Call index 1 fails, but retry (index 2) succeeds
        policy = _FakePolicy(fail_indices={1})
        tg = asyncio.run(do_group_rollout(builder, policy, strategy=RetryOnFailure(max_retries=3)))
        # Original success + retry success = 2 trajectories
        assert len(tg.trajectories_G) == 2
        assert len(tg.rollout_errors) == 1
        assert tg.rollout_errors[0].error_type == "RuntimeError"

    def test_retry_creates_fresh_envs(self):
        """Retry calls make_envs again to get a fresh environment."""
        builder = _FakeEnvGroupBuilder(n_envs=2)
        policy = _FakePolicy(fail_indices={1})  # one failure triggers one retry
        asyncio.run(do_group_rollout(builder, policy, strategy=RetryOnFailure(max_retries=3)))
        # Initial make_envs + 1 retry make_envs
        assert builder.make_envs_call_count == 2

    def test_all_fail_raises_after_retries(self):
        """All trajectories fail and retries exhausted -> re-raises last error."""
        builder = _FakeEnvGroupBuilder(n_envs=2)
        # All calls fail (indices 0,1 initial + 2,3,4 retries = 5 total calls)
        policy = _FakePolicy(fail_indices={0, 1, 2, 3, 4})
        with pytest.raises(RuntimeError, match="fake error"):
            asyncio.run(do_group_rollout(builder, policy, strategy=RetryOnFailure(max_retries=3)))

    def test_budget_exhausted_cancels_and_raises(self):
        """When retry budget runs out, cancel remaining tasks and re-raise."""
        builder = _FakeEnvGroupBuilder(n_envs=4)
        # Indices 0,1 succeed; 2 fails, retry at 4 fails; 3 fails, retry at 5 fails
        # Budget is 2 — first retry succeeds at index 4? No: indices 2,3 fail, retries at 4,5 also fail
        # After 2 retries exhausted, next failure re-raises
        policy = _FakePolicy(fail_indices={2, 3, 4, 5})
        with pytest.raises(RuntimeError, match="fake error"):
            asyncio.run(do_group_rollout(builder, policy, strategy=RetryOnFailure(max_retries=2)))

    def test_zero_retries_raises_on_any_failure(self):
        """max_retries=0 means no retries — any failure crashes the group."""
        builder = _FakeEnvGroupBuilder(n_envs=4)
        policy = _FakePolicy(fail_indices={2})
        with pytest.raises(RuntimeError, match="fake error"):
            asyncio.run(do_group_rollout(builder, policy, strategy=RetryOnFailure(max_retries=0)))

    def test_cancelled_error_not_swallowed(self):
        builder = _FakeEnvGroupBuilder(n_envs=2)
        policy = _FakePolicy(fail_indices={0}, error=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(do_group_rollout(builder, policy, strategy=RetryOnFailure(max_retries=3)))

    def test_keyboard_interrupt_not_swallowed(self):
        builder = _FakeEnvGroupBuilder(n_envs=2)
        policy = _FakePolicy(fail_indices={0}, error=KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(do_group_rollout(builder, policy, strategy=RetryOnFailure(max_retries=3)))

    def test_make_envs_failure_during_retry_propagates(self):
        """If make_envs() fails during retry, the error propagates."""

        call_count = 0

        class _FailOnSecondMakeEnvs(EnvGroupBuilder):
            async def make_envs(self):
                nonlocal call_count
                call_count += 1
                if call_count > 1:
                    raise RuntimeError("container pool exhausted")
                return [_FakeEnv() for _ in range(2)]

        builder = _FailOnSecondMakeEnvs()
        policy = _FakePolicy(fail_indices={1})  # triggers a retry
        with pytest.raises(RuntimeError, match="container pool exhausted"):
            asyncio.run(do_group_rollout(builder, policy, strategy=RetryOnFailure(max_retries=3)))


# ---------------------------------------------------------------------------
# _do_group_rollout_and_filter_constant_reward_impl tests
# ---------------------------------------------------------------------------


class TestImplErrorHandling:
    def test_fail_fast_propagates_error(self):
        builder = _FakeEnvGroupBuilder(n_envs=2)
        sampling_client = MagicMock(spec=tinker.SamplingClient)
        with (
            patch(
                "tinker_cookbook.rl.rollouts.do_group_rollout",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            asyncio.run(
                _do_group_rollout_and_filter_constant_reward_impl(
                    sampling_client,
                    builder,
                    max_tokens=100,
                    temperature=1.0,
                    do_remove_constant_reward_groups=False,
                    strategy=FailFast(),
                )
            )

    def test_retry_strategy_returns_none_on_group_error(self):
        builder = _FakeEnvGroupBuilder(n_envs=2)
        sampling_client = MagicMock(spec=tinker.SamplingClient)
        with patch(
            "tinker_cookbook.rl.rollouts.do_group_rollout",
            side_effect=RuntimeError("boom"),
        ):
            result = asyncio.run(
                _do_group_rollout_and_filter_constant_reward_impl(
                    sampling_client,
                    builder,
                    max_tokens=100,
                    temperature=1.0,
                    do_remove_constant_reward_groups=False,
                    strategy=RetryOnFailure(max_retries=3),
                )
            )
        assert result is None

    def test_all_trajectories_failed_returns_none(self):
        builder = _FakeEnvGroupBuilder(n_envs=2)
        sampling_client = MagicMock(spec=tinker.SamplingClient)
        with patch(
            "tinker_cookbook.rl.rollouts.do_group_rollout",
            side_effect=AllTrajectoriesFailedError("all failed"),
        ):
            result = asyncio.run(
                _do_group_rollout_and_filter_constant_reward_impl(
                    sampling_client,
                    builder,
                    max_tokens=100,
                    temperature=1.0,
                    do_remove_constant_reward_groups=False,
                    strategy=RetryOnFailure(max_retries=3),
                )
            )
        assert result is None
