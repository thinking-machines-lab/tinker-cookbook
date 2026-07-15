"""Tests for RolloutLimits / ParseErrorPolicy / TerminationRewardPolicy construction."""

import pickle
from typing import Any

import pytest

from tinker_cookbook.rl.rollout_limits import (
    DEFAULT_LIMIT_STOP_REASONS,
    DEFAULT_RETRY_MESSAGE_TEMPLATE,
    ParseErrorPolicy,
    RolloutLimits,
    TerminationRewardPolicy,
)

_INT_FIELDS = [
    "max_turns",
    "max_trajectory_tokens",
    "max_sampled_tokens",
    "max_tool_calls",
    "max_turn_tokens",
]
_FLOAT_FIELDS = [
    "rollout_timeout_seconds",
    "sampling_turn_timeout_seconds",
]
_ALL_FIELDS = _INT_FIELDS + _FLOAT_FIELDS


class TestConstruction:
    def test_default_is_unlimited(self):
        limits = RolloutLimits()
        for name in _ALL_FIELDS:
            assert getattr(limits, name) is None, name

    def test_all_fields_settable(self):
        limits = RolloutLimits(
            max_turns=10,
            max_trajectory_tokens=65536,
            max_sampled_tokens=32768,
            max_tool_calls=30,
            max_turn_tokens=16384,
            rollout_timeout_seconds=600.0,
            sampling_turn_timeout_seconds=120.0,
        )
        assert limits.max_turns == 10
        assert limits.max_trajectory_tokens == 65536
        assert limits.max_sampled_tokens == 32768
        assert limits.max_tool_calls == 30
        assert limits.max_turn_tokens == 16384
        assert limits.rollout_timeout_seconds == 600.0
        assert limits.sampling_turn_timeout_seconds == 120.0

    def test_pickle_round_trip(self):
        limits = RolloutLimits(max_turns=3, rollout_timeout_seconds=5.0)
        restored = pickle.loads(pickle.dumps(limits))
        assert restored == limits


class TestValidation:
    @pytest.mark.parametrize("name", _INT_FIELDS)
    @pytest.mark.parametrize("bad_value", [0, -1])
    def test_non_positive_int_rejected(self, name: str, bad_value: int):
        kwargs: dict[str, Any] = {name: bad_value}
        with pytest.raises(ValueError, match=name):
            RolloutLimits(**kwargs)

    @pytest.mark.parametrize("name", _FLOAT_FIELDS)
    @pytest.mark.parametrize("bad_value", [0.0, -0.5])
    def test_non_positive_float_rejected(self, name: str, bad_value: float):
        kwargs: dict[str, Any] = {name: bad_value}
        with pytest.raises(ValueError, match=name):
            RolloutLimits(**kwargs)

    @pytest.mark.parametrize("name", _INT_FIELDS)
    def test_positive_int_accepted(self, name: str):
        kwargs: dict[str, Any] = {name: 1}
        assert getattr(RolloutLimits(**kwargs), name) == 1

    @pytest.mark.parametrize("name", _FLOAT_FIELDS)
    def test_positive_float_accepted(self, name: str):
        kwargs: dict[str, Any] = {name: 0.1}
        assert getattr(RolloutLimits(**kwargs), name) == 0.1


class TestParseErrorPolicy:
    def test_defaults(self):
        policy = ParseErrorPolicy()
        assert policy.max_consecutive == 0
        assert policy.retry_message_template == DEFAULT_RETRY_MESSAGE_TEMPLATE
        assert "{details}" in policy.retry_message_template
        assert policy.penalty_per_error == 0.0
        assert policy.terminal_reward == 0.0
        assert policy.mask_error_turns is False

    def test_all_fields_settable(self):
        policy = ParseErrorPolicy(
            max_consecutive=2,
            retry_message_template="Fix it.\n{details}",
            penalty_per_error=0.05,
            terminal_reward=-1.0,
            mask_error_turns=True,
        )
        assert policy.max_consecutive == 2
        assert policy.retry_message_template == "Fix it.\n{details}"
        assert policy.penalty_per_error == 0.05
        assert policy.terminal_reward == -1.0
        assert policy.mask_error_turns is True

    def test_pickle_round_trip(self):
        policy = ParseErrorPolicy(max_consecutive=2, penalty_per_error=0.1)
        restored = pickle.loads(pickle.dumps(policy))
        assert restored == policy

    def test_negative_max_consecutive_rejected(self):
        with pytest.raises(ValueError, match="max_consecutive"):
            ParseErrorPolicy(max_consecutive=-1)

    def test_negative_penalty_rejected(self):
        with pytest.raises(ValueError, match="penalty_per_error"):
            ParseErrorPolicy(penalty_per_error=-0.1)

    def test_template_without_placeholder_rejected(self):
        with pytest.raises(ValueError, match="details"):
            ParseErrorPolicy(retry_message_template="Please try again.")

    def test_negative_terminal_reward_accepted(self):
        assert ParseErrorPolicy(terminal_reward=-0.5).terminal_reward == -0.5


class TestTerminationRewardPolicy:
    def test_defaults_are_inert(self):
        policy = TerminationRewardPolicy()
        assert policy.zero_reward_on_limit is False
        assert policy.skip_grading_on_timeout is False
        assert policy.grader_timeout_seconds is None
        assert policy.pass_all_messages_to_grader is False

    def test_default_limit_stop_reasons_are_the_five_limit_stops(self):
        assert TerminationRewardPolicy().limit_stop_reasons == DEFAULT_LIMIT_STOP_REASONS
        assert tuple(str(r) for r in DEFAULT_LIMIT_STOP_REASONS) == (
            "max_tokens",
            "max_sampled_tokens",
            "max_turns",
            "max_tool_calls",
            "rollout_timeout",
        )

    def test_all_fields_settable(self):
        policy = TerminationRewardPolicy(
            zero_reward_on_limit=True,
            limit_stop_reasons=("max_turns",),
            skip_grading_on_timeout=True,
            grader_timeout_seconds=900.0,
            pass_all_messages_to_grader=True,
        )
        assert policy.zero_reward_on_limit is True
        assert policy.limit_stop_reasons == ("max_turns",)
        assert policy.skip_grading_on_timeout is True
        assert policy.grader_timeout_seconds == 900.0
        assert policy.pass_all_messages_to_grader is True

    @pytest.mark.parametrize("bad_value", [0.0, -1.0])
    def test_non_positive_grader_timeout_rejected(self, bad_value: float):
        with pytest.raises(ValueError, match="grader_timeout_seconds"):
            TerminationRewardPolicy(grader_timeout_seconds=bad_value)

    def test_pickle_round_trip(self):
        policy = TerminationRewardPolicy(
            zero_reward_on_limit=True, skip_grading_on_timeout=True, grader_timeout_seconds=900
        )
        restored = pickle.loads(pickle.dumps(policy))
        assert restored == policy
