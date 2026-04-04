"""Tests for composite reward utilities."""

import pytest

from tinker_cookbook.rewards.composite import (
    WeightedReward,
    reward_max,
    reward_min,
    reward_product,
    threshold,
    weighted_sum,
)


def _always_one(_text: str) -> float:
    return 1.0


def _always_zero(_text: str) -> float:
    return 0.0


def _length_reward(text: str) -> float:
    return min(len(text) / 100.0, 1.0)


class TestWeightedSum:
    def test_single(self):
        rewards = [WeightedReward(name="a", fn=_always_one, weight=2.0)]
        total, metrics = weighted_sum("hello", rewards)
        assert total == pytest.approx(2.0)
        assert metrics == {"a": 1.0}

    def test_multiple(self):
        rewards = [
            WeightedReward(name="one", fn=_always_one, weight=0.5),
            WeightedReward(name="zero", fn=_always_zero, weight=0.5),
        ]
        total, metrics = weighted_sum("hello", rewards)
        assert total == pytest.approx(0.5)
        assert metrics == {"one": 1.0, "zero": 0.0}

    def test_empty(self):
        total, metrics = weighted_sum("hello", [])
        assert total == 0.0
        assert metrics == {}


class TestRewardMin:
    def test_all_ones(self):
        assert reward_min("x", [_always_one, _always_one]) == 1.0

    def test_with_zero(self):
        assert reward_min("x", [_always_one, _always_zero]) == 0.0


class TestRewardMax:
    def test_with_zero(self):
        assert reward_max("x", [_always_one, _always_zero]) == 1.0

    def test_all_zeros(self):
        assert reward_max("x", [_always_zero, _always_zero]) == 0.0


class TestRewardProduct:
    def test_all_ones(self):
        assert reward_product("x", [_always_one, _always_one]) == 1.0

    def test_with_zero(self):
        assert reward_product("x", [_always_one, _always_zero]) == 0.0


class TestThreshold:
    def test_above(self):
        fn = threshold(_always_one, cutoff=0.5)
        assert fn("x") == 1.0

    def test_below(self):
        fn = threshold(_always_zero, cutoff=0.5)
        assert fn("x") == 0.0

    def test_at_cutoff(self):
        fn = threshold(lambda _: 0.5, cutoff=0.5)
        assert fn("x") == 1.0
