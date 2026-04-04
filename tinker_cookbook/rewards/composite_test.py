"""Tests for composite reward utilities."""

import pytest

from tinker_cookbook.rewards.composite import (
    WeightedReward,
    combine_max,
    combine_min,
    combine_product,
    combine_threshold,
    combine_weighted,
)


def _always_one(_text: str) -> float:
    return 1.0


def _always_zero(_text: str) -> float:
    return 0.0


def _length_reward(text: str) -> float:
    return min(len(text) / 100.0, 1.0)


class TestCombineWeighted:
    def test_single(self):
        rewards = [WeightedReward(name="a", fn=_always_one, weight=2.0)]
        total, metrics = combine_weighted("hello", rewards)
        assert total == pytest.approx(2.0)
        assert metrics == {"a": 1.0}

    def test_multiple(self):
        rewards = [
            WeightedReward(name="one", fn=_always_one, weight=0.5),
            WeightedReward(name="zero", fn=_always_zero, weight=0.5),
        ]
        total, metrics = combine_weighted("hello", rewards)
        assert total == pytest.approx(0.5)
        assert metrics == {"one": 1.0, "zero": 0.0}

    def test_empty(self):
        total, metrics = combine_weighted("hello", [])
        assert total == 0.0
        assert metrics == {}


class TestCombineMin:
    def test_all_ones(self):
        assert combine_min("x", [_always_one, _always_one]) == 1.0

    def test_with_zero(self):
        assert combine_min("x", [_always_one, _always_zero]) == 0.0


class TestCombineMax:
    def test_with_zero(self):
        assert combine_max("x", [_always_one, _always_zero]) == 1.0

    def test_all_zeros(self):
        assert combine_max("x", [_always_zero, _always_zero]) == 0.0


class TestCombineProduct:
    def test_all_ones(self):
        assert combine_product("x", [_always_one, _always_one]) == 1.0

    def test_with_zero(self):
        assert combine_product("x", [_always_one, _always_zero]) == 0.0


class TestCombineThreshold:
    def test_above(self):
        fn = combine_threshold(_always_one, cutoff=0.5)
        assert fn("x") == 1.0

    def test_below(self):
        fn = combine_threshold(_always_zero, cutoff=0.5)
        assert fn("x") == 0.0

    def test_at_cutoff(self):
        fn = combine_threshold(lambda _: 0.5, cutoff=0.5)
        assert fn("x") == 1.0
