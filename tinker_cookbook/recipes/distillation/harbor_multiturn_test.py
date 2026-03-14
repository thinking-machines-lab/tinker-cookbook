"""Tests for harbor_multiturn zero-reward distillation."""

import asyncio
from unittest.mock import MagicMock

import pytest

from tinker_cookbook.renderers.base import Message


class TestZeroReward:
    def test_zero_reward_returns_zero(self):
        mod = pytest.importorskip(
            "tinker_cookbook.recipes.distillation.harbor_multiturn",
            reason="requires modal",
        )
        result = asyncio.run(mod.zero_reward([Message(role="user", content="test")]))
        assert result == (0.0, {})

    def test_zero_reward_ignores_history_content(self):
        mod = pytest.importorskip(
            "tinker_cookbook.recipes.distillation.harbor_multiturn",
            reason="requires modal",
        )
        for history in [[], [Message(role="user", content="x")] * 50]:
            result = asyncio.run(mod.zero_reward(history))
            assert result == (0.0, {})

    def test_env_group_builder_compute_group_rewards_returns_zeros(self):
        mod = pytest.importorskip(
            "tinker_cookbook.recipes.distillation.harbor_multiturn",
            reason="requires modal",
        )
        harbor_env = pytest.importorskip(
            "tinker_cookbook.recipes.harbor_rl.harbor_env",
            reason="requires modal",
        )
        builder = harbor_env.HarborEnvGroupBuilder(
            task=MagicMock(),
            model_name="test",
            renderer_name="test",
            max_turns=5,
            group_size=2,
            reward_fn=mod.zero_reward,
        )
        builder._sandboxes = []
        trajectories = [MagicMock(), MagicMock()]
        result = asyncio.run(builder.compute_group_rewards(trajectories, []))
        assert result == [(0.0, {}), (0.0, {})]
