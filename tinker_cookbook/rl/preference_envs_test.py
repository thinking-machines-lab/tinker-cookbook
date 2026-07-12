"""Tests for pairwise preference capture dimensions (metadata, matchup metrics)."""

import asyncio
from typing import Any, cast
from unittest.mock import MagicMock

import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.preference.types import Comparison, PreferenceModel
from tinker_cookbook.renderers.base import ParseTermination
from tinker_cookbook.rl.preference_envs import (
    PairwisePreferenceGroupBuilder,
    TournamentPattern,
)
from tinker_cookbook.rl.types import Trajectory, Transition


class AlwaysPrefersB(PreferenceModel):
    async def __call__(self, comparison: Comparison) -> float:
        return 1.0


def _renderer() -> MagicMock:
    renderer = MagicMock()
    renderer.parse_response = MagicMock(
        return_value=({"role": "assistant", "content": "hi"}, ParseTermination.STOP_SEQUENCE)
    )
    return renderer


def _trajectory() -> Trajectory:
    return Trajectory(
        transitions=[
            Transition(
                ob=tinker.ModelInput.from_ints([1]),
                ac=TokensWithLogprobs(tokens=[2], maybe_logprobs=[-0.1]),
                reward=0.0,
                episode_done=True,
            )
        ],
        final_ob=tinker.ModelInput.from_ints([]),
    )


def _builder() -> PairwisePreferenceGroupBuilder:
    return PairwisePreferenceGroupBuilder(
        convo_prefix=[{"role": "user", "content": "q"}],
        policy_renderer=cast(Any, _renderer()),
        tournament_pattern=TournamentPattern.ALL_PAIRS_ONE_WAY,
        preference_model=AlwaysPrefersB(),
        num_envs=2,
    )


class TestMetadata:
    def test_matchup_structure_dimensions(self):
        assert _builder().metadata() == {
            "tournament_pattern": "all_pairs_one_way",
            "matchup_group_size": 4,
        }


class TestMatchupCountMetric:
    def test_matchup_count_reported_and_rewards_unchanged(self):
        builder = _builder()
        results = asyncio.run(
            builder.compute_group_rewards([_trajectory(), _trajectory()], env_group=[])
        )
        assert len(results) == 2
        (reward_a, metrics_a), (reward_b, metrics_b) = results
        # One matchup between the two completions; B is always preferred.
        assert metrics_a["matchup_count"] == 1
        assert metrics_b["matchup_count"] == 1
        assert reward_a == -1.0
        assert reward_b == 1.0
        assert metrics_a["win_minus_loss"] == -1.0
        assert metrics_b["win_minus_loss"] == 1.0
