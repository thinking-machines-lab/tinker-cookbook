"""Tests for RL data processing helpers."""

from __future__ import annotations

import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import (
    nonconstant_reward_group_indices,
    remove_constant_reward_groups,
)
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition


def _make_trajectory(reward: float, num_action_tokens: int = 2) -> Trajectory:
    return Trajectory(
        transitions=[
            Transition(
                ob=tinker.ModelInput.from_ints([1, 2, 3]),
                ac=TokensWithLogprobs(
                    tokens=[4] * num_action_tokens,
                    maybe_logprobs=[0.0] * num_action_tokens,
                ),
                reward=reward,
                episode_done=True,
            )
        ],
        final_ob=tinker.ModelInput.from_ints([]),
    )


def _make_trajectory_group(rewards: list[float], num_action_tokens: int = 2) -> TrajectoryGroup:
    return TrajectoryGroup(
        trajectories_G=[_make_trajectory(r, num_action_tokens) for r in rewards],
        final_rewards_G=[0.0] * len(rewards),
        metrics_G=[{} for _ in rewards],
    )


class TestNonconstantRewardGroupIndices:
    def test_keeps_mixed_drops_uniform(self):
        groups_P = [
            _make_trajectory_group([1.0, 1.0]),  # uniform
            _make_trajectory_group([1.0, 0.0]),  # mixed
            _make_trajectory_group([0.0, 0.0]),  # uniform
            _make_trajectory_group([0.5, 0.25]),  # mixed
        ]
        assert nonconstant_reward_group_indices(groups_P) == [1, 3]

    def test_all_uniform_keeps_first_group(self):
        groups_P = [
            _make_trajectory_group([1.0, 1.0]),
            _make_trajectory_group([0.0, 0.0]),
        ]
        assert nonconstant_reward_group_indices(groups_P) == [0]

    def test_no_groups(self):
        assert nonconstant_reward_group_indices([]) == []

    def test_agrees_with_remove_constant_reward_groups(self):
        groups_P = [
            _make_trajectory_group([1.0, 1.0]),
            _make_trajectory_group([1.0, 0.0]),
            _make_trajectory_group([0.5, 0.25]),
        ]
        by_index = [groups_P[i] for i in nonconstant_reward_group_indices(groups_P)]
        assert by_index == remove_constant_reward_groups(groups_P)

    def test_agrees_with_remove_constant_reward_groups_when_all_uniform(self):
        groups_P = [
            _make_trajectory_group([1.0, 1.0]),
            _make_trajectory_group([0.0, 0.0]),
        ]
        by_index = [groups_P[i] for i in nonconstant_reward_group_indices(groups_P)]
        assert by_index == remove_constant_reward_groups(groups_P)


class TestFilteringKeepsParallelListsAligned:
    """Lists running parallel to the trajectory groups must be filtered in step.

    ``do_sync_training`` drops constant-reward groups and then hands both the
    groups and their ``EnvGroupBuilder`` list to ``prepare_minibatch``, which
    zips the two to tag metrics. Filtering only the groups shifts every tag
    onto a later problem's rollouts, so per-tag reward curves report another
    dataset's numbers.
    """

    def test_tags_follow_their_own_groups(self):
        # Distinct action-token counts make each group identifiable in the metrics.
        groups_P = [
            _make_trajectory_group([1.0, 1.0], num_action_tokens=100),  # uniform
            _make_trajectory_group([1.0, 0.6], num_action_tokens=200),
            _make_trajectory_group([0.2, 0.0], num_action_tokens=300),
        ]
        taglist_P = [["math"], ["code"], ["chat"]]

        keep_P = nonconstant_reward_group_indices(groups_P)
        assert keep_P == [1, 2]

        metrics = compute_trajectory_metrics(
            [groups_P[i] for i in keep_P],
            [taglist_P[i] for i in keep_P],
        )

        # "code" is the 200-token group with mean reward 0.8; "chat" is the
        # 300-token group with mean reward 0.1. "math" was dropped entirely.
        assert metrics["env/code/ac_tokens_per_turn"] == 200.0
        assert metrics["env/chat/ac_tokens_per_turn"] == 300.0
        assert metrics["env/code/reward/total"] == 0.8
        assert metrics["env/chat/reward/total"] == 0.1
        assert not any(key.startswith("env/math/") for key in metrics)

    def test_filtering_groups_alone_misattributes_tags(self):
        """Pin the failure mode, so a regression is unmistakable."""
        groups_P = [
            _make_trajectory_group([1.0, 1.0], num_action_tokens=100),  # uniform
            _make_trajectory_group([1.0, 0.6], num_action_tokens=200),
            _make_trajectory_group([0.2, 0.0], num_action_tokens=300),
        ]
        taglist_P = [["math"], ["code"], ["chat"]]

        # What the old call site did: groups filtered, parallel list left whole.
        metrics = compute_trajectory_metrics(remove_constant_reward_groups(groups_P), taglist_P)

        # Every tag slides onto the next group: "math" reports "code"'s rollouts,
        # "code" reports "chat"'s, and "chat" disappears.
        assert metrics["env/math/ac_tokens_per_turn"] == 200.0
        assert metrics["env/code/ac_tokens_per_turn"] == 300.0
        assert "env/chat/ac_tokens_per_turn" not in metrics
