"""Tests for RL data processing helpers."""

from __future__ import annotations

import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
    successful_rollout_indices,
)
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


class TestSuccessfulRolloutIndices:
    def test_skips_failed_rollouts(self):
        groups_raw = [
            None,
            _make_trajectory_group([1.0, 0.0]),
            None,
            _make_trajectory_group([0.5, 0.25]),
        ]
        assert successful_rollout_indices(groups_raw) == [1, 3]

    def test_all_succeeded(self):
        groups_raw = [_make_trajectory_group([1.0, 0.0]) for _ in range(3)]
        assert successful_rollout_indices(groups_raw) == [0, 1, 2]

    def test_all_failed(self):
        assert successful_rollout_indices([None, None]) == []

    def test_no_rollouts(self):
        assert successful_rollout_indices([]) == []

    def test_group_index_stamped_on_datums_matches_the_filtered_list(self):
        """``assemble_training_data`` numbers groups by position in the list it is given.

        Anything the caller looks up by that index -- a teacher client, a
        dataset index, a prompt -- must come from a list filtered by these same
        indices, or it belongs to a different problem.
        """
        groups_raw = [None, _make_trajectory_group([1.0, 0.0]), _make_trajectory_group([0.5, 0.25])]
        dataset_indices_P = [7, 8, 9]

        keep_P = successful_rollout_indices(groups_raw)
        groups_P = [groups_raw[i] for i in keep_P]
        kept_dataset_indices_P = [dataset_indices_P[i] for i in keep_P]

        _data_D, metadata_D = assemble_training_data(groups_P, compute_advantages(groups_P))
        stamped = sorted({m["group_idx"] for m in metadata_D})
        assert stamped == [0, 1]
        assert [kept_dataset_indices_P[i] for i in stamped] == [8, 9]

        # The hazard: the same indices into the unfiltered list name other problems.
        assert [dataset_indices_P[i] for i in stamped] == [7, 8]
