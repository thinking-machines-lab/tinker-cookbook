"""Tests for SDFT batch bookkeeping."""

from __future__ import annotations

import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.distillation.sdft import drop_failed_rollouts
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)


class _FakeBuilder(EnvGroupBuilder):
    def __init__(self, tag: str):
        self.tag = tag

    async def make_envs(self):  # type: ignore[override]
        return []

    def logging_tags(self) -> list[str]:
        return [self.tag]


def _make_trajectory_group(marker_token: int) -> TrajectoryGroup:
    """A one-trajectory group whose action tokens identify which problem it is."""
    return TrajectoryGroup(
        trajectories_G=[
            Trajectory(
                transitions=[
                    Transition(
                        ob=tinker.ModelInput.from_ints([1, 2, 3]),
                        ac=TokensWithLogprobs(
                            tokens=[marker_token, marker_token],
                            maybe_logprobs=[0.0, 0.0],
                        ),
                        reward=0.0,
                        episode_done=True,
                    )
                ],
                final_ob=tinker.ModelInput.from_ints([]),
            )
        ],
        final_rewards_G=[0.0],
        metrics_G=[{}],
    )


class TestDropFailedRollouts:
    def test_keeps_lists_aligned(self):
        builders_P = [_FakeBuilder("a"), _FakeBuilder("b"), _FakeBuilder("c")]
        questions_P = ["qa", "qb", "qc"]
        golden_answers_P = ["ga", "gb", "gc"]
        # The first problem's rollout failed.
        raw = [None, _make_trajectory_group(20), _make_trajectory_group(30)]

        groups, builders, questions, goldens = drop_failed_rollouts(
            raw, builders_P, questions_P, golden_answers_P
        )

        assert len(groups) == 2
        assert [b.tag for b in builders] == ["b", "c"]
        assert questions == ["qb", "qc"]
        assert goldens == ["gb", "gc"]

    def test_no_failures_is_identity(self):
        builders_P = [_FakeBuilder("a"), _FakeBuilder("b")]
        raw = [_make_trajectory_group(10), _make_trajectory_group(20)]

        groups, builders, questions, goldens = drop_failed_rollouts(
            raw, builders_P, ["qa", "qb"], ["ga", "gb"]
        )

        assert groups == raw
        assert [b.tag for b in builders] == ["a", "b"]
        assert questions == ["qa", "qb"]
        assert goldens == ["ga", "gb"]

    def test_all_failed(self):
        groups, builders, questions, goldens = drop_failed_rollouts(
            [None, None], [_FakeBuilder("a"), _FakeBuilder("b")], ["qa", "qb"], ["ga", "gb"]
        )
        assert groups == []
        assert builders == []
        assert questions == []
        assert goldens == []

    def test_group_idx_selects_the_datums_own_question(self):
        """The teacher prompt is looked up by ``metadata_D["group_idx"]``.

        That index points into the *filtered* group list, so the questions must
        be filtered alongside it -- otherwise a student completion is teacher-
        forced against another problem's prompt.
        """
        builders_P = [_FakeBuilder("a"), _FakeBuilder("b"), _FakeBuilder("c")]
        questions_P = ["qa", "qb", "qc"]
        # Marker tokens tie each trajectory back to the problem it came from.
        marker_by_question = {"qa": 10, "qb": 20, "qc": 30}
        raw = [None, _make_trajectory_group(20), _make_trajectory_group(30)]

        groups, _builders, questions, _goldens = drop_failed_rollouts(
            raw, builders_P, questions_P, ["ga", "gb", "gc"]
        )

        data_D, metadata_D = assemble_training_data(groups, compute_advantages(groups))
        assert data_D

        for datum, metadata in zip(data_D, metadata_D, strict=True):
            question = questions[metadata["group_idx"]]
            expected_marker = marker_by_question[question]
            # Every action token in this datum came from that question's rollout.
            assert expected_marker in datum.model_input.to_ints()

        # The hazard this guards against: the same index into the *unfiltered*
        # questions picks a different problem for every surviving datum.
        assert all(
            questions_P[metadata["group_idx"]] != questions[metadata["group_idx"]]
            for metadata in metadata_D
        )
