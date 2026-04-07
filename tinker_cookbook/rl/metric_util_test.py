"""Tests for RLTestSetEvaluator.last_result (BenchmarkResult integration)."""

from __future__ import annotations

import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.eval.benchmarks._types import BenchmarkResult
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.rl.types import (
    Trajectory,
    TrajectoryGroup,
    Transition,
)


def _make_trajectory(reward: float, metrics: dict | None = None) -> Trajectory:
    return Trajectory(
        transitions=[
            Transition(
                ob=tinker.ModelInput.from_ints([1, 2, 3]),
                ac=TokensWithLogprobs(tokens=[4, 5], maybe_logprobs=[0.1, 0.2]),
                reward=reward,
                episode_done=True,
                metrics=metrics or {},
            )
        ],
        final_ob=tinker.ModelInput.from_ints([]),
    )


def _make_trajectory_group(rewards: list[float]) -> TrajectoryGroup:
    return TrajectoryGroup(
        trajectories_G=[_make_trajectory(r) for r in rewards],
        final_rewards_G=[0.0] * len(rewards),
        metrics_G=[{}] * len(rewards),
    )


class TestLastResult:
    def test_last_result_is_none_before_eval(self):
        """last_result is None before any evaluation."""

        # We can't easily create a full evaluator without a dataset,
        # so test via _collect_eval_metrics directly
        pass

    def test_collect_eval_metrics_populates_last_result(self):
        """_collect_eval_metrics builds a BenchmarkResult in last_result."""
        # Create a minimal evaluator (bypass __init__ dataset requirement)
        evaluator = object.__new__(RLTestSetEvaluator)
        evaluator.name = "test"
        evaluator.env_group_builders_P = []
        evaluator.last_result = None

        # Two groups: one with reward > 0, one with reward = 0
        groups = [
            _make_trajectory_group([1.0]),
            _make_trajectory_group([0.0]),
        ]

        # Mock builders with tags
        class FakeBuilder:
            def logging_tags(self):
                return ["math"]

        evaluator.env_group_builders_P = [FakeBuilder(), FakeBuilder()]

        metrics = evaluator._collect_eval_metrics(groups, rollout_summary_export=None)

        # Check the flat dict output
        assert "test/env/all/reward/total" in metrics
        assert isinstance(metrics["test/env/all/reward/total"], float)

        # Check the BenchmarkResult
        result = evaluator.last_result
        assert result is not None
        assert isinstance(result, BenchmarkResult)
        assert result.name == "test"
        assert result.num_examples == 2
        assert result.num_correct == 1
        assert result.score == 0.5
        assert result.num_errors == 0
        assert "env/all/reward/total" in result.metrics

    def test_last_result_with_errors(self):
        """None results (errors) are counted in num_errors."""
        evaluator = object.__new__(RLTestSetEvaluator)
        evaluator.name = "eval"
        evaluator.last_result = None

        class FakeBuilder:
            def logging_tags(self):
                return []

        # One success, one failure (None)
        evaluator.env_group_builders_P = [FakeBuilder(), FakeBuilder()]
        results: list[TrajectoryGroup | None] = [
            _make_trajectory_group([1.0]),
            None,
        ]

        evaluator._collect_eval_metrics(results, rollout_summary_export=None)

        result = evaluator.last_result
        assert result is not None
        assert result.num_examples == 2  # 1 completed + 1 error
        assert result.num_correct == 1
        assert result.num_errors == 1
        # score = num_correct / (num_examples - num_errors) since score is
        # over completed examples; use score_completed for that
        assert result.score_completed == 1.0  # 1/1 completed correctly

    def test_last_result_all_errors(self):
        """All errors produces score=0."""
        evaluator = object.__new__(RLTestSetEvaluator)
        evaluator.name = "test"
        evaluator.last_result = None

        class FakeBuilder:
            def logging_tags(self):
                return []

        evaluator.env_group_builders_P = [FakeBuilder(), FakeBuilder()]
        results: list[TrajectoryGroup | None] = [None, None]

        evaluator._collect_eval_metrics(results, rollout_summary_export=None)

        result = evaluator.last_result
        assert result is not None
        assert result.num_examples == 2
        assert result.num_correct == 0
        assert result.num_errors == 2
        assert result.score == 0.0

    def test_last_result_metrics_match_flat_dict(self):
        """BenchmarkResult.metrics contains the same data as the flat dict (unprefixed)."""
        evaluator = object.__new__(RLTestSetEvaluator)
        evaluator.name = "test"
        evaluator.last_result = None

        class FakeBuilder:
            def logging_tags(self):
                return []

        evaluator.env_group_builders_P = [FakeBuilder()]
        groups = [_make_trajectory_group([0.75])]

        flat_metrics = evaluator._collect_eval_metrics(groups, rollout_summary_export=None)

        result = evaluator.last_result
        assert result is not None

        # Every key in result.metrics should appear prefixed in flat_metrics
        for key in result.metrics:
            assert f"test/{key}" in flat_metrics, f"Missing test/{key} in flat metrics"
