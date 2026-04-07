"""Tests for RLTestSetEvaluator.last_result (BenchmarkResult integration)."""

from __future__ import annotations

import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.eval.benchmarks._types import BenchmarkResult
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)


def _make_trajectory(reward: float) -> Trajectory:
    return Trajectory(
        transitions=[
            Transition(
                ob=tinker.ModelInput.from_ints([1, 2, 3]),
                ac=TokensWithLogprobs(tokens=[4, 5], maybe_logprobs=[0.1, 0.2]),
                reward=reward,
                episode_done=True,
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


class _FakeBuilder(EnvGroupBuilder):
    """Minimal builder stub for tests."""

    def __init__(self, tags: list[str] | None = None):
        self._tags = tags or []

    async def make_envs(self):  # type: ignore[override]
        return []

    def logging_tags(self) -> list[str]:
        return self._tags


def _make_evaluator(name: str, builders: list[EnvGroupBuilder]) -> RLTestSetEvaluator:
    """Create a minimal RLTestSetEvaluator without requiring a full dataset."""
    evaluator = object.__new__(RLTestSetEvaluator)
    evaluator.name = name
    evaluator.env_group_builders_P = builders
    evaluator.last_result = None
    return evaluator


class TestLastResult:
    def test_collect_eval_metrics_populates_last_result(self):
        """_collect_eval_metrics builds a BenchmarkResult in last_result."""
        builders: list[EnvGroupBuilder] = [_FakeBuilder(["math"]), _FakeBuilder(["math"])]
        evaluator = _make_evaluator("test", builders)

        results: list[TrajectoryGroup | None] = [
            _make_trajectory_group([1.0]),
            _make_trajectory_group([0.0]),
        ]
        metrics = evaluator._collect_eval_metrics(results, rollout_summary_export=None)

        # Check the flat dict output
        assert "test/env/all/reward/total" in metrics

        # Check the BenchmarkResult
        result = evaluator.last_result
        assert isinstance(result, BenchmarkResult)
        assert result.name == "test"
        assert result.num_examples == 2
        assert result.num_correct == 1
        assert result.score == 0.5
        assert result.num_errors == 0
        assert "env/all/reward/total" in result.metrics

    def test_last_result_with_errors(self):
        """None results (errors) are counted in num_errors."""
        builders: list[EnvGroupBuilder] = [_FakeBuilder(), _FakeBuilder()]
        evaluator = _make_evaluator("eval", builders)

        results: list[TrajectoryGroup | None] = [
            _make_trajectory_group([1.0]),
            None,
        ]
        evaluator._collect_eval_metrics(results, rollout_summary_export=None)

        result = evaluator.last_result
        assert isinstance(result, BenchmarkResult)
        assert result.num_examples == 2
        assert result.num_correct == 1
        assert result.num_errors == 1
        assert result.score_completed == 1.0

    def test_last_result_all_errors(self):
        """All errors produces score=0."""
        builders: list[EnvGroupBuilder] = [_FakeBuilder(), _FakeBuilder()]
        evaluator = _make_evaluator("test", builders)

        results: list[TrajectoryGroup | None] = [None, None]
        evaluator._collect_eval_metrics(results, rollout_summary_export=None)

        result = evaluator.last_result
        assert isinstance(result, BenchmarkResult)
        assert result.num_examples == 2
        assert result.num_correct == 0
        assert result.num_errors == 2
        assert result.score == 0.0

    def test_last_result_metrics_match_flat_dict(self):
        """BenchmarkResult.metrics contains the same data as the flat dict (unprefixed)."""
        builders: list[EnvGroupBuilder] = [_FakeBuilder()]
        evaluator = _make_evaluator("test", builders)

        results: list[TrajectoryGroup | None] = [_make_trajectory_group([0.75])]
        flat_metrics = evaluator._collect_eval_metrics(results, rollout_summary_export=None)

        result = evaluator.last_result
        assert isinstance(result, BenchmarkResult)
        for key in result.metrics:
            assert f"test/{key}" in flat_metrics, f"Missing test/{key} in flat metrics"
