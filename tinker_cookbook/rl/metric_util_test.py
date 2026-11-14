"""Tests for RLTestSetEvaluator: last_result (BenchmarkResult integration) and
isolation between overlapping evaluations."""

from __future__ import annotations

import asyncio

import pytest
import tinker

from tinker_cookbook.completers import StopCondition, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.eval.benchmarks._types import BenchmarkResult
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    RLDataset,
    StepResult,
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
        # score treats errors as incorrect (consistent with benchmark runner)
        assert result.score == 0.5
        # score_completed excludes errors from denominator
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

    def test_last_result_counts_groups_not_trajectories(self):
        """num_examples counts groups (test problems), not individual trajectories."""
        builders: list[EnvGroupBuilder] = [_FakeBuilder(), _FakeBuilder()]
        evaluator = _make_evaluator("test", builders)

        # Group with 2 trajectories: one correct, one not.
        # The group counts as "correct" because at least one trajectory succeeded.
        results: list[TrajectoryGroup | None] = [
            _make_trajectory_group([1.0, 0.0]),  # group_size=2, mixed
            _make_trajectory_group([0.0, 0.0]),  # group_size=2, all wrong
        ]
        evaluator._collect_eval_metrics(results, rollout_summary_export=None)

        result = evaluator.last_result
        assert isinstance(result, BenchmarkResult)
        assert result.num_examples == 2  # 2 groups, not 4 trajectories
        assert result.num_correct == 1  # first group has reward > 0


class _Sandbox:
    def __init__(self) -> None:
        self.alive = True


class _SandboxEnv(Env):
    """Records whether its sandbox was still alive when it stepped."""

    def __init__(self, sandbox: _Sandbox, alive_at_step: list[bool]) -> None:
        self.sandbox = sandbox
        self.alive_at_step = alive_at_step

    async def initial_observation(self) -> tuple[tinker.ModelInput, list[int]]:
        return tinker.ModelInput.from_ints([1]), []

    async def step(self, action: list[int], *, extra: object = None) -> StepResult:
        self.alive_at_step.append(self.sandbox.alive)
        return StepResult(
            reward=1.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
        )


class _SandboxBuilder(EnvGroupBuilder):
    """Keeps the sandboxes its make_envs creates on the builder for cleanup,
    like HarborEnvGroupBuilder."""

    def __init__(self, alive_at_step: list[bool]) -> None:
        self.alive_at_step = alive_at_step
        self.created: list[_Sandbox] = []
        self._sandboxes: list[_Sandbox] = []

    async def make_envs(self) -> list[Env]:
        self._sandboxes = [_Sandbox()]
        self.created.extend(self._sandboxes)
        return [_SandboxEnv(sandbox, self.alive_at_step) for sandbox in self._sandboxes]

    async def cleanup(self) -> None:
        for sandbox in self._sandboxes:
            sandbox.alive = False
        self._sandboxes.clear()


class _CachedDataset(RLDataset):
    """Returns the same builder objects on every call, like HarborDataset."""

    def __init__(self, builders: list[EnvGroupBuilder]) -> None:
        self.builders = builders

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        return self.builders

    def __len__(self) -> int:
        return 1


class _GatedPolicy(TokenCompleter):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition, *, max_tokens: int | None = None
    ) -> TokensWithLogprobs:
        self.started.set()
        await self.release.wait()
        return TokensWithLogprobs(tokens=[2], maybe_logprobs=[0.0])


@pytest.mark.parametrize("first_eval_ends_by", ["finishing", "cancellation"])
def test_overlapping_evals_keep_their_own_sandboxes(first_eval_ends_by: str):
    """Async training can run evals of different steps at once. When the first
    one ends, its cleanup must not destroy the sandboxes the second is using."""
    alive_at_step: list[bool] = []
    builder = _SandboxBuilder(alive_at_step)
    evaluator = RLTestSetEvaluator(_CachedDataset([builder]), max_tokens=4)

    async def run() -> None:
        first_policy, second_policy = _GatedPolicy(), _GatedPolicy()
        first = asyncio.create_task(evaluator.eval_token_completer(first_policy))
        await first_policy.started.wait()
        second = asyncio.create_task(evaluator.eval_token_completer(second_policy))
        await second_policy.started.wait()
        if first_eval_ends_by == "finishing":
            first_policy.release.set()
            await first
        else:
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        second_policy.release.set()
        await second

    asyncio.run(run())
    # The second eval's sandbox was alive when it stepped (the first eval's
    # step comes first if it finished), and every sandbox was cleaned up.
    assert alive_at_step == ([True, True] if first_eval_ends_by == "finishing" else [True])
    assert len(builder.created) == 2
    assert not any(sandbox.alive for sandbox in builder.created)
