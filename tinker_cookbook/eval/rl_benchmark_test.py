"""Tests for the RL test set → benchmark adapter."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

import pytest
import tinker

from tinker_cookbook.completers import StopCondition
from tinker_cookbook.eval.rl_benchmark import (
    RLTestSetBenchmarkBuilder,
    _LazyEnv,
)
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    StepResult,
    Trajectory,
)

# ---------------------------------------------------------------------------
# Test fixtures — minimal Env / EnvGroupBuilder stubs
# ---------------------------------------------------------------------------


class _StubEnv(Env):
    """Minimal env that returns a fixed reward on step()."""

    def __init__(self, reward: float, metrics: Metrics | None = None) -> None:
        self.reward = reward
        self._metrics = metrics or {}
        self.example_id = f"stub_{id(self)}"

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        return tinker.ModelInput.from_ints([1, 2, 3]), []

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        return StepResult(
            reward=self.reward,
            episode_done=True,
            next_observation=tinker.ModelInput.from_ints([]),
            next_stop_condition=[],
            metrics=self._metrics,
            logs={"expected": "42"},
        )


@dataclass(frozen=True)
class _StubGroupBuilder(EnvGroupBuilder):
    """Creates a single _StubEnv with the given reward."""

    reward: float
    metrics: dict | None = None
    tags: tuple[str, ...] = ()

    async def make_envs(self) -> Sequence[Env]:
        return [_StubEnv(self.reward, dict(self.metrics) if self.metrics else None)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return list(self.tags)


@dataclass(frozen=True)
class _MultiEnvGroupBuilder(EnvGroupBuilder):
    """Creates multiple envs (for testing the group_size assertion)."""

    n_envs: int = 3

    async def make_envs(self) -> Sequence[Env]:
        return [_StubEnv(1.0) for _ in range(self.n_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]


class _StubDataset(RLDataset):
    """Minimal RLDataset wrapping a flat list of builders."""

    def __init__(self, builders: list[EnvGroupBuilder]) -> None:
        self._builders = builders

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [self._builders[index]]

    def __len__(self) -> int:
        return len(self._builders)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLazyEnv:
    def test_delegates_to_inner(self):
        builder = _StubGroupBuilder(reward=1.0)
        lazy = _LazyEnv(builder, builder_idx=0)

        ob, stop = asyncio.get_event_loop().run_until_complete(lazy.initial_observation())
        assert ob is not None

        result = asyncio.get_event_loop().run_until_complete(lazy.step([42]))
        assert result.reward == 1.0
        assert result.episode_done is True

    def test_delegates_step_metrics(self):
        builder = _StubGroupBuilder(reward=0.5, metrics={"correct": 1.0})
        lazy = _LazyEnv(builder, builder_idx=7)

        asyncio.get_event_loop().run_until_complete(lazy.initial_observation())
        result = asyncio.get_event_loop().run_until_complete(lazy.step([1]))

        # Metrics pass through without internal keys injected
        assert result.metrics["correct"] == 1.0
        assert "_builder_idx" not in result.metrics

    def test_forwards_example_id(self):
        builder = _StubGroupBuilder(reward=1.0)
        lazy = _LazyEnv(builder, builder_idx=0)

        asyncio.get_event_loop().run_until_complete(lazy.initial_observation())
        assert hasattr(lazy, "example_id")
        assert lazy.example_id is not None

    def test_rejects_group_size_gt_1(self):
        builder = _MultiEnvGroupBuilder(n_envs=3)
        lazy = _LazyEnv(builder, builder_idx=0)

        with pytest.raises(ValueError, match="group_size=1"):
            asyncio.get_event_loop().run_until_complete(lazy.initial_observation())

    def test_step_before_init_fails(self):
        builder = _StubGroupBuilder(reward=1.0)
        lazy = _LazyEnv(builder, builder_idx=0)

        with pytest.raises(AssertionError):
            asyncio.get_event_loop().run_until_complete(lazy.step([1]))


class TestRLTestSetBenchmarkBuilder:
    def test_make_envs_returns_lazy_envs(self):
        builders = [
            _StubGroupBuilder(reward=1.0),
            _StubGroupBuilder(reward=0.0),
        ]
        benchmark = RLTestSetBenchmarkBuilder(builders, name="test_bench")

        # make_envs should return _LazyEnv instances
        envs = benchmark.make_envs(renderer=None, config=None)  # type: ignore
        assert len(envs) == 2
        assert all(isinstance(e, _LazyEnv) for e in envs)

    def test_aggregate_basic(self):
        builders = [
            _StubGroupBuilder(reward=1.0, metrics={"correct": 1.0}),
            _StubGroupBuilder(reward=0.0, metrics={"correct": 0.0}),
            _StubGroupBuilder(reward=1.0, metrics={"correct": 1.0}),
        ]
        benchmark = RLTestSetBenchmarkBuilder(builders, name="my_test")

        rewards = [1.0, 0.0, 1.0]
        metrics_list: list[Metrics] = [
            {"correct": 1.0},
            {"correct": 0.0},
            {"correct": 1.0},
        ]

        result = benchmark.aggregate(rewards, metrics_list)
        assert result.name == "my_test"
        assert result.num_examples == 3
        assert result.num_correct == 2
        assert abs(result.score - 2 / 3) < 1e-9
        # Metrics use env/all/ prefix for backward compat with RLTestSetEvaluator
        assert abs(result.metrics["env/all/reward/total"] - 2 / 3) < 1e-9
        assert abs(result.metrics["env/all/correct"] - 2 / 3) < 1e-9

    def test_aggregate_empty(self):
        benchmark = RLTestSetBenchmarkBuilder([], name="empty")
        result = benchmark.aggregate([], [])
        assert result.score == 0.0
        assert result.num_examples == 0

    def test_aggregate_per_tag(self):
        builders = [
            _StubGroupBuilder(reward=1.0, tags=("math", "gsm")),
            _StubGroupBuilder(reward=0.0, tags=("math", "gsm")),
            _StubGroupBuilder(reward=1.0, tags=("code",)),
        ]
        benchmark = RLTestSetBenchmarkBuilder(builders, name="tagged")

        rewards = [1.0, 0.0, 1.0]
        metrics_list: list[Metrics] = [
            {"correct": 1.0},
            {"correct": 0.0},
            {"correct": 1.0},
        ]

        result = benchmark.aggregate(rewards, metrics_list)

        # Per-tag metrics should exist for tags that are strict subsets
        # math: 2 examples (not all 3), so per-tag is emitted
        assert "env/math/reward/total" in result.metrics
        assert abs(result.metrics["env/math/reward/total"] - 0.5) < 1e-9

        # gsm: same 2 examples as math
        assert "env/gsm/reward/total" in result.metrics
        assert abs(result.metrics["env/gsm/reward/total"] - 0.5) < 1e-9

        # code: 1 example (strict subset)
        assert "env/code/reward/total" in result.metrics
        assert abs(result.metrics["env/code/reward/total"] - 1.0) < 1e-9

    def test_aggregate_per_tag_skips_universal_tags(self):
        """Tags that select all examples are not emitted as per-tag metrics."""
        builders = [
            _StubGroupBuilder(reward=1.0, tags=("universal_tag",)),
            _StubGroupBuilder(reward=0.0, tags=("universal_tag",)),
        ]
        benchmark = RLTestSetBenchmarkBuilder(builders, name="all_tags")

        rewards = [1.0, 0.0]
        metrics_list: list[Metrics] = [
            {},
            {},
        ]

        result = benchmark.aggregate(rewards, metrics_list)
        # "universal_tag" covers all 2 examples — should NOT be emitted as per-tag
        assert "env/universal_tag/reward/total" not in result.metrics
        # But env/all/ global aggregate is always present
        assert "env/all/reward/total" in result.metrics

    def test_name_attribute(self):
        builders = [_StubGroupBuilder(reward=1.0)]
        b = RLTestSetBenchmarkBuilder(builders, name="custom_name")
        assert b.name == "custom_name"
