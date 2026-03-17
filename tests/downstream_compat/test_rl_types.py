"""Downstream compatibility tests for tinker_cookbook.rl.types.

Validates that the RL type system — Env, StepResult, Transition, Trajectory,
TrajectoryGroup, EnvGroupBuilder, RLDataset, RLDatasetBuilder — remains stable.
"""

import inspect
from dataclasses import fields

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Logs,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
    TrajectoryGroup,
    Transition,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------


class TestTypeAliases:
    def test_action_is_list_int(self):
        val: Action = [1, 2, 3]
        assert isinstance(val, list)

    def test_metrics_is_dict(self):
        val: Metrics = {"acc": 0.5}
        assert isinstance(val, dict)

    def test_logs_is_dict(self):
        val: Logs = {"msg": "ok", "step": 1}
        assert isinstance(val, dict)

    def test_observation_alias_exists(self):
        assert Observation is not None


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------


class TestStepResult:
    def test_fields(self):
        names = {f.name for f in fields(StepResult)}
        expected = {
            "reward",
            "episode_done",
            "next_observation",
            "next_stop_condition",
            "metrics",
            "logs",
        }
        assert expected.issubset(names)

    def test_metrics_defaults_to_empty(self):
        sr = StepResult(
            reward=1.0,
            episode_done=False,
            next_observation=None,  # type: ignore[arg-type]
            next_stop_condition=[],
        )
        assert sr.metrics == {}
        assert sr.logs == {}


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------


class TestTransition:
    def test_fields(self):
        names = {f.name for f in fields(Transition)}
        expected = {"ob", "ac", "reward", "episode_done", "metrics", "logs"}
        assert expected.issubset(names)

    def test_constructable(self):
        t = Transition(
            ob=None,  # type: ignore[arg-type]
            ac=TokensWithLogprobs(tokens=[1, 2], maybe_logprobs=None),
            reward=0.5,
            episode_done=False,
        )
        assert t.reward == 0.5


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class TestEnv:
    def test_is_abstract(self):
        assert inspect.isabstract(Env)

    def test_has_initial_observation(self):
        assert hasattr(Env, "initial_observation")
        assert inspect.iscoroutinefunction(Env.initial_observation)

    def test_initial_observation_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert_params(Env.initial_observation, [])

    def test_has_step(self):
        assert hasattr(Env, "step")
        assert inspect.iscoroutinefunction(Env.step)

    def test_step_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert_params(Env.step, ["action"])


# ---------------------------------------------------------------------------
# Trajectory and TrajectoryGroup
# ---------------------------------------------------------------------------


class TestTrajectory:
    def test_fields(self):
        names = {f.name for f in fields(Trajectory)}
        assert "transitions" in names
        assert "final_ob" in names

    def test_frozen(self):
        assert Trajectory.__dataclass_params__.frozen  # type: ignore[attr-defined]


class TestTrajectoryGroup:
    def test_has_trajectories_field(self):
        names = {f.name for f in fields(TrajectoryGroup)}
        assert "trajectories_G" in names
        assert "metrics_G" in names

    def test_get_total_rewards_method(self):
        assert hasattr(TrajectoryGroup, "get_total_rewards")
        assert callable(TrajectoryGroup.get_total_rewards)


# ---------------------------------------------------------------------------
# EnvGroupBuilder
# ---------------------------------------------------------------------------


class TestEnvGroupBuilder:
    def test_is_abstract(self):
        assert inspect.isabstract(EnvGroupBuilder)

    def test_make_envs_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert hasattr(EnvGroupBuilder, "make_envs")
        assert inspect.iscoroutinefunction(EnvGroupBuilder.make_envs)
        assert_params(EnvGroupBuilder.make_envs, [])

    def test_compute_group_rewards_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert hasattr(EnvGroupBuilder, "compute_group_rewards")
        assert inspect.iscoroutinefunction(EnvGroupBuilder.compute_group_rewards)
        assert_params(EnvGroupBuilder.compute_group_rewards, ["trajectory_group", "env_group"])

    def test_logging_tags_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert hasattr(EnvGroupBuilder, "logging_tags")
        assert_params(EnvGroupBuilder.logging_tags, [])


# ---------------------------------------------------------------------------
# RLDataset and RLDatasetBuilder
# ---------------------------------------------------------------------------


class TestRLDataset:
    def test_is_abstract(self):
        assert inspect.isabstract(RLDataset)

    def test_get_batch_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert hasattr(RLDataset, "get_batch")
        assert_params(RLDataset.get_batch, ["index"])

    def test_has_len(self):
        assert hasattr(RLDataset, "__len__")


class TestRLDatasetBuilder:
    def test_has_call(self):
        assert callable(RLDatasetBuilder)
