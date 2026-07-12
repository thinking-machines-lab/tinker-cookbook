"""Tests for the metadata-capture additions to the core RL types.

Everything here is additive with defaults: existing envs and recipes that
construct StepResult / Transition / RolloutSummaryGroup without the new
fields must be unaffected.
"""

from collections.abc import Mapping, Sequence

import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.message_env import MessageStepResult
from tinker_cookbook.rl.rollout_logging import RolloutSummaryGroup
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    StepResult,
    ToolCallRecord,
    Trajectory,
    TrajectoryGroup,
    Transition,
)


def _minimal_step_result(**kwargs) -> StepResult:
    return StepResult(
        reward=1.0,
        episode_done=True,
        next_observation=tinker.ModelInput.from_ints([1]),
        next_stop_condition=[],
        **kwargs,
    )


def _minimal_transition(**kwargs) -> Transition:
    return Transition(
        ob=tinker.ModelInput.from_ints([1]),
        ac=TokensWithLogprobs(tokens=[2], maybe_logprobs=[-0.1]),
        reward=0.0,
        episode_done=True,
        **kwargs,
    )


class TestStepResultDefaults:
    def test_old_constructor_still_works(self):
        result = _minimal_step_result()
        assert result.attrs == {}
        assert result.tool_calls is None

    def test_new_fields_accepted(self):
        record = ToolCallRecord(name="search", args_json="{}", error_type=None, should_stop=False)
        result = _minimal_step_result(attrs={"phase": "solve"}, tool_calls=[record])
        assert result.attrs == {"phase": "solve"}
        assert result.tool_calls == [record]

    def test_attrs_instances_are_independent(self):
        a = _minimal_step_result()
        b = _minimal_step_result()
        a.attrs["k"] = "v"
        assert b.attrs == {}


class TestTransitionDefaults:
    def test_old_constructor_still_works(self):
        transition = _minimal_transition()
        assert transition.attrs == {}
        assert transition.tool_calls is None

    def test_new_fields_accepted(self):
        record = ToolCallRecord(
            name="calc", args_json='{"x": 1}', error_type="execution_failed", should_stop=True
        )
        transition = _minimal_transition(attrs={"tool": "calc"}, tool_calls=[record])
        assert transition.attrs == {"tool": "calc"}
        assert transition.tool_calls == [record]


class TestMessageStepResultDefaults:
    def test_old_constructor_still_works(self):
        result = MessageStepResult(reward=0.0, episode_done=True, next_messages=[])
        assert result.attrs == {}
        assert result.tool_calls is None


class TestEnvGroupBuilderMetadata:
    def test_default_is_empty_mapping(self):
        class MinimalBuilder(EnvGroupBuilder):
            async def make_envs(self) -> Sequence[Env]:
                return []

        builder = MinimalBuilder()
        metadata = builder.metadata()
        assert isinstance(metadata, Mapping)
        assert dict(metadata) == {}
        # Sibling of logging_tags, which keeps its default too.
        assert builder.logging_tags() == []

    def test_override(self):
        class MyBuilder(EnvGroupBuilder):
            async def make_envs(self) -> Sequence[Env]:
                return []

            def metadata(self) -> Mapping[str, str | int | float]:
                return {"dataset": "gsm8k", "difficulty": 3, "row_id": "gsm8k-42"}

        assert MyBuilder().metadata() == {
            "dataset": "gsm8k",
            "difficulty": 3,
            "row_id": "gsm8k-42",
        }


class TestRolloutSummaryGroupDefaults:
    def test_metadata_defaults_to_empty(self):
        group = RolloutSummaryGroup(
            trajectory_group=TrajectoryGroup(
                trajectories_G=[
                    Trajectory(transitions=[], final_ob=tinker.ModelInput.from_ints([]))
                ],
                final_rewards_G=[0.0],
                metrics_G=[{}],
            ),
            tags=["unit"],
        )
        assert dict(group.metadata) == {}
        assert group.sampling_client_step is None
