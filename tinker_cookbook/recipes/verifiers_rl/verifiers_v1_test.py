"""Tests for the native verifiers v1 integration."""

from types import SimpleNamespace
from typing import cast

import pytest
import tinker
from renderers import RenderedTokens
from renderers.base import ParsedResponse, RendererPool

try:
    import verifiers.v1 as vf
except ImportError:
    pytest.skip("verifiers v1 not installed", allow_module_level=True)


def _assistant(parent: int, token: int) -> "vf.MessageNode":
    return vf.MessageNode(
        parent=parent,
        message=vf.AssistantMessage(content=str(token)),
        sampled=True,
        token_ids=[token - 1, token],
        mask=[False, True],
        logprobs=[-0.5],
    )


def _linear_trace() -> "vf.Trace":
    return vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="test")),
        nodes=[
            vf.MessageNode(
                message=vf.UserMessage(content="test"), token_ids=[1], mask=[False]
            ),
            _assistant(0, 3),
        ],
    )


def test_load_tasks_bounds_infinite_tasksets() -> None:
    from tinker_cookbook.recipes.verifiers_rl.verifiers_env import load_tasks

    class InfiniteTaskset:
        INFINITE = True

        def load(self):
            index = 0
            while True:
                yield index
                index += 1

    taskset = cast(vf.Taskset, InfiniteTaskset())
    assert load_tasks(taskset, 3) == [0, 1, 2]
    with pytest.raises(ValueError, match="num_tasks is required"):
        load_tasks(taskset, None)
    with pytest.raises(ValueError, match="non-negative"):
        load_tasks(taskset, -1)


def test_trace_conversion_preserves_branches_and_masks_shared_actions() -> None:
    from tinker_cookbook.recipes.verifiers_rl.verifiers_env import trace_to_trajectory
    from tinker_cookbook.rl.data_processing import trajectory_to_data

    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="test")),
        nodes=[
            vf.MessageNode(
                message=vf.UserMessage(content="test"), token_ids=[1], mask=[False]
            ),
            _assistant(0, 3),
            vf.MessageNode(
                parent=1,
                message=vf.ToolMessage(tool_call_id="a", content="left"),
                token_ids=[4],
                mask=[False],
            ),
            _assistant(2, 6),
            vf.MessageNode(
                parent=1,
                message=vf.ToolMessage(tool_call_id="b", content="right"),
                token_ids=[7],
                mask=[False],
            ),
            _assistant(4, 9),
        ],
    )

    trajectory = trace_to_trajectory(trace)

    assert [transition.ac.tokens for transition in trajectory.transitions] == [
        [3],
        [6],
        [3],
        [9],
    ]
    assert [transition.action_mask for transition in trajectory.transitions] == [
        1.0,
        1.0,
        0.0,
        1.0,
    ]
    data = trajectory_to_data(trajectory, traj_advantage=1.0)
    assert len(data) == 2
    assert [sum(datum.loss_fn_inputs["mask"].to_torch().tolist()) for datum in data] == [
        2.0,
        1.0,
    ]


def test_group_scored_partial_failure_drops_the_group() -> None:
    from tinker_cookbook.exceptions import AllTrajectoriesFailedError
    from tinker_cookbook.recipes.verifiers_rl.verifiers_env import traces_to_trajectory_group

    failed = vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="test")),
        errors=[vf.Error(type="ProviderError", message="failed")],
    )

    with pytest.raises(AllTrajectoriesFailedError, match="group-scored"):
        traces_to_trajectory_group(
            [_linear_trace(), failed], requires_group_scoring=True
        )

    group = traces_to_trajectory_group(
        [_linear_trace(), failed], requires_group_scoring=False
    )
    assert len(group.trajectories_G) == 1
    assert len(group.rollout_errors) == 1


@pytest.mark.asyncio
async def test_client_returns_v1_response_with_training_tokens() -> None:
    from verifiers.v1.dialects import ChatDialect

    from tinker_cookbook.recipes.verifiers_rl.tinker_client import TinkerClient

    class FakeRenderer:
        def render(self, messages, *, tools=None, add_generation_prompt=False):
            assert messages == [{"role": "user", "content": "hello"}]
            assert tools is None
            assert add_generation_prompt
            return RenderedTokens(
                token_ids=[1, 2],
                message_indices=[0, 0],
                sampled_mask=[False, False],
                is_content=[True, True],
                message_roles=["user"],
                message_tool_names=[None],
            )

        def get_stop_token_ids(self):
            return [99]

        def parse_response(self, token_ids, *, tools=None):
            assert token_ids == [3, 4]
            assert tools is None
            return ParsedResponse(content="world")

    class FakeSamplingClient:
        async def sample_async(self, prompt, num_samples, sampling_params):
            assert prompt.to_ints() == [1, 2]
            assert num_samples == 1
            assert sampling_params.stop == [99]
            return SimpleNamespace(
                sequences=[
                    SimpleNamespace(
                        tokens=[3, 4], logprobs=[-0.1, -0.2], stop_reason="stop"
                    )
                ]
            )

    client = TinkerClient.__new__(TinkerClient)
    client.sampling_client = cast(tinker.SamplingClient, FakeSamplingClient())
    client.renderer = cast(RendererPool, FakeRenderer())

    response = await client.get_response(
        ChatDialect(),
        {"messages": [{"role": "user", "content": "hello"}]},
        "model",
        vf.SamplingConfig(max_tokens=8, temperature=0.5),
    )

    assert response.message.content == "world"
    assert response.tokens is not None
    assert response.tokens.prompt_ids == [1, 2]
    assert response.tokens.completion_ids == [3, 4]
    assert response.raw is not None
    assert response.raw["choices"][0]["message"]["content"] == "world"
