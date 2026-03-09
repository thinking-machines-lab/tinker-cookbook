"""Tests for multi-turn weight assignment in trajectory_to_data.

Verifies that agent-generated tokens get mask=1 (trained on) and
environment-provided tokens get mask=0 (masked out) when trajectories
are converted to training data.
"""

import asyncio
from unittest.mock import MagicMock

import pytest
import tinker

from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.renderers.base import Message, ToolCall
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.types import Trajectory, Transition
from tinker_cookbook.tool_use import build_agent_tool_env, simple_tool_result, tool
from tinker_cookbook.tool_use.types import ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transition(
    ob_tokens: list[int],
    ac_tokens: list[int],
    logprobs: list[float] | None = None,
    reward: float = 0.0,
    done: bool = False,
) -> Transition:
    if logprobs is None:
        logprobs = [0.0] * len(ac_tokens)
    return Transition(
        ob=tinker.ModelInput.from_ints(ob_tokens),
        ac=TokensWithLogprobs(tokens=ac_tokens, maybe_logprobs=logprobs),
        reward=reward,
        episode_done=done,
    )


def _get_mask(datum: tinker.Datum) -> list[float]:
    return datum.loss_fn_inputs["mask"].to_torch().tolist()


@tool
async def _stub_tool() -> ToolResult:
    """A test tool."""
    return simple_tool_result("ok")


async def _zero_reward_fn(history) -> tuple[float, dict[str, float]]:
    return 0.0, {}


# ---------------------------------------------------------------------------
# Test A: Multi-turn prefix trajectory
# ---------------------------------------------------------------------------


class TestMultiTurnPrefixTrajectory:
    """3-turn trajectory where each observation is a prefix extension.

    Turn 1: ob=[1,2,3,4,5]                              ac=[10,11,12]
    Turn 2: ob=[1,2,3,4,5, 10,11,12, 20,21]             ac=[30,31]
    Turn 3: ob=[1,2,3,4,5, 10,11,12, 20,21, 30,31, 40]  ac=[50,51,52]
    """

    def _make_trajectory(self) -> Trajectory:
        return Trajectory(
            transitions=[
                _make_transition([1, 2, 3, 4, 5], [10, 11, 12]),
                _make_transition([1, 2, 3, 4, 5, 10, 11, 12, 20, 21], [30, 31]),
                _make_transition(
                    [1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 30, 31, 40],
                    [50, 51, 52],
                    done=True,
                ),
            ],
            final_ob=tinker.ModelInput.from_ints([]),
        )

    def test_returns_single_datum(self):
        data = trajectory_to_data(self._make_trajectory(), traj_advantage=1.0)
        assert len(data) == 1

    def test_mask_matches_expected(self):
        data = trajectory_to_data(self._make_trajectory(), traj_advantage=1.0)
        mask = _get_mask(data[0])
        # After [1:] shift:
        # targets: [2,3,4,5, 10,11,12, 20,21, 30,31, 40, 50,51,52]
        # mask:    [0,0,0,0,  1, 1, 1,  0, 0,  1, 1,  0,  1, 1, 1]
        expected = [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1]
        assert mask == expected

    def test_mask_sum_equals_action_token_count(self):
        data = trajectory_to_data(self._make_trajectory(), traj_advantage=1.0)
        mask = _get_mask(data[0])
        # 3 (ac1) + 2 (ac2) + 3 (ac3) = 8 action tokens
        assert sum(mask) == 8


# ---------------------------------------------------------------------------
# Test B: Single-turn trajectory
# ---------------------------------------------------------------------------


class TestSingleTurnTrajectory:
    def test_single_turn_mask(self):
        traj = Trajectory(
            transitions=[_make_transition([1, 2, 3], [10, 11, 12], done=True)],
            final_ob=tinker.ModelInput.from_ints([]),
        )
        data = trajectory_to_data(traj, traj_advantage=1.0)
        assert len(data) == 1
        mask = _get_mask(data[0])
        # After [1:]: targets=[2,3,10,11,12], mask=[0,0,1,1,1]
        assert mask == [0, 0, 1, 1, 1]
        assert sum(mask) == 3


# ---------------------------------------------------------------------------
# Test C: Prefix break splits into multiple Datums
# ---------------------------------------------------------------------------


class TestPrefixBreak:
    def test_non_prefix_observation_produces_two_datums(self):
        traj = Trajectory(
            transitions=[
                _make_transition([1, 2, 3], [10, 11]),
                _make_transition([50, 51, 52], [60, 61], done=True),
            ],
            final_ob=tinker.ModelInput.from_ints([]),
        )
        data = trajectory_to_data(traj, traj_advantage=1.0)
        assert len(data) == 2

    def test_each_datum_has_correct_mask(self):
        traj = Trajectory(
            transitions=[
                _make_transition([1, 2, 3], [10, 11]),
                _make_transition([50, 51, 52], [60, 61], done=True),
            ],
            final_ob=tinker.ModelInput.from_ints([]),
        )
        data = trajectory_to_data(traj, traj_advantage=1.0)
        # Datum 1: [1,2,3,10,11] → after shift: mask=[0,0,1,1]
        assert _get_mask(data[0]) == [0, 0, 1, 1]
        # Datum 2: [50,51,52,60,61] → after shift: mask=[0,0,1,1]
        assert _get_mask(data[1]) == [0, 0, 1, 1]


# ---------------------------------------------------------------------------
# Test D: End-to-end through build_agent_tool_env → rollout → trajectory_to_data
# ---------------------------------------------------------------------------


def _make_stub_renderer():
    """Mock renderer with deterministic tokens and extension property.

    Token mapping:
        system  → [100, 101]
        user    → [200, 201]
        assistant → [500] (header) + action_tokens (output)
        tool    → [400, 401]
        suffix  → [500] (assistant generation header)

    The suffix [500] matches the assistant header, so the extension
    property holds: build_generation_prompt([msgs]) is always a prefix
    of build_generation_prompt([msgs, asst_msg, tool_msg, ...]).
    """
    renderer = MagicMock()
    renderer.get_stop_sequences.return_value = ["<stop>"]

    # Side-channel to pass action tokens from _parse to _build (can't stash on
    # Message since it's a TypedDict).
    action_tokens_by_id: dict[int, list[int]] = {}

    parse_results = iter(
        [
            # Call 1: model makes a tool call
            (
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            function=ToolCall.FunctionBody(name="_stub_tool", arguments="{}"),
                            id="call_1",
                        )
                    ],
                ),
                True,
            ),
            # Call 2: model gives final answer (no tool calls)
            (Message(role="assistant", content="done"), True),
        ]
    )

    def _parse(tokens):
        msg, success = next(parse_results)
        action_tokens_by_id[id(msg)] = list(tokens)
        return msg, success

    renderer.parse_response.side_effect = _parse

    def _build(messages, role="assistant", prefill=None):
        tokens = []
        for msg in messages:
            r = msg["role"]
            if r == "system":
                tokens.extend([100, 101])
            elif r == "user":
                tokens.extend([200, 201])
            elif r == "assistant":
                tokens.extend([500])  # assistant header
                tokens.extend(action_tokens_by_id.get(id(msg), []))
            elif r == "tool":
                tokens.extend([400, 401])
        tokens.extend([500])  # suffix: assistant generation header
        return tinker.ModelInput.from_ints(tokens)

    renderer.build_generation_prompt.side_effect = _build
    return renderer


def _make_stub_policy():
    """TokenCompleter returning predetermined responses."""
    responses = iter(
        [
            TokensWithLogprobs(tokens=[300, 301], maybe_logprobs=[-0.5, -0.3]),
            TokensWithLogprobs(tokens=[310, 311], maybe_logprobs=[-0.4, -0.2]),
        ]
    )

    class StubPolicy(TokenCompleter):
        async def __call__(self, model_input, stop):
            return next(responses)

    return StubPolicy()


def _run_e2e_rollout():
    """Run end-to-end rollout and return (trajectory, data)."""
    env = build_agent_tool_env(
        renderer=_make_stub_renderer(),
        tools=[_stub_tool],
        initial_messages=[
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Do something"),
        ],
        reward_fn=_zero_reward_fn,
        max_turns=5,
    )
    traj = asyncio.run(do_single_rollout(_make_stub_policy(), env))
    data = trajectory_to_data(traj, traj_advantage=1.0)
    return traj, data


class TestEndToEndToolUseRollout:
    def test_trajectory_has_two_transitions(self):
        traj, _ = _run_e2e_rollout()
        assert len(traj.transitions) == 2
        assert traj.transitions[0].episode_done is False
        assert traj.transitions[1].episode_done is True

    def test_produces_single_datum(self):
        _, data = _run_e2e_rollout()
        assert len(data) == 1

    def test_mask_only_on_agent_tokens(self):
        _, data = _run_e2e_rollout()
        mask = _get_mask(data[0])
        # Full sequence: [100,101, 200,201, 500, 300,301, 400,401, 500, 310,311]
        #                 sys      user     hdr  ac1      tool     hdr  ac2
        # Full mask:      [0,  0,   0,  0,   0,   1,  1,   0,  0,   0,   1,  1]
        # After [1:] shift:
        # Mask:           [0,  0,   0,   0,   1,   1,   0,   0,   0,   1,   1]
        expected = [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]
        assert mask == expected
        assert sum(mask) == 4  # 2 tokens per action × 2 actions


# ---------------------------------------------------------------------------
# Test E: Zero reward for distillation
# ---------------------------------------------------------------------------


class TestZeroReward:
    def test_zero_reward_returns_zero(self):
        mod = pytest.importorskip(
            "tinker_cookbook.distillation.multiturn_env",
            reason="requires modal",
        )
        result = asyncio.run(mod._zero_reward([Message(role="user", content="test")]))
        assert result == (0.0, {})

    def test_zero_reward_ignores_history_content(self):
        mod = pytest.importorskip(
            "tinker_cookbook.distillation.multiturn_env",
            reason="requires modal",
        )
        for history in [[], [Message(role="user", content="x")] * 50]:
            result = asyncio.run(mod._zero_reward(history))
            assert result == (0.0, {})

    def test_env_group_builder_compute_group_rewards_returns_zeros(self):
        mod = pytest.importorskip(
            "tinker_cookbook.distillation.multiturn_env",
            reason="requires modal",
        )
        builder = mod.HarborDistillationEnvGroupBuilder(
            task=MagicMock(),
            model_name="test",
            renderer_name="test",
            max_turns=5,
            group_size=2,
        )
        builder._sandboxes = []
        trajectories = [MagicMock(), MagicMock()]
        result = asyncio.run(builder.compute_group_rewards(trajectories, []))
        assert result == [(0.0, {}), (0.0, {})]
