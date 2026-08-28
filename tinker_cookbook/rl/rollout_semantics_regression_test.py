"""Regression lock for the unconfigured end-to-end rollout semantics
(no RolloutLimits, no ParseErrorPolicy, no TerminationRewardPolicy).

These tests pin the behavior of the rollout loop
(``do_single_rollout`` + ``EnvFromMessageEnv`` + ``AgentToolMessageEnv``)
with scripted fake completers, renderers, and tools.  They define the
"simple" rollout semantics that configured rollout strategies must preserve:

1. Mid-turn length stop (sampler ``stop_reason == "length"``) ends the episode
   with ``context_overflow_reward`` *before* the message env's step runs.
2. The ``max_trajectory_tokens`` post-step check ends the episode with a flat
   ``context_overflow_reward``; ``reward_fn`` is skipped.
3. Prompt overflow in ``initial_observation`` ends the rollout gracefully
   with ``stop_reason="max_tokens"`` and the flat ``context_overflow_reward``
   (``FailFast`` would otherwise drop the whole group on one oversized
   prompt).
4. Parse failure applies ``failed_parse_reward`` one-shot with
   ``terminate_on_parse_error`` semantics; ``reward_fn`` is skipped.
5. Hitting ``max_turns`` emits ``metrics["max_turns"]`` while the graded
   reward stands.
6. ``RetryOnFailure`` is all-or-nothing: retry-budget exhaustion re-raises,
   and ``AllTrajectoriesFailedError`` skips the whole group.

If a change breaks one of these tests, it changes user-visible training
behavior — do not "fix" the test without an explicit decision.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import tinker

from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.exceptions import AllTrajectoriesFailedError
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import Message, ParseTermination, ToolCall, ToolSpec
from tinker_cookbook.rl.rollout_strategy import RetryOnFailure
from tinker_cookbook.rl.rollouts import (
    _do_group_rollout_and_filter_constant_reward_impl,
    do_single_rollout,
)
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, StepResult, Trajectory
from tinker_cookbook.tool_use.agent_tool_message_env import build_agent_tool_env
from tinker_cookbook.tool_use.tools import simple_tool_result
from tinker_cookbook.tool_use.types import ToolInput, ToolResult

# ---------------------------------------------------------------------------
# Scripted fakes
# ---------------------------------------------------------------------------


class ScriptedCompleter(TokenCompleter):
    """Token completer that replays a fixed sequence of results."""

    def __init__(self, results: list[TokensWithLogprobs]):
        self._results = list(results)
        self.calls = 0

    async def __call__(
        self, model_input: tinker.ModelInput, stop: Any, *, max_tokens: int | None = None
    ) -> TokensWithLogprobs:
        self.calls += 1
        return self._results.pop(0)


class ScriptedRenderer:
    """Fake renderer: fixed tokens per message; parse_response replays a script.

    Prompt length grows linearly with the number of messages, which lets tests
    drive the ``max_trajectory_tokens`` overflow check deterministically.
    """

    def __init__(
        self,
        parse_results: list[tuple[Message, ParseTermination]],
        tokens_per_message: int = 10,
    ):
        self._parse_results = list(parse_results)
        self.tokens_per_message = tokens_per_message
        self.parse_calls = 0

    def get_stop_sequences(self) -> list[str]:
        return ["<stop>"]

    def build_generation_prompt(self, messages: list[Message], **kwargs: Any) -> tinker.ModelInput:
        return tinker.ModelInput.from_ints(list(range(self.tokens_per_message * len(messages))))

    def parse_response(self, action: list[int]) -> tuple[Message, ParseTermination]:
        self.parse_calls += 1
        return self._parse_results.pop(0)


class RecordingReward:
    """reward_fn stub that records how often it was called."""

    def __init__(self, reward: float = 0.7):
        self.calls = 0
        self.reward = reward

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        self.calls += 1
        return self.reward, {"graded": 1.0}


class EchoTool:
    """Minimal tool that records invocations and returns fixed text."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo tool"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def to_spec(self) -> ToolSpec:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    async def run(self, input: ToolInput) -> ToolResult:
        self.calls += 1
        return simple_tool_result("echoed", call_id=input.call_id or "", name=self.name)


def _tokens(tokens: list[int], stop_reason: tinker.StopReason = "stop") -> TokensWithLogprobs:
    return TokensWithLogprobs(
        tokens=tokens, maybe_logprobs=[0.0] * len(tokens), stop_reason=stop_reason
    )


def _tool_call_message(name: str = "echo") -> Message:
    return {
        "role": "assistant",
        "content": "calling a tool",
        "tool_calls": [
            ToolCall(id="call_1", function=ToolCall.FunctionBody(name=name, arguments="{}"))
        ],
    }


def _final_message() -> Message:
    return {"role": "assistant", "content": "done"}


INITIAL_MESSAGES: list[Message] = [{"role": "user", "content": "please do the task"}]


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1. Mid-turn length stop
# ---------------------------------------------------------------------------


class TestMidTurnLengthStop:
    def test_length_stop_ends_episode_before_env_step(self):
        """A sampler 'length' stop ends the episode with context_overflow_reward
        (-0.1 default) before the message env runs: no parse, no tools, no grading."""
        renderer = ScriptedRenderer(parse_results=[])
        tool = EchoTool()
        reward_fn = RecordingReward()
        env = build_agent_tool_env(
            renderer=cast(Renderer, renderer),
            tools=[tool],
            initial_messages=INITIAL_MESSAGES,
            reward_fn=reward_fn,
        )
        policy = ScriptedCompleter([_tokens([1, 2, 3], stop_reason="length")])

        traj: Trajectory = _run(do_single_rollout(policy, env))

        assert len(traj.transitions) == 1
        final = traj.transitions[0]
        assert final.reward == -0.1  # default context_overflow_reward
        assert final.episode_done is True
        assert final.metrics["max_tokens_reached"] == 1.0
        # The env step never ran: nothing parsed, no tools, no grading.
        assert renderer.parse_calls == 0
        assert tool.calls == 0
        assert reward_fn.calls == 0

    def test_length_stop_uses_configured_overflow_reward(self):
        renderer = ScriptedRenderer(parse_results=[])
        env = build_agent_tool_env(
            renderer=cast(Renderer, renderer),
            tools=[],
            initial_messages=INITIAL_MESSAGES,
            reward_fn=RecordingReward(),
            context_overflow_reward=-0.5,
        )
        policy = ScriptedCompleter([_tokens([1], stop_reason="length")])

        traj = _run(do_single_rollout(policy, env))

        assert traj.transitions[-1].reward == -0.5


# ---------------------------------------------------------------------------
# 2. max_trajectory_tokens post-step check
# ---------------------------------------------------------------------------


class TestMaxTrajectoryTokens:
    def test_overflow_after_step_flat_penalty_reward_fn_skipped(self):
        """When the rendered conversation exceeds max_trajectory_tokens after a
        step, the episode ends with a flat context_overflow_reward and the
        reward_fn never runs (grade-then-clamp applies only with a
        TerminationRewardPolicy, not here)."""
        # 30 tokens/message: initial obs = 30 <= 100; after turn 1 the history is
        # [user, assistant, tool] = 90 tokens... use limit 80 so it overflows.
        renderer = ScriptedRenderer(
            parse_results=[(_tool_call_message(), ParseTermination.STOP_SEQUENCE)],
            tokens_per_message=30,
        )
        tool = EchoTool()
        reward_fn = RecordingReward()
        env = build_agent_tool_env(
            renderer=cast(Renderer, renderer),
            tools=[tool],
            initial_messages=INITIAL_MESSAGES,
            reward_fn=reward_fn,
            max_trajectory_tokens=80,
        )
        policy = ScriptedCompleter([_tokens([1, 2])])

        traj = _run(do_single_rollout(policy, env))

        assert len(traj.transitions) == 1
        final = traj.transitions[0]
        assert final.episode_done is True
        assert final.reward == -0.1  # flat penalty replaces any task reward
        assert final.metrics["context_overflow"] == 1.0
        assert tool.calls == 1  # the tool DID run before the overflow check
        assert reward_fn.calls == 0  # grading skipped


# ---------------------------------------------------------------------------
# 3. Initial prompt overflow ends the rollout gracefully
# ---------------------------------------------------------------------------


class TestInitialPromptOverflow:
    """A too-long initial prompt ends the rollout gracefully with the flat
    ``context_overflow_reward`` and ``stop_reason="max_tokens"`` (a single
    synthetic transition with empty observation/action, which contributes no
    training tokens) rather than raising, which would kill the whole group
    under FailFast."""

    def test_initial_observation_overflow_stops_gracefully(self):
        renderer = ScriptedRenderer(parse_results=[], tokens_per_message=60)
        reward_fn = RecordingReward()
        env = build_agent_tool_env(
            renderer=cast(Renderer, renderer),
            tools=[],
            initial_messages=INITIAL_MESSAGES,
            reward_fn=reward_fn,
            max_trajectory_tokens=50,
        )
        policy = ScriptedCompleter([])  # would raise IndexError if sampled from

        traj: Trajectory = _run(do_single_rollout(policy, env))

        assert policy.calls == 0  # no sampling call was made
        assert traj.stop_reason == "max_tokens"
        assert len(traj.transitions) == 1
        final = traj.transitions[0]
        assert final.reward == -0.1  # flat context_overflow_reward
        assert final.episode_done is True
        assert final.metrics["max_tokens_reached"] == 1.0
        assert final.metrics["stop/max_tokens"] == 1.0
        assert "too long for the model's context window" in str(
            final.logs["initial_observation_overflow"]
        )
        # Empty-episode representation: no tokens, so no training datums.
        assert final.ob.length == 0
        assert final.ac.tokens == []
        assert reward_fn.calls == 0  # grading skipped

    def test_group_survives_one_oversized_prompt(self):
        """End-to-end group of 2 under the default FailFast strategy: the
        member with an oversized prompt stops with stop/max_tokens + the
        overflow reward, while the other member completes normally."""
        from tinker_cookbook.rl.rollouts import do_group_rollout

        oversized_renderer = ScriptedRenderer(parse_results=[], tokens_per_message=60)
        oversized_env = build_agent_tool_env(
            renderer=cast(Renderer, oversized_renderer),
            tools=[],
            initial_messages=INITIAL_MESSAGES,
            reward_fn=RecordingReward(),
            max_trajectory_tokens=50,
        )
        normal_renderer = ScriptedRenderer(
            parse_results=[(_final_message(), ParseTermination.STOP_SEQUENCE)]
        )
        normal_reward_fn = RecordingReward(reward=0.7)
        normal_env = build_agent_tool_env(
            renderer=cast(Renderer, normal_renderer),
            tools=[],
            initial_messages=INITIAL_MESSAGES,
            reward_fn=normal_reward_fn,
            max_trajectory_tokens=50,
        )

        class _TwoEnvGroupBuilder(EnvGroupBuilder):
            async def make_envs(self) -> list[Env]:
                return [oversized_env, normal_env]

        policy = ScriptedCompleter([_tokens([1, 2])])  # only the normal env samples

        group = _run(do_group_rollout(_TwoEnvGroupBuilder(), policy))

        assert len(group.trajectories_G) == 2
        overflow_traj, normal_traj = group.trajectories_G
        assert overflow_traj.stop_reason == "max_tokens"
        assert overflow_traj.transitions[-1].metrics["stop/max_tokens"] == 1.0
        assert normal_traj.stop_reason == "completed"
        assert group.get_total_rewards() == [-0.1, 0.7]
        assert normal_reward_fn.calls == 1
        assert policy.calls == 1


# ---------------------------------------------------------------------------
# 4. Parse failure semantics
# ---------------------------------------------------------------------------


class TestParseFailure:
    def test_parse_failure_terminates_with_failed_parse_reward(self):
        """Default: one malformed response ends the episode with
        failed_parse_reward (-0.1 via build_agent_tool_env); reward_fn skipped."""
        renderer = ScriptedRenderer(parse_results=[(_final_message(), ParseTermination.MALFORMED)])
        reward_fn = RecordingReward()
        env = build_agent_tool_env(
            renderer=cast(Renderer, renderer),
            tools=[],
            initial_messages=INITIAL_MESSAGES,
            reward_fn=reward_fn,
        )
        policy = ScriptedCompleter([_tokens([1, 2])])

        traj = _run(do_single_rollout(policy, env))

        assert len(traj.transitions) == 1
        final = traj.transitions[0]
        assert final.reward == -0.1
        assert final.episode_done is True
        assert final.metrics["parse_error"] == 1.0
        assert reward_fn.calls == 0

    def test_parse_failure_no_terminate_continues_episode(self):
        """With terminate_on_parse_error=False the episode continues; the
        failed turn keeps the penalty and a later clean turn is graded."""
        renderer = ScriptedRenderer(
            parse_results=[
                (_final_message(), ParseTermination.MALFORMED),
                (_final_message(), ParseTermination.STOP_SEQUENCE),
            ]
        )
        reward_fn = RecordingReward(reward=0.9)
        env = build_agent_tool_env(
            renderer=cast(Renderer, renderer),
            tools=[],
            initial_messages=INITIAL_MESSAGES,
            reward_fn=reward_fn,
            terminate_on_parse_error=False,
        )
        policy = ScriptedCompleter([_tokens([1]), _tokens([2])])

        traj = _run(do_single_rollout(policy, env))

        assert len(traj.transitions) == 2
        assert traj.transitions[0].reward == -0.1
        assert traj.transitions[0].episode_done is False
        assert traj.transitions[1].reward == 0.9
        assert traj.transitions[1].episode_done is True
        assert reward_fn.calls == 1


# ---------------------------------------------------------------------------
# 5. max_turns: metric emitted, graded reward stands
# ---------------------------------------------------------------------------


class TestMaxTurns:
    def test_max_turns_metric_with_reward_standing(self):
        """Hitting max_turns emits metrics['max_turns']=1.0 but the reward_fn
        still runs and its reward is kept (no truncation penalty)."""
        renderer = ScriptedRenderer(
            parse_results=[(_tool_call_message(), ParseTermination.STOP_SEQUENCE)]
        )
        tool = EchoTool()
        reward_fn = RecordingReward(reward=0.7)
        env = build_agent_tool_env(
            renderer=cast(Renderer, renderer),
            tools=[tool],
            initial_messages=INITIAL_MESSAGES,
            reward_fn=reward_fn,
            max_turns=1,
        )
        policy = ScriptedCompleter([_tokens([1, 2])])

        traj = _run(do_single_rollout(policy, env))

        assert len(traj.transitions) == 1
        final = traj.transitions[0]
        assert final.episode_done is True
        assert final.metrics["max_turns"] == 1.0
        assert final.reward == 0.7  # graded reward stands
        assert reward_fn.calls == 1
        assert tool.calls == 1


# ---------------------------------------------------------------------------
# 6. RetryOnFailure all-or-nothing
# ---------------------------------------------------------------------------


class _AlwaysFailPolicy(TokenCompleter):
    def __init__(self, error: BaseException | None = None):
        self.error = error or RuntimeError("scripted failure")

    async def __call__(
        self, model_input: tinker.ModelInput, stop: Any, *, max_tokens: int | None = None
    ) -> TokensWithLogprobs:
        raise self.error


class _TrivialEnv(Env):
    async def initial_observation(self) -> tuple[tinker.ModelInput, list[str]]:
        return tinker.ModelInput.from_ints([1, 2]), ["<stop>"]

    async def step(self, action: list[int], *, extra: Any = None) -> StepResult:
        return StepResult(
            reward=1.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=["<stop>"],
        )


class _TrivialEnvGroupBuilder(EnvGroupBuilder):
    def __init__(self, n_envs: int = 2):
        self.n_envs = n_envs

    async def make_envs(self) -> list[Env]:
        return [_TrivialEnv() for _ in range(self.n_envs)]


class TestRetryOnFailureAllOrNothing:
    def test_budget_exhaustion_reraises_no_partial_group(self):
        """When the retry budget runs out, the failing exception propagates —
        the strategy never returns a partial group of trajectories."""
        strategy = RetryOnFailure(max_retries=1)
        builder = _TrivialEnvGroupBuilder(n_envs=2)
        policy = _AlwaysFailPolicy()

        with pytest.raises(RuntimeError, match="scripted failure"):
            _run(strategy.execute(builder, policy))

    def test_all_trajectories_failed_skips_group(self):
        """AllTrajectoriesFailedError from the rollout path skips the group
        (returns None) instead of crashing the run."""
        builder = _TrivialEnvGroupBuilder(n_envs=2)
        sampling_client = MagicMock(spec=tinker.SamplingClient)
        with patch(
            "tinker_cookbook.rl.rollouts.do_group_rollout",
            side_effect=AllTrajectoriesFailedError("all failed"),
        ):
            result = _run(
                _do_group_rollout_and_filter_constant_reward_impl(
                    sampling_client,
                    builder,
                    max_tokens=16,
                    temperature=1.0,
                    do_remove_constant_reward_groups=False,
                    strategy=RetryOnFailure(max_retries=1),
                )
            )
        assert result is None


# ---------------------------------------------------------------------------
# StopReason foundation: stop/<reason> metrics + Trajectory.stop_reason
# ---------------------------------------------------------------------------


class TestStopReasonFoundation:
    """The stop-reason markers are additive: they mirror (never replace) the
    pre-existing metric keys, and Trajectory.stop_reason reflects them."""

    def test_stop_reason_values(self):
        from tinker_cookbook.rl.types import StopReason

        assert StopReason.COMPLETED == "completed"
        assert StopReason.TOOL_STOPPED == "tool_stopped"
        assert StopReason.MAX_TURNS == "max_turns"
        assert StopReason.MAX_TOKENS == "max_tokens"
        assert StopReason.MAX_SAMPLED_TOKENS == "max_sampled_tokens"
        assert StopReason.MAX_TOOL_CALLS == "max_tool_calls"
        assert StopReason.CONTEXT_OVERFLOW == "context_overflow"
        assert StopReason.PARSE_ERROR == "parse_error"
        assert StopReason.ROLLOUT_TIMEOUT == "rollout_timeout"

    def test_trajectory_stop_reason_defaults_to_none(self):
        traj = Trajectory(transitions=[], final_ob=tinker.ModelInput.empty())
        assert traj.stop_reason is None

    def test_length_stop_sets_max_tokens(self):
        renderer = ScriptedRenderer(parse_results=[])
        env = build_agent_tool_env(
            renderer=cast(Renderer, renderer),
            tools=[],
            initial_messages=INITIAL_MESSAGES,
            reward_fn=RecordingReward(),
        )
        traj = _run(do_single_rollout(ScriptedCompleter([_tokens([1], stop_reason="length")]), env))
        assert traj.stop_reason == "max_tokens"
        assert traj.transitions[-1].metrics["stop/max_tokens"] == 1.0
        assert traj.transitions[-1].metrics["max_tokens_reached"] == 1.0

    def test_context_overflow_sets_stop_reason(self):
        renderer = ScriptedRenderer(
            parse_results=[(_tool_call_message(), ParseTermination.STOP_SEQUENCE)],
            tokens_per_message=30,
        )
        env = build_agent_tool_env(
            renderer=cast(Renderer, renderer),
            tools=[EchoTool()],
            initial_messages=INITIAL_MESSAGES,
            reward_fn=RecordingReward(),
            max_trajectory_tokens=80,
        )
        traj = _run(do_single_rollout(ScriptedCompleter([_tokens([1])]), env))
        assert traj.stop_reason == "context_overflow"
        assert traj.transitions[-1].metrics["context_overflow"] == 1.0

    def test_parse_error_sets_stop_reason(self):
        renderer = ScriptedRenderer(parse_results=[(_final_message(), ParseTermination.MALFORMED)])
        env = build_agent_tool_env(
            renderer=cast(Renderer, renderer),
            tools=[],
            initial_messages=INITIAL_MESSAGES,
            reward_fn=RecordingReward(),
        )
        traj = _run(do_single_rollout(ScriptedCompleter([_tokens([1])]), env))
        assert traj.stop_reason == "parse_error"
        assert traj.transitions[-1].metrics["parse_error"] == 1.0

    def test_completed_sets_stop_reason(self):
        renderer = ScriptedRenderer(
            parse_results=[(_final_message(), ParseTermination.STOP_SEQUENCE)]
        )
        env = build_agent_tool_env(
            renderer=cast(Renderer, renderer),
            tools=[],
            initial_messages=INITIAL_MESSAGES,
            reward_fn=RecordingReward(),
        )
        traj = _run(do_single_rollout(ScriptedCompleter([_tokens([1])]), env))
        assert traj.stop_reason == "completed"

    def test_max_turns_sets_stop_reason_alongside_existing_metric(self):
        renderer = ScriptedRenderer(
            parse_results=[(_tool_call_message(), ParseTermination.STOP_SEQUENCE)]
        )
        env = build_agent_tool_env(
            renderer=cast(Renderer, renderer),
            tools=[EchoTool()],
            initial_messages=INITIAL_MESSAGES,
            reward_fn=RecordingReward(),
            max_turns=1,
        )
        traj = _run(do_single_rollout(ScriptedCompleter([_tokens([1])]), env))
        assert traj.stop_reason == "max_turns"
        assert traj.transitions[-1].metrics["max_turns"] == 1.0
        assert traj.transitions[-1].metrics["stop/max_turns"] == 1.0
