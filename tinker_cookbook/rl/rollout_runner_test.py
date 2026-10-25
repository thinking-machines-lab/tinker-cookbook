"""Tests for the configurable rollout runner (``rl/rollout_runner.py``).

Covers the runner-owned behaviors layered on top of the unconfigured loop
(whose default semantics are pinned by ``rollout_semantics_regression_test.py``):

- per-turn token budget math (each min-term binding, per-turn recomputation
  with a growing observation, the ``<= 1`` boundary, and trajectory-vs-sampled
  precedence),
- LENGTH-continues vs cumulative-budget exhaustion,
- runner-level ``max_turns``,
- the tool-call cap (pre-dispatch and mid-batch) through ``AgentToolMessageEnv``,
- graceful whole-rollout timeouts (partial trajectory returned) vs the raising
  per-sample timeout,
- hooks: message injection, stop clearing/overriding (including that the outer
  timeout is not clearable), and custom string stop reasons.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, cast

import pytest
import tinker

from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import (
    Message,
    ParseTermination,
    ToolCall,
    ToolSpec,
    UnparsedToolCall,
)
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy, RolloutLimits
from tinker_cookbook.rl.rollout_runner import (
    TRUNCATION_METRIC_KEY,
    RolloutHooks,
    RolloutState,
    SamplingTurnTimeoutError,
    run_rollout,
)
from tinker_cookbook.rl.types import (
    PARSE_ERROR_MASKED_METRIC_KEY,
    Env,
    StepResult,
    Trajectory,
)
from tinker_cookbook.tool_use.agent_tool_message_env import (
    AgentToolMessageEnv,
    build_agent_tool_env,
)
from tinker_cookbook.tool_use.tools import simple_tool_result
from tinker_cookbook.tool_use.types import ToolInput, ToolResult

# ---------------------------------------------------------------------------
# Scripted fakes (same patterns as rollout_semantics_regression_test.py, plus
# recording of the per-call max_tokens override)
# ---------------------------------------------------------------------------


class RecordingCompleter(TokenCompleter):
    """Replays scripted results and records the per-call ``max_tokens``."""

    def __init__(self, results: list[TokensWithLogprobs], sleep_on: set[int] | None = None):
        self._results = list(results)
        self._sleep_on = sleep_on or set()
        self.calls = 0
        self.max_tokens_seen: list[int | None] = []

    async def __call__(
        self, model_input: tinker.ModelInput, stop: Any, *, max_tokens: int | None = None
    ) -> TokensWithLogprobs:
        idx = self.calls
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        if idx in self._sleep_on:
            await asyncio.sleep(5.0)
        return self._results.pop(0)


class ScriptedRenderer:
    """Fake renderer: fixed tokens per message; parse_response replays a script."""

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


class RecordingHooks(RolloutHooks):
    """Hook base for tests: records every call; no-op behavior by default."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    async def on_turn_begin(self, state: RolloutState) -> list[Message] | None:
        self.events.append(("turn_begin", state.turn_index))
        return None

    async def on_stop_reason(self, reason: str, state: RolloutState) -> str | None:
        self.events.append(("stop_reason", reason))
        return reason

    async def on_grade(self, state: RolloutState) -> None:
        self.events.append(("grade", state.turn_index))

    async def on_artifacts(self, state: RolloutState) -> None:
        self.events.append(("artifacts", state.turn_index))

    def event_names(self) -> list[str]:
        return [name for name, _ in self.events]

    def stop_reasons_seen(self) -> list[str]:
        return [arg for name, arg in self.events if name == "stop_reason"]


def _tokens(tokens: list[int], stop_reason: tinker.StopReason = "stop") -> TokensWithLogprobs:
    return TokensWithLogprobs(
        tokens=tokens, maybe_logprobs=[0.0] * len(tokens), stop_reason=stop_reason
    )


def _tool_call_message(n_calls: int = 1, name: str = "echo") -> Message:
    return {
        "role": "assistant",
        "content": "calling tools",
        "tool_calls": [
            ToolCall(id=f"call_{i}", function=ToolCall.FunctionBody(name=name, arguments="{}"))
            for i in range(n_calls)
        ],
    }


def _final_message() -> Message:
    return {"role": "assistant", "content": "done"}


INITIAL_MESSAGES: list[Message] = [{"role": "user", "content": "please do the task"}]


def _build_env(
    renderer: ScriptedRenderer,
    reward_fn: RecordingReward | None = None,
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> EnvFromMessageEnv:
    return build_agent_tool_env(
        renderer=cast(Renderer, renderer),
        tools=tools if tools is not None else [EchoTool()],
        initial_messages=INITIAL_MESSAGES,
        reward_fn=reward_fn or RecordingReward(),
        **kwargs,
    )


def _stop_metric_keys(traj: Trajectory) -> list[str]:
    return [k for k in traj.transitions[-1].metrics if k.startswith("stop/")]


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Per-turn token budget math
# ---------------------------------------------------------------------------


class TestPerTurnBudget:
    """The budget is min(max_turn_tokens, max_trajectory_tokens - obs,
    max_sampled_tokens - sampled) over the non-None terms, recomputed before
    every policy call and passed as the per-call max_tokens override."""

    def test_no_limits_passes_no_override(self):
        renderer = ScriptedRenderer(
            parse_results=[(_final_message(), ParseTermination.STOP_SEQUENCE)]
        )
        policy = RecordingCompleter([_tokens([1, 2])])
        _run(run_rollout(policy, _build_env(renderer)))
        assert policy.max_tokens_seen == [None]

    def test_max_turn_tokens_binds_every_turn(self):
        renderer = ScriptedRenderer(
            parse_results=[
                (_tool_call_message(), ParseTermination.STOP_SEQUENCE),
                (_final_message(), ParseTermination.STOP_SEQUENCE),
            ]
        )
        policy = RecordingCompleter([_tokens([1, 2]), _tokens([3])])
        _run(run_rollout(policy, _build_env(renderer), limits=RolloutLimits(max_turn_tokens=7)))
        assert policy.max_tokens_seen == [7, 7]

    def test_trajectory_budget_recomputed_with_growing_obs(self):
        # 10 tokens/message: obs is 10 on turn 1, 30 on turn 2 ([user,
        # assistant, tool]) -> budgets 90 then 70.
        renderer = ScriptedRenderer(
            parse_results=[
                (_tool_call_message(), ParseTermination.STOP_SEQUENCE),
                (_final_message(), ParseTermination.STOP_SEQUENCE),
            ]
        )
        policy = RecordingCompleter([_tokens([1, 2]), _tokens([3])])
        _run(
            run_rollout(
                policy, _build_env(renderer), limits=RolloutLimits(max_trajectory_tokens=100)
            )
        )
        assert policy.max_tokens_seen == [90, 70]

    def test_sampled_budget_recomputed_with_sampled_tokens(self):
        renderer = ScriptedRenderer(
            parse_results=[
                (_tool_call_message(), ParseTermination.STOP_SEQUENCE),
                (_final_message(), ParseTermination.STOP_SEQUENCE),
            ]
        )
        policy = RecordingCompleter([_tokens([1, 2, 3]), _tokens([4])])
        _run(run_rollout(policy, _build_env(renderer), limits=RolloutLimits(max_sampled_tokens=10)))
        assert policy.max_tokens_seen == [10, 7]

    def test_min_of_all_terms_binds_per_turn(self):
        # obs: 10, 30, 50; sampled: 0, 3, 6. Terms per turn:
        #   turn 1: min(80, 90, 95) = 80  (max_turn_tokens binds)
        #   turn 2: min(80, 70, 92) = 70  (trajectory binds)
        #   turn 3: min(80, 50, 89) = 50  (trajectory binds tighter)
        renderer = ScriptedRenderer(
            parse_results=[
                (_tool_call_message(), ParseTermination.STOP_SEQUENCE),
                (_tool_call_message(), ParseTermination.STOP_SEQUENCE),
                (_final_message(), ParseTermination.STOP_SEQUENCE),
            ]
        )
        policy = RecordingCompleter([_tokens([1, 2, 3]), _tokens([4, 5, 6]), _tokens([7])])
        limits = RolloutLimits(max_turn_tokens=80, max_trajectory_tokens=100, max_sampled_tokens=95)
        _run(run_rollout(policy, _build_env(renderer), limits=limits))
        assert policy.max_tokens_seen == [80, 70, 50]

    def test_trajectory_remaining_of_one_stops_before_sampling(self):
        """Budget <= 1 boundary: remaining trajectory budget of 1 stops the
        rollout with MAX_TOKENS before any sampling call."""
        renderer = ScriptedRenderer(parse_results=[])
        reward_fn = RecordingReward()
        policy = RecordingCompleter([])
        traj: Trajectory = _run(
            run_rollout(
                policy,
                _build_env(renderer, reward_fn=reward_fn),
                limits=RolloutLimits(max_trajectory_tokens=11),  # obs=10 -> remaining=1
            )
        )
        assert policy.calls == 0
        assert traj.stop_reason == "max_tokens"
        assert len(traj.transitions) == 1
        final = traj.transitions[0]
        assert final.episode_done is True
        assert final.metrics["stop/max_tokens"] == 1.0
        assert final.ob.length == 0 and final.ac.tokens == []  # synthetic, no training tokens
        assert reward_fn.calls == 0  # runner stop: env never graded

    def test_trajectory_remaining_of_two_still_samples(self):
        renderer = ScriptedRenderer(
            parse_results=[(_final_message(), ParseTermination.STOP_SEQUENCE)]
        )
        policy = RecordingCompleter([_tokens([1])])
        traj = _run(
            run_rollout(
                policy,
                _build_env(renderer),
                limits=RolloutLimits(max_trajectory_tokens=12),  # obs=10 -> remaining=2
            )
        )
        assert policy.max_tokens_seen == [2]
        assert traj.stop_reason == "completed"

    def test_trajectory_budget_wins_over_sampled_budget(self):
        """When both cumulative budgets are exhausted, MAX_TOKENS is reported."""
        renderer = ScriptedRenderer(parse_results=[])
        policy = RecordingCompleter([])
        traj = _run(
            run_rollout(
                policy,
                _build_env(renderer),
                limits=RolloutLimits(max_trajectory_tokens=10, max_sampled_tokens=1),
            )
        )
        assert policy.calls == 0
        assert traj.stop_reason == "max_tokens"
        assert _stop_metric_keys(traj) == ["stop/max_tokens"]

    def test_sampled_budget_exhaustion_stops_with_max_sampled_tokens(self):
        renderer = ScriptedRenderer(
            parse_results=[
                (_tool_call_message(), ParseTermination.STOP_SEQUENCE),
                (_tool_call_message(), ParseTermination.STOP_SEQUENCE),
            ]
        )
        reward_fn = RecordingReward()
        policy = RecordingCompleter([_tokens([1, 2]), _tokens([3, 4])])
        traj = _run(
            run_rollout(
                policy,
                _build_env(renderer, reward_fn=reward_fn),
                limits=RolloutLimits(max_sampled_tokens=4),
            )
        )
        assert policy.calls == 2
        assert policy.max_tokens_seen == [4, 2]
        assert traj.stop_reason == "max_sampled_tokens"
        final = traj.transitions[-1]
        assert final.episode_done is True  # runner marked the last transition terminal
        assert final.metrics["stop/max_sampled_tokens"] == 1.0
        assert reward_fn.calls == 0


# ---------------------------------------------------------------------------
# LENGTH-continues vs cumulative exhaustion
# ---------------------------------------------------------------------------


class TestLengthContinue:
    def test_length_continue_counts_truncations_and_completes(self):
        """With terminate_on_length=False, a per-turn 'length' stop keeps the
        truncated turn and continues; the runner counts it."""
        renderer = ScriptedRenderer(
            parse_results=[
                # Truncated turn: parsed leniently, MALFORMED must NOT be
                # treated as a parse failure on this path.
                ({"role": "assistant", "content": "partial"}, ParseTermination.MALFORMED),
                (_final_message(), ParseTermination.STOP_SEQUENCE),
            ]
        )
        reward_fn = RecordingReward()
        env = _build_env(renderer, reward_fn=reward_fn, terminate_on_length=False)
        policy = RecordingCompleter([_tokens([1, 2, 3, 4, 5], stop_reason="length"), _tokens([6])])

        traj = _run(run_rollout(policy, env, limits=RolloutLimits(max_turn_tokens=5)))

        assert len(traj.transitions) == 2
        truncated, final = traj.transitions
        assert truncated.episode_done is False
        assert truncated.reward == 0.0  # no context_overflow penalty
        assert traj.stop_reason == "completed"
        assert final.metrics[TRUNCATION_METRIC_KEY] == 1.0
        assert reward_fn.calls == 1
        # The truncated assistant message made it into the conversation.
        message_env = cast(AgentToolMessageEnv, env.message_env)
        assert {"role": "assistant", "content": "partial"} in message_env.history

    def test_length_continues_until_trajectory_budget_exhausted(self):
        # obs grows 10 -> 20 -> 30 -> 40 as each truncated assistant message is
        # appended; with max_trajectory_tokens=35, turn 4's pre-sample check
        # sees remaining <= 1 and stops with MAX_TOKENS.
        partial_message: Message = {"role": "assistant", "content": "partial"}
        partial = (partial_message, ParseTermination.MALFORMED)
        renderer = ScriptedRenderer(parse_results=[partial, partial, partial])
        reward_fn = RecordingReward()
        env = _build_env(renderer, reward_fn=reward_fn, terminate_on_length=False)
        policy = RecordingCompleter([_tokens([1], stop_reason="length") for _ in range(3)])

        traj = _run(
            run_rollout(
                policy,
                env,
                limits=RolloutLimits(max_turn_tokens=5, max_trajectory_tokens=35),
            )
        )

        assert policy.calls == 3
        assert traj.stop_reason == "max_tokens"
        final = traj.transitions[-1]
        assert final.episode_done is True
        assert final.metrics["stop/max_tokens"] == 1.0
        assert final.metrics[TRUNCATION_METRIC_KEY] == 3.0
        assert reward_fn.calls == 0

    def test_legacy_terminate_on_length_stands_even_with_limits(self):
        """The env-side length-terminate default is untouched: configuring
        limits alone does not opt into LENGTH-continues."""
        renderer = ScriptedRenderer(parse_results=[])
        reward_fn = RecordingReward()
        env = _build_env(renderer, reward_fn=reward_fn)  # terminate_on_length default True
        policy = RecordingCompleter([_tokens([1, 2], stop_reason="length")])

        traj = _run(run_rollout(policy, env, limits=RolloutLimits(max_turn_tokens=5)))

        assert len(traj.transitions) == 1
        assert traj.stop_reason == "max_tokens"
        assert traj.transitions[0].reward == -0.1  # env's context_overflow_reward
        assert reward_fn.calls == 0

    def test_unsupported_message_env_falls_back_to_terminate(self):
        """terminate_on_length=False over a MessageEnv that can't record
        truncated responses falls back to the default terminate behavior."""

        class MinimalMessageEnv(MessageEnv):
            async def initial_observation(self) -> list[Message]:
                return list(INITIAL_MESSAGES)

            async def step(self, message: Message) -> MessageStepResult:
                raise AssertionError("step should not run on the length path")

        renderer = ScriptedRenderer(
            parse_results=[
                ({"role": "assistant", "content": "partial"}, ParseTermination.MALFORMED)
            ]
        )
        env = EnvFromMessageEnv(
            renderer=cast(Renderer, renderer),
            message_env=MinimalMessageEnv(),
            terminate_on_length=False,
        )
        policy = RecordingCompleter([_tokens([1], stop_reason="length")])

        traj = _run(run_rollout(policy, env))

        assert traj.stop_reason == "max_tokens"
        assert traj.transitions[0].episode_done is True


# ---------------------------------------------------------------------------
# Runner-level max_turns
# ---------------------------------------------------------------------------


class TestRunnerMaxTurns:
    def test_runner_max_turns_stops_before_next_sample(self):
        renderer = ScriptedRenderer(
            parse_results=[
                (_tool_call_message(), ParseTermination.STOP_SEQUENCE),
                (_tool_call_message(), ParseTermination.STOP_SEQUENCE),
            ]
        )
        reward_fn = RecordingReward()
        env = _build_env(renderer, reward_fn=reward_fn, max_turns=99)
        policy = RecordingCompleter([_tokens([1]), _tokens([2])])

        traj = _run(run_rollout(policy, env, limits=RolloutLimits(max_turns=2)))

        assert policy.calls == 2
        assert traj.stop_reason == "max_turns"
        final = traj.transitions[-1]
        assert final.episode_done is True
        assert _stop_metric_keys(traj) == ["stop/max_turns"]
        assert reward_fn.calls == 0  # runner stop: env never observed the episode end

    def test_env_level_max_turns_still_grades(self):
        """Env-level max_turns keeps its default semantics (graded reward
        stands) when runner limits are configured but its turn cap is not."""
        renderer = ScriptedRenderer(
            parse_results=[(_tool_call_message(), ParseTermination.STOP_SEQUENCE)]
        )
        reward_fn = RecordingReward(reward=0.7)
        env = _build_env(renderer, reward_fn=reward_fn, max_turns=1)
        policy = RecordingCompleter([_tokens([1])])

        traj = _run(run_rollout(policy, env, limits=RolloutLimits(max_turn_tokens=100)))

        assert traj.stop_reason == "max_turns"
        assert traj.transitions[-1].reward == 0.7
        assert reward_fn.calls == 1


# ---------------------------------------------------------------------------
# Tool-call cap (pre-dispatch + mid-batch) through AgentToolMessageEnv
# ---------------------------------------------------------------------------


def _noop_reward_factory() -> RecordingReward:
    return RecordingReward(reward=1.0)


class TestMaxToolCallsMessageEnv:
    def _make_env(
        self, max_tool_calls: int
    ) -> tuple[AgentToolMessageEnv, EchoTool, RecordingReward]:
        tool = EchoTool()
        reward_fn = _noop_reward_factory()
        env = AgentToolMessageEnv(
            tools=[tool],
            initial_messages=list(INITIAL_MESSAGES),
            max_turns=50,
            reward_fn=reward_fn,
            max_tool_calls=max_tool_calls,
        )
        _run(env.initial_observation())
        return env, tool, reward_fn

    def test_mid_batch_cap_executes_allowed_prefix_then_stops(self):
        env, tool, reward_fn = self._make_env(max_tool_calls=3)

        first = _run(env.step(_tool_call_message(n_calls=2)))
        assert first.episode_done is False
        assert tool.calls == 2

        second = _run(env.step(_tool_call_message(n_calls=2)))  # remaining budget: 1
        assert second.episode_done is True
        assert tool.calls == 3  # only the allowed prefix ran
        assert second.metrics["max_tool_calls"] == 1.0
        assert second.metrics["stop/max_tool_calls"] == 1.0
        assert "tool_calls_dropped" in second.logs
        assert reward_fn.calls == 1  # env-driven stop grades

    def test_pre_dispatch_cap_dispatches_nothing(self):
        env, tool, _ = self._make_env(max_tool_calls=2)

        _run(env.step(_tool_call_message(n_calls=2)))  # consumes the whole budget
        assert tool.calls == 2

        result = _run(env.step(_tool_call_message(n_calls=1)))  # remaining budget: 0
        assert result.episode_done is True
        assert tool.calls == 2  # nothing dispatched
        assert result.metrics["stop/max_tool_calls"] == 1.0

    def test_exact_budget_consumption_continues(self):
        env, tool, _ = self._make_env(max_tool_calls=2)
        result = _run(env.step(_tool_call_message(n_calls=2)))
        assert result.episode_done is False
        assert tool.calls == 2
        assert "max_tool_calls" not in result.metrics

    def test_set_max_tool_calls_tighter_cap_wins(self):
        env, _, _ = self._make_env(max_tool_calls=5)
        env.set_max_tool_calls(2)
        assert env.max_tool_calls == 2
        env.set_max_tool_calls(10)
        assert env.max_tool_calls == 2


class TestMaxToolCallsThroughRunner:
    def test_runner_limit_enforced_end_to_end(self):
        """RolloutLimits.max_tool_calls is pushed into the env via the
        set_max_tool_calls seam and enforced mid-batch."""
        renderer = ScriptedRenderer(
            parse_results=[(_tool_call_message(n_calls=2), ParseTermination.STOP_SEQUENCE)]
        )
        tool = EchoTool()
        reward_fn = RecordingReward()
        env = _build_env(renderer, reward_fn=reward_fn, tools=[tool])
        policy = RecordingCompleter([_tokens([1])])

        traj = _run(run_rollout(policy, env, limits=RolloutLimits(max_tool_calls=1)))

        assert cast(AgentToolMessageEnv, env.message_env).max_tool_calls == 1
        assert tool.calls == 1
        assert traj.stop_reason == "max_tool_calls"
        assert traj.transitions[-1].metrics["stop/max_tool_calls"] == 1.0
        assert reward_fn.calls == 1  # env-driven stop grades

    def test_unsupported_env_warns_and_continues(self, caplog: pytest.LogCaptureFixture):
        class TrivialEnv(Env):
            async def initial_observation(self) -> tuple[tinker.ModelInput, list[str]]:
                return tinker.ModelInput.from_ints([1, 2]), ["<stop>"]

            async def step(self, action: list[int], *, extra: Any = None) -> StepResult:
                return StepResult(
                    reward=1.0,
                    episode_done=True,
                    next_observation=tinker.ModelInput.empty(),
                    next_stop_condition=["<stop>"],
                )

        policy = RecordingCompleter([_tokens([1])])
        with caplog.at_level("WARNING"):
            traj = _run(run_rollout(policy, TrivialEnv(), limits=RolloutLimits(max_tool_calls=1)))
        assert any("max_tool_calls" in rec.message for rec in caplog.records)
        assert len(traj.transitions) == 1  # rollout ran normally


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------


class _NonSuspendingEnv(Env):
    """Env whose step blocks synchronously (no await), so the outer
    asyncio.timeout cannot cancel mid-rollout — exercising the between-turns
    deadline check specifically."""

    def __init__(self, sync_sleeps: list[float], done_on_step: int):
        self._sleeps = list(sync_sleeps)
        self._done_on_step = done_on_step
        self._steps = 0

    async def initial_observation(self) -> tuple[tinker.ModelInput, list[str]]:
        return tinker.ModelInput.from_ints([1, 2, 3]), ["<stop>"]

    async def step(self, action: list[int], *, extra: Any = None) -> StepResult:
        if self._sleeps:
            time.sleep(self._sleeps.pop(0))
        self._steps += 1
        done = self._steps >= self._done_on_step
        return StepResult(
            reward=1.0 if done else 0.0,
            episode_done=done,
            next_observation=tinker.ModelInput.from_ints([1, 2, 3]),
            next_stop_condition=["<stop>"],
            metrics={"stop/completed": 1.0} if done else {},
        )


class TestTimeouts:
    def test_whole_rollout_timeout_returns_partial_trajectory(self):
        """The outer timeout is graceful: completed turns are kept and the
        trajectory reports ROLLOUT_TIMEOUT instead of raising."""
        renderer = ScriptedRenderer(
            parse_results=[(_tool_call_message(), ParseTermination.STOP_SEQUENCE)]
        )
        reward_fn = RecordingReward()
        env = _build_env(renderer, reward_fn=reward_fn)
        policy = RecordingCompleter([_tokens([1]), _tokens([2])], sleep_on={1})

        traj = _run(run_rollout(policy, env, limits=RolloutLimits(rollout_timeout_seconds=0.1)))

        assert traj.stop_reason == "rollout_timeout"
        assert len(traj.transitions) == 1  # turn 1 completed and is kept
        final = traj.transitions[0]
        assert final.episode_done is True
        assert final.metrics["stop/rollout_timeout"] == 1.0
        assert reward_fn.calls == 0

    def test_timeout_before_any_turn_yields_synthetic_transition(self):
        renderer = ScriptedRenderer(parse_results=[])
        env = _build_env(renderer)
        policy = RecordingCompleter([_tokens([1])], sleep_on={0})

        traj = _run(run_rollout(policy, env, limits=RolloutLimits(rollout_timeout_seconds=0.05)))

        assert traj.stop_reason == "rollout_timeout"
        assert len(traj.transitions) == 1
        assert traj.transitions[0].ob.length == 0
        assert traj.transitions[0].ac.tokens == []

    def test_outer_timeout_is_not_hook_clearable(self):
        """A hook that clears every stop cannot clear the outer timeout: the
        loop is already cancelled when it fires."""

        class ClearEverything(RecordingHooks):
            async def on_stop_reason(self, reason: str, state: RolloutState) -> str | None:
                await super().on_stop_reason(reason, state)
                return None

        renderer = ScriptedRenderer(parse_results=[])
        env = _build_env(renderer)
        policy = RecordingCompleter([_tokens([1])], sleep_on={0})
        hooks = ClearEverything()

        traj = _run(
            run_rollout(
                policy, env, limits=RolloutLimits(rollout_timeout_seconds=0.05), hooks=hooks
            )
        )

        assert traj.stop_reason == "rollout_timeout"
        assert "rollout_timeout" not in hooks.stop_reasons_seen()  # hook never consulted
        assert "artifacts" in hooks.event_names()

    def test_between_turns_deadline_check_stops(self):
        env = _NonSuspendingEnv(sync_sleeps=[0.05], done_on_step=99)
        policy = RecordingCompleter([_tokens([1]), _tokens([2])])

        traj = _run(run_rollout(policy, env, limits=RolloutLimits(rollout_timeout_seconds=0.02)))

        assert policy.calls == 1  # the deadline check stopped turn 2
        assert traj.stop_reason == "rollout_timeout"

    def test_between_turns_check_is_hook_clearable(self):
        class ClearTimeout(RecordingHooks):
            async def on_stop_reason(self, reason: str, state: RolloutState) -> str | None:
                await super().on_stop_reason(reason, state)
                return None if reason == "rollout_timeout" else reason

        env = _NonSuspendingEnv(sync_sleeps=[0.05], done_on_step=2)
        policy = RecordingCompleter([_tokens([1]), _tokens([2])])
        hooks = ClearTimeout()

        traj = _run(
            run_rollout(
                policy, env, limits=RolloutLimits(rollout_timeout_seconds=0.02), hooks=hooks
            )
        )

        assert "rollout_timeout" in hooks.stop_reasons_seen()
        assert policy.calls == 2  # cleared and continued to completion
        assert traj.stop_reason == "completed"

    def test_sampling_turn_timeout_raises(self):
        renderer = ScriptedRenderer(parse_results=[])
        env = _build_env(renderer)
        policy = RecordingCompleter([_tokens([1])], sleep_on={0})

        with pytest.raises(SamplingTurnTimeoutError):
            _run(run_rollout(policy, env, limits=RolloutLimits(sampling_turn_timeout_seconds=0.05)))

    def test_sampling_timeout_raises_even_inside_rollout_timeout(self):
        """The per-sample timeout keeps its raising semantics when the
        (larger) whole-rollout timeout is also configured."""
        renderer = ScriptedRenderer(parse_results=[])
        env = _build_env(renderer)
        policy = RecordingCompleter([_tokens([1])], sleep_on={0})

        with pytest.raises(SamplingTurnTimeoutError):
            _run(
                run_rollout(
                    policy,
                    env,
                    limits=RolloutLimits(
                        rollout_timeout_seconds=30.0, sampling_turn_timeout_seconds=0.05
                    ),
                )
            )


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


class TestHooks:
    def test_default_hooks_are_noops(self):
        class PassHooks(RolloutHooks):
            pass

        renderer = ScriptedRenderer(
            parse_results=[(_final_message(), ParseTermination.STOP_SEQUENCE)]
        )
        reward_fn = RecordingReward(reward=0.9)
        env = _build_env(renderer, reward_fn=reward_fn)
        policy = RecordingCompleter([_tokens([1])])

        traj = _run(run_rollout(policy, env, hooks=PassHooks()))

        assert traj.stop_reason == "completed"
        assert traj.transitions[-1].reward == 0.9

    def test_on_turn_begin_injects_messages_and_budget_sees_them(self):
        class InjectOnTurnOne(RecordingHooks):
            async def on_turn_begin(self, state: RolloutState) -> list[Message] | None:
                await super().on_turn_begin(state)
                if state.turn_index == 1:
                    return [{"role": "user", "content": "keep going"}]
                return None

        renderer = ScriptedRenderer(
            parse_results=[
                (_tool_call_message(), ParseTermination.STOP_SEQUENCE),
                (_final_message(), ParseTermination.STOP_SEQUENCE),
            ]
        )
        env = _build_env(renderer)
        policy = RecordingCompleter([_tokens([1]), _tokens([2])])
        hooks = InjectOnTurnOne()

        traj = _run(
            run_rollout(policy, env, limits=RolloutLimits(max_trajectory_tokens=200), hooks=hooks)
        )

        # Turn 1: obs 10 -> budget 190. Turn 2: [user, assistant, tool] +
        # injected user = 4 messages -> obs 40 -> budget 160 (injected tokens
        # count toward the trajectory budget).
        assert policy.max_tokens_seen == [190, 160]
        message_env = cast(AgentToolMessageEnv, env.message_env)
        assert {"role": "user", "content": "keep going"} in message_env.history
        assert traj.stop_reason == "completed"

    def test_injection_without_env_support_raises(self):
        class AlwaysInject(RecordingHooks):
            async def on_turn_begin(self, state: RolloutState) -> list[Message] | None:
                return [{"role": "user", "content": "hi"}]

        class TrivialEnv(Env):
            async def initial_observation(self) -> tuple[tinker.ModelInput, list[str]]:
                return tinker.ModelInput.from_ints([1]), ["<stop>"]

            async def step(self, action: list[int], *, extra: Any = None) -> StepResult:
                raise AssertionError("unreachable")

        with pytest.raises(TypeError, match="message injection"):
            _run(run_rollout(RecordingCompleter([]), TrivialEnv(), hooks=AlwaysInject()))

    def test_clearing_env_stop_continues_rollout(self):
        class ClearFirstCompleted(RecordingHooks):
            def __init__(self) -> None:
                super().__init__()
                self._cleared = False

            async def on_stop_reason(self, reason: str, state: RolloutState) -> str | None:
                await super().on_stop_reason(reason, state)
                if reason == "completed" and not self._cleared:
                    self._cleared = True
                    return None
                return reason

            async def on_turn_begin(self, state: RolloutState) -> list[Message] | None:
                await super().on_turn_begin(state)
                if self._cleared and state.turn_index == 1:
                    return [{"role": "user", "content": "actually, continue"}]
                return None

        renderer = ScriptedRenderer(
            parse_results=[
                (_final_message(), ParseTermination.STOP_SEQUENCE),
                (_final_message(), ParseTermination.STOP_SEQUENCE),
            ]
        )
        reward_fn = RecordingReward()
        env = _build_env(renderer, reward_fn=reward_fn)
        policy = RecordingCompleter([_tokens([1]), _tokens([2])])

        traj = _run(run_rollout(policy, env, hooks=ClearFirstCompleted()))

        assert len(traj.transitions) == 2
        cleared, final = traj.transitions
        assert cleared.episode_done is False  # un-terminated by the hook
        assert not any(k.startswith("stop/") for k in cleared.metrics)
        assert final.episode_done is True
        assert traj.stop_reason == "completed"
        assert reward_fn.calls == 2  # env graded both terminal-looking steps

    def test_custom_string_stop_reason_for_env_stop(self):
        class RenameCompleted(RecordingHooks):
            async def on_stop_reason(self, reason: str, state: RolloutState) -> str | None:
                await super().on_stop_reason(reason, state)
                return "user_sim_done" if reason == "completed" else reason

        renderer = ScriptedRenderer(
            parse_results=[(_final_message(), ParseTermination.STOP_SEQUENCE)]
        )
        env = _build_env(renderer)
        policy = RecordingCompleter([_tokens([1])])

        traj = _run(run_rollout(policy, env, hooks=RenameCompleted()))

        assert traj.stop_reason == "user_sim_done"
        assert traj.transitions[-1].metrics["stop/user_sim_done"] == 1.0
        assert _stop_metric_keys(traj) == ["stop/user_sim_done"]  # exactly one stop metric

    def test_custom_string_stop_reason_for_runner_stop(self):
        class RenameMaxTurns(RecordingHooks):
            async def on_stop_reason(self, reason: str, state: RolloutState) -> str | None:
                await super().on_stop_reason(reason, state)
                return "budget_exhausted" if reason == "max_turns" else reason

        renderer = ScriptedRenderer(
            parse_results=[(_tool_call_message(), ParseTermination.STOP_SEQUENCE)]
        )
        env = _build_env(renderer, max_turns=99)
        policy = RecordingCompleter([_tokens([1])])

        traj = _run(
            run_rollout(policy, env, limits=RolloutLimits(max_turns=1), hooks=RenameMaxTurns())
        )

        assert traj.stop_reason == "budget_exhausted"
        assert _stop_metric_keys(traj) == ["stop/budget_exhausted"]

    def test_hook_call_ordering_and_state(self):
        hooks = RecordingHooks()
        renderer = ScriptedRenderer(
            parse_results=[
                (_tool_call_message(), ParseTermination.STOP_SEQUENCE),
                (_final_message(), ParseTermination.STOP_SEQUENCE),
            ]
        )
        env = _build_env(renderer)
        policy = RecordingCompleter([_tokens([1, 2]), _tokens([3])])

        _run(run_rollout(policy, env, hooks=hooks))

        assert hooks.event_names() == [
            "turn_begin",  # turn 0
            "turn_begin",  # turn 1
            "stop_reason",  # env-reported "completed"
            "grade",
            "artifacts",
        ]
        assert hooks.events[0] == ("turn_begin", 0)
        assert hooks.events[1] == ("turn_begin", 1)
        assert hooks.stop_reasons_seen() == ["completed"]

    def test_runner_stop_skips_on_grade(self):
        hooks = RecordingHooks()
        renderer = ScriptedRenderer(
            parse_results=[(_tool_call_message(), ParseTermination.STOP_SEQUENCE)]
        )
        env = _build_env(renderer, max_turns=99)
        policy = RecordingCompleter([_tokens([1])])

        _run(run_rollout(policy, env, limits=RolloutLimits(max_turns=1), hooks=hooks))

        names = hooks.event_names()
        assert "grade" not in names  # runner stop: env never graded
        assert names[-1] == "artifacts"


# ---------------------------------------------------------------------------
# Parse-error policy through the runner
# ---------------------------------------------------------------------------


def _content_error_message(error: str = "Invalid JSON: boom") -> Message:
    """A cleanly terminated turn whose only tool call is malformed."""
    return {
        "role": "assistant",
        "content": "calling tools",
        "unparsed_tool_calls": [
            UnparsedToolCall(raw_text="<tool_call>{bad}</tool_call>", error=error)
        ],
    }


class TestParseErrorPolicyThroughRunner:
    """End-to-end: run_rollout(parse_errors=...) pushes the policy into the
    env; content failures inject a corrective message and continue, structural
    failures stop instantly, and the consecutive budget is terminal."""

    def test_content_error_retry_then_recovery(self):
        """Content error -> injected corrective message (with the error detail)
        -> model recovers -> normal graded completion."""
        renderer = ScriptedRenderer(
            parse_results=[
                (_content_error_message("Invalid JSON: boom"), ParseTermination.STOP_SEQUENCE),
                (_final_message(), ParseTermination.STOP_SEQUENCE),
            ]
        )
        reward_fn = RecordingReward(reward=0.9)
        env = _build_env(renderer, reward_fn=reward_fn)
        policy = RecordingCompleter([_tokens([1, 2]), _tokens([3, 4])])

        traj = _run(run_rollout(policy, env, parse_errors=ParseErrorPolicy(max_consecutive=1)))

        assert traj.stop_reason == "completed"
        assert len(traj.transitions) == 2
        assert traj.transitions[0].episode_done is False
        assert traj.transitions[1].episode_done is True
        assert traj.transitions[1].reward == 0.9  # recovery graded normally
        assert reward_fn.calls == 1
        # The corrective message reached the conversation with the detail.
        history = env.message_env.history  # type: ignore[attr-defined]
        injected = [
            m for m in history if m["role"] == "user" and "formatting issue" in str(m["content"])
        ]
        assert len(injected) == 1
        assert "Invalid JSON: boom" in str(injected[0]["content"])
        # Two sampling calls: the retry consumed a policy call.
        assert policy.calls == 2

    def test_injected_retry_tokens_count_toward_trajectory_budget(self):
        """The injected corrective message grows the next observation, so the
        trajectory budget sees it."""
        renderer = ScriptedRenderer(
            parse_results=[
                (_content_error_message(), ParseTermination.STOP_SEQUENCE),
                (_final_message(), ParseTermination.STOP_SEQUENCE),
            ],
            tokens_per_message=10,
        )
        env = _build_env(renderer)
        policy = RecordingCompleter([_tokens([1, 2]), _tokens([3, 4])])

        traj = _run(
            run_rollout(
                policy,
                env,
                limits=RolloutLimits(max_trajectory_tokens=1000),
                parse_errors=ParseErrorPolicy(max_consecutive=1),
            )
        )

        assert traj.stop_reason == "completed"
        # obs 1: initial user message (10 tokens). obs 2 re-renders the grown
        # conversation: user + error assistant turn + injected user message.
        first_ob = traj.transitions[0].ob.length
        second_ob = traj.transitions[1].ob.length
        assert second_ob >= first_ob + 2 * renderer.tokens_per_message
        # And the per-call budget the completer saw shrank accordingly.
        assert policy.max_tokens_seen[1] == 1000 - second_ob

    def test_consecutive_budget_exceeded_is_terminal(self):
        renderer = ScriptedRenderer(
            parse_results=[
                (_content_error_message("err one"), ParseTermination.STOP_SEQUENCE),
                (_content_error_message("err two"), ParseTermination.STOP_SEQUENCE),
            ]
        )
        reward_fn = RecordingReward()
        env = _build_env(renderer, reward_fn=reward_fn)
        policy = RecordingCompleter([_tokens([1]), _tokens([2])])

        traj = _run(
            run_rollout(
                policy,
                env,
                parse_errors=ParseErrorPolicy(max_consecutive=1, terminal_reward=-0.5),
            )
        )

        assert traj.stop_reason == "parse_error"
        assert len(traj.transitions) == 2
        assert traj.transitions[-1].episode_done is True
        assert traj.transitions[-1].reward == -0.5
        assert traj.transitions[-1].metrics["stop/parse_error"] == 1.0
        assert reward_fn.calls == 0

    def test_penalty_applied_to_retried_turn_reward(self):
        renderer = ScriptedRenderer(
            parse_results=[
                (_content_error_message(), ParseTermination.STOP_SEQUENCE),
                (_final_message(), ParseTermination.STOP_SEQUENCE),
            ]
        )
        env = _build_env(renderer)
        policy = RecordingCompleter([_tokens([1]), _tokens([2])])

        traj = _run(
            run_rollout(
                policy,
                env,
                parse_errors=ParseErrorPolicy(max_consecutive=1, penalty_per_error=0.25),
            )
        )

        assert traj.transitions[0].reward == -0.25
        assert traj.stop_reason == "completed"

    def test_structural_failure_instant_stop_no_retry(self):
        """MALFORMED termination (broken framing) stops immediately with
        StopReason.PARSE_ERROR even when the content budget would allow a
        retry; the env's reward_fn never runs."""
        renderer = ScriptedRenderer(
            parse_results=[
                ({"role": "assistant", "content": "trunca"}, ParseTermination.MALFORMED),
            ]
        )
        reward_fn = RecordingReward()
        env = _build_env(renderer, reward_fn=reward_fn)
        policy = RecordingCompleter([_tokens([1])])

        traj = _run(
            run_rollout(
                policy,
                env,
                parse_errors=ParseErrorPolicy(max_consecutive=5, terminal_reward=-1.0),
            )
        )

        assert traj.stop_reason == "parse_error"
        assert len(traj.transitions) == 1
        assert traj.transitions[0].episode_done is True
        assert traj.transitions[0].reward == -1.0
        assert traj.transitions[0].logs["parse_failure_kind"] == "structural"
        assert "parse_failure_detail" in traj.transitions[0].logs
        assert reward_fn.calls == 0
        assert policy.calls == 1  # no retry sample

    def test_mask_error_turns_marks_transitions(self):
        renderer = ScriptedRenderer(
            parse_results=[
                (_content_error_message(), ParseTermination.STOP_SEQUENCE),
                (_final_message(), ParseTermination.STOP_SEQUENCE),
            ]
        )
        env = _build_env(renderer)
        policy = RecordingCompleter([_tokens([1]), _tokens([2])])

        traj = _run(
            run_rollout(
                policy,
                env,
                parse_errors=ParseErrorPolicy(max_consecutive=1, mask_error_turns=True),
            )
        )

        assert traj.transitions[0].metrics[PARSE_ERROR_MASKED_METRIC_KEY] == 1.0
        assert PARSE_ERROR_MASKED_METRIC_KEY not in traj.transitions[1].metrics

    def test_env_without_seam_warns_and_keeps_legacy(self, caplog: pytest.LogCaptureFixture):
        """An env that lacks set_parse_error_policy logs a warning and keeps
        the default one-shot semantics."""

        class PlainEnv(Env):
            async def initial_observation(self) -> tuple[tinker.ModelInput, list[str]]:
                return tinker.ModelInput.from_ints([1, 2, 3]), ["<stop>"]

            async def step(self, action: list[int], *, extra: Any = None) -> StepResult:
                return StepResult(
                    reward=0.3,
                    episode_done=True,
                    next_observation=tinker.ModelInput.empty(),
                    next_stop_condition=["<stop>"],
                )

        policy = RecordingCompleter([_tokens([1])])
        with caplog.at_level("WARNING"):
            traj = _run(
                run_rollout(policy, PlainEnv(), parse_errors=ParseErrorPolicy(max_consecutive=1))
            )

        assert any("set_parse_error_policy" in r.message for r in caplog.records)
        assert traj.transitions[-1].reward == 0.3

    def test_policy_none_keeps_legacy_one_shot_semantics(self):
        """No policy: a content parse failure keeps the default one-shot
        behavior (failed_parse_reward, terminate, reward_fn skipped)."""
        renderer = ScriptedRenderer(
            parse_results=[
                (_content_error_message(), ParseTermination.STOP_SEQUENCE),
            ]
        )
        reward_fn = RecordingReward()
        env = _build_env(renderer, reward_fn=reward_fn)
        policy = RecordingCompleter([_tokens([1])])

        traj = _run(run_rollout(policy, env))

        assert traj.stop_reason == "parse_error"
        assert traj.transitions[-1].reward == -0.1  # default failed_parse_reward
        assert reward_fn.calls == 0
