"""Tests for AgentToolMessageEnv log population."""

import asyncio
from typing import Any

from PIL import Image

from tinker_cookbook.renderers.base import (
    PARSE_FAILURE_DETAIL_MAX_CHARS,
    ImagePart,
    Message,
    TextPart,
    ToolCall,
    ToolSpec,
    UnparsedToolCall,
)
from tinker_cookbook.rl import types
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy
from tinker_cookbook.tool_use.agent_tool_message_env import AgentToolMessageEnv
from tinker_cookbook.tool_use.tools import simple_tool_result
from tinker_cookbook.tool_use.types import ToolInput, ToolResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _noop_reward(history: list[Message]) -> tuple[float, dict[str, float]]:
    return 1.0, {}


class StubTool:
    """Minimal Tool implementation for testing."""

    def __init__(self, name: str, response: str, should_stop: bool = False):
        self._name = name
        self._response = response
        self._should_stop = should_stop

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Stub tool: {self._name}"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def run(self, input: ToolInput) -> ToolResult:
        return simple_tool_result(
            self._response,
            call_id=input.call_id or "",
            name=self._name,
            should_stop=self._should_stop,
        )

    def to_spec(self) -> ToolSpec:
        return {
            "name": self._name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }


def _make_tool_call(name: str, arguments: str = "{}", call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, function=ToolCall.FunctionBody(name=name, arguments=arguments))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStepLogs:
    """AgentToolMessageEnv.step() should populate logs with diagnostic info."""

    def test_logs_assistant_content(self):
        """Logs include assistant_content when message has text content."""
        env = AgentToolMessageEnv(
            tools=[],
            initial_messages=[{"role": "user", "content": "hi"}],
            max_turns=5,
            reward_fn=_noop_reward,
        )
        asyncio.run(env.initial_observation())

        result = asyncio.run(env.step({"role": "assistant", "content": "Hello world"}))

        assert result.logs["assistant_content"] == "Hello world"

    def test_logs_empty_when_no_content(self):
        """Logs omit assistant_content when message has empty content."""
        env = AgentToolMessageEnv(
            tools=[],
            initial_messages=[{"role": "user", "content": "hi"}],
            max_turns=5,
            reward_fn=_noop_reward,
        )
        asyncio.run(env.initial_observation())

        result = asyncio.run(env.step({"role": "assistant", "content": ""}))

        assert "assistant_content" not in result.logs

    def test_logs_multimodal_content(self):
        """Logs extract text from multimodal content via get_text_content."""
        env = AgentToolMessageEnv(
            tools=[],
            initial_messages=[{"role": "user", "content": "hi"}],
            max_turns=5,
            reward_fn=_noop_reward,
        )
        asyncio.run(env.initial_observation())

        result = asyncio.run(
            env.step(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "extracted text"}],
                }
            )
        )

        assert result.logs["assistant_content"] == "extracted text"

    def test_logs_tool_calls_and_results(self):
        """Logs include tool call names/args and tool result content."""
        search_tool = StubTool("search", '{"results": ["a", "b"]}')
        env = AgentToolMessageEnv(
            tools=[search_tool],
            initial_messages=[{"role": "user", "content": "find stuff"}],
            max_turns=5,
            reward_fn=_noop_reward,
        )
        asyncio.run(env.initial_observation())

        tc = _make_tool_call("search", '{"query": "weather"}')
        result = asyncio.run(
            env.step({"role": "assistant", "content": "Let me search.", "tool_calls": [tc]})
        )

        assert result.logs["assistant_content"] == "Let me search."
        assert result.logs["tool_call_0"] == 'search({"query": "weather"})'
        assert result.logs["tool_result_0"] == '{"results": ["a", "b"]}'

    def test_logs_multiple_tool_calls(self):
        """Logs index multiple tool calls and results separately."""
        search_tool = StubTool("search", "search result")
        calc_tool = StubTool("calc", "42")
        env = AgentToolMessageEnv(
            tools=[search_tool, calc_tool],
            initial_messages=[{"role": "user", "content": "hi"}],
            max_turns=5,
            reward_fn=_noop_reward,
        )
        asyncio.run(env.initial_observation())

        tc1 = _make_tool_call("search", '{"q": "x"}', call_id="call_1")
        tc2 = _make_tool_call("calc", '{"expr": "1+1"}', call_id="call_2")
        result = asyncio.run(
            env.step({"role": "assistant", "content": "Doing both.", "tool_calls": [tc1, tc2]})
        )

        assert result.logs["tool_call_0"] == 'search({"q": "x"})'
        assert result.logs["tool_call_1"] == 'calc({"expr": "1+1"})'
        assert result.logs["tool_result_0"] == "search result"
        assert result.logs["tool_result_1"] == "42"

    def test_logs_no_tool_calls(self):
        """When there are no tool calls, only assistant_content is logged."""
        env = AgentToolMessageEnv(
            tools=[],
            initial_messages=[{"role": "user", "content": "hi"}],
            max_turns=5,
            reward_fn=_noop_reward,
        )
        asyncio.run(env.initial_observation())

        result = asyncio.run(env.step({"role": "assistant", "content": "Just text."}))

        assert result.logs == {"assistant_content": "Just text."}
        assert "tool_call_0" not in result.logs
        assert "tool_result_0" not in result.logs


# ---------------------------------------------------------------------------
# Stop-reason metrics
# ---------------------------------------------------------------------------


class TestStopReasonMetrics:
    """Episode-ending steps emit a one-hot stop/<reason> metric alongside the
    pre-existing metric keys."""

    def test_no_tool_calls_emits_completed(self):
        env = AgentToolMessageEnv(
            tools=[],
            initial_messages=[{"role": "user", "content": "hi"}],
            max_turns=5,
            reward_fn=_noop_reward,
        )
        asyncio.run(env.initial_observation())

        result = asyncio.run(env.step({"role": "assistant", "content": "final answer"}))

        assert result.episode_done is True
        assert result.metrics["stop/completed"] == 1.0

    def test_tool_stop_emits_tool_stopped(self):
        stop_tool = StubTool("halt", "stopping", should_stop=True)
        env = AgentToolMessageEnv(
            tools=[stop_tool],
            initial_messages=[{"role": "user", "content": "hi"}],
            max_turns=5,
            reward_fn=_noop_reward,
        )
        asyncio.run(env.initial_observation())

        result = asyncio.run(
            env.step({"role": "assistant", "content": "", "tool_calls": [_make_tool_call("halt")]})
        )

        assert result.episode_done is True
        assert result.metrics["tool_stopped"] == 1.0  # pre-existing key kept
        assert result.metrics["stop/tool_stopped"] == 1.0

    def test_max_turns_emits_max_turns(self):
        search_tool = StubTool("search", "result")
        env = AgentToolMessageEnv(
            tools=[search_tool],
            initial_messages=[{"role": "user", "content": "hi"}],
            max_turns=1,
            reward_fn=_noop_reward,
        )
        asyncio.run(env.initial_observation())

        result = asyncio.run(
            env.step(
                {"role": "assistant", "content": "", "tool_calls": [_make_tool_call("search")]}
            )
        )

        assert result.episode_done is True
        assert result.metrics["max_turns"] == 1.0  # pre-existing key kept
        assert result.metrics["stop/max_turns"] == 1.0

    def test_non_terminal_step_has_no_stop_metric(self):
        search_tool = StubTool("search", "result")
        env = AgentToolMessageEnv(
            tools=[search_tool],
            initial_messages=[{"role": "user", "content": "hi"}],
            max_turns=5,
            reward_fn=_noop_reward,
        )
        asyncio.run(env.initial_observation())

        result = asyncio.run(
            env.step(
                {"role": "assistant", "content": "", "tool_calls": [_make_tool_call("search")]}
            )
        )

        assert result.episode_done is False
        assert not any(k.startswith("stop/") for k in result.metrics)


# ---------------------------------------------------------------------------
# Unparsed (malformed) tool calls
# ---------------------------------------------------------------------------


class RecordingReward:
    """reward_fn stub that records how often it was called."""

    def __init__(self, reward: float = 1.0):
        self.calls = 0
        self.reward = reward

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        self.calls += 1
        return self.reward, {"graded": 1.0}


def _make_unparsed(error: str = "Invalid JSON: Expecting value") -> UnparsedToolCall:
    return UnparsedToolCall(raw_text='{"name": "search", bad json}', error=error)


class TestUnparsedToolCalls:
    """A turn whose only tool call is malformed must be treated as a parse
    failure — NOT as a 'no tool calls' completion with normal grading.

    Regression tests for the silent unparsed-tool-call drop: previously the
    env read only message['tool_calls'], so a malformed-JSON-only turn looked
    like 'no tool calls' and ended the episode with the normal reward.
    """

    def _make_env(self, **kwargs: Any) -> tuple[AgentToolMessageEnv, RecordingReward]:
        reward_fn = kwargs.pop("reward_fn", RecordingReward())
        env = AgentToolMessageEnv(
            tools=[StubTool("search", "result")],
            initial_messages=[{"role": "user", "content": "hi"}],
            max_turns=kwargs.pop("max_turns", 5),
            reward_fn=reward_fn,
            **kwargs,
        )
        asyncio.run(env.initial_observation())
        return env, reward_fn

    def test_unparsed_only_turn_is_parse_failure(self):
        """Malformed-JSON-only turn: failed_parse_reward, parse_error metric,
        errors in logs, reward_fn skipped, episode terminated (default)."""
        env, reward_fn = self._make_env()

        result = asyncio.run(
            env.step(
                {
                    "role": "assistant",
                    "content": "calling search",
                    "unparsed_tool_calls": [_make_unparsed("Invalid JSON: Expecting value")],
                }
            )
        )

        assert result.reward == -0.1  # default failed_parse_reward
        assert result.episode_done is True  # terminate_on_parse_error default
        assert result.metrics["parse_error"] == 1.0
        assert result.metrics["stop/parse_error"] == 1.0
        assert result.logs["parse_errors"] == "Invalid JSON: Expecting value"
        assert reward_fn.calls == 0  # grading skipped: this is a parse failure

    def test_unparsed_only_custom_reward(self):
        env, reward_fn = self._make_env(failed_parse_reward=-0.7)

        result = asyncio.run(
            env.step(
                {"role": "assistant", "content": "", "unparsed_tool_calls": [_make_unparsed()]}
            )
        )

        assert result.reward == -0.7
        assert reward_fn.calls == 0

    def test_unparsed_only_no_terminate_continues(self):
        """With terminate_on_parse_error=False the episode continues; the
        penalty applies to the failed turn only."""
        env, reward_fn = self._make_env(terminate_on_parse_error=False)

        result = asyncio.run(
            env.step(
                {"role": "assistant", "content": "", "unparsed_tool_calls": [_make_unparsed()]}
            )
        )

        assert result.episode_done is False
        assert result.reward == -0.1
        assert result.metrics["parse_error"] == 1.0
        assert "stop/parse_error" not in result.metrics  # episode is not over
        assert reward_fn.calls == 0

    def test_unparsed_only_on_final_turn_still_ends(self):
        """Even with terminate_on_parse_error=False, max_turns still ends the
        episode; the parse-failure reward applies and reward_fn stays skipped."""
        env, reward_fn = self._make_env(terminate_on_parse_error=False, max_turns=1)

        result = asyncio.run(
            env.step(
                {"role": "assistant", "content": "", "unparsed_tool_calls": [_make_unparsed()]}
            )
        )

        assert result.episode_done is True
        assert result.reward == -0.1
        assert result.metrics["max_turns"] == 1.0
        assert reward_fn.calls == 0

    def test_multiple_unparsed_errors_joined_in_logs(self):
        env, _ = self._make_env()

        result = asyncio.run(
            env.step(
                {
                    "role": "assistant",
                    "content": "",
                    "unparsed_tool_calls": [_make_unparsed("err one"), _make_unparsed("err two")],
                }
            )
        )

        assert result.logs["parse_errors"] == "err one\nerr two"

    def test_mixed_valid_and_unparsed_executes_valid_and_surfaces_error(self):
        """Mixed valid + malformed tool calls: the valid ones execute and the
        episode proceeds normally with NO failure reward (some tool progress
        happened), but the parse error is surfaced in metrics and logs.

        This mirrors per-message error handling: content errors on individual
        tool calls don't void the turn when other calls succeeded.
        """
        env, reward_fn = self._make_env()

        result = asyncio.run(
            env.step(
                {
                    "role": "assistant",
                    "content": "one good, one bad",
                    "tool_calls": [_make_tool_call("search", '{"q": "x"}')],
                    "unparsed_tool_calls": [_make_unparsed("Invalid JSON: Expecting value")],
                }
            )
        )

        # Valid tool executed and its result is in history/logs.
        assert result.logs["tool_call_0"] == 'search({"q": "x"})'
        assert result.logs["tool_result_0"] == "result"
        assert any(m["role"] == "tool" for m in env.history)
        # Parse error surfaced without the failure reward.
        assert result.metrics["parse_error"] == 1.0
        assert result.logs["parse_errors"] == "Invalid JSON: Expecting value"
        assert result.reward == 0.0  # not failed_parse_reward
        # Episode continues (tool calls present, turns remain).
        assert result.episode_done is False
        assert reward_fn.calls == 0


# ---------------------------------------------------------------------------
# simple_tool_result with images
# ---------------------------------------------------------------------------


class TestSimpleToolResultMultimodal:
    """Unit tests for simple_tool_result() with list[ContentPart] content."""

    def test_with_text_and_images(self):
        img1 = Image.new("RGB", (10, 10), color="red")
        img2 = Image.new("RGB", (10, 10), color="blue")
        parts = [
            TextPart(type="text", text="Page loaded"),
            ImagePart(type="image", image=img1),
            ImagePart(type="image", image=img2),
        ]
        result = simple_tool_result(parts)

        assert len(result.messages) == 1
        msg = result.messages[0]
        assert msg["role"] == "tool"

        content = msg["content"]
        assert isinstance(content, list)
        assert len(content) == 3
        assert content[0] == {"type": "text", "text": "Page loaded"}
        assert content[1] == {"type": "image", "image": img1}
        assert content[2] == {"type": "image", "image": img2}

    def test_string_content(self):
        result = simple_tool_result("just text", call_id="c1", name="t")

        msg = result.messages[0]
        assert isinstance(msg["content"], str)
        assert msg["content"] == "just text"
        assert msg.get("tool_call_id") == "c1"
        assert msg.get("name") == "t"

    def test_interleaved_order_preserved(self):
        img = Image.new("RGB", (10, 10))
        parts = [
            ImagePart(type="image", image=img),
            TextPart(type="text", text="caption"),
            ImagePart(type="image", image="https://example.com/img.png"),
        ]
        result = simple_tool_result(parts)

        content = result.messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 3
        assert content[0] == {"type": "image", "image": img}
        assert content[1] == {"type": "text", "text": "caption"}
        assert content[2] == {"type": "image", "image": "https://example.com/img.png"}

    def test_passthrough_fields(self):
        img = Image.new("RGB", (10, 10))
        result = simple_tool_result(
            [
                TextPart(type="text", text="text"),
                ImagePart(type="image", image=img),
            ],
            call_id="call_123",
            name="screenshot",
            should_stop=True,
            metrics={"latency": 0.5},
            metadata={"source": "browser"},
        )

        assert result.should_stop is True
        assert result.metrics == {"latency": 0.5}
        assert result.metadata == {"source": "browser"}
        assert result.messages[0].get("tool_call_id") == "call_123"
        assert result.messages[0].get("name") == "screenshot"

    def test_defaults_with_list_content(self):
        img = Image.new("RGB", (10, 10))
        result = simple_tool_result(
            [TextPart(type="text", text="text"), ImagePart(type="image", image=img)]
        )

        assert result.should_stop is False
        assert result.metrics == {}
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# Multimodal tool results in AgentToolMessageEnv
# ---------------------------------------------------------------------------


class MultimodalTool:
    """A tool that returns multimodal content (text + image) for testing."""

    @property
    def name(self) -> str:
        return "screenshot"

    @property
    def description(self) -> str:
        return "Take a screenshot"

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
        img = Image.new("RGB", (10, 10), color="red")
        return simple_tool_result(
            [
                TextPart(type="text", text="Screenshot taken"),
                ImagePart(type="image", image=img),
            ],
            call_id=input.call_id or "",
            name=self.name,
        )


class TestMultimodalToolResults:
    """AgentToolMessageEnv should handle tools that return multimodal content."""

    def _make_env(self) -> AgentToolMessageEnv:
        return AgentToolMessageEnv(
            tools=[MultimodalTool()],
            initial_messages=[{"role": "user", "content": "Take a screenshot"}],
            max_turns=5,
            reward_fn=_noop_reward,
        )

    def test_multimodal_tool_result_in_history(self):
        """Tool result with image content is appended to history."""
        env = self._make_env()
        asyncio.run(env.initial_observation())

        msg: Message = {
            "role": "assistant",
            "content": "I'll take a screenshot.",
            "tool_calls": [_make_tool_call("screenshot")],
        }
        asyncio.run(env.step(msg))

        tool_msgs = [m for m in env.history if m["role"] == "tool"]
        assert len(tool_msgs) == 1

        content = tool_msgs[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Screenshot taken"
        assert content[1]["type"] == "image"
        assert isinstance(content[1]["image"], Image.Image)

    def test_multimodal_logging_uses_image_placeholder(self):
        """Logs replace images with <image>...</image> placeholder."""
        env = self._make_env()
        asyncio.run(env.initial_observation())

        msg: Message = {
            "role": "assistant",
            "content": "Taking screenshot now.",
            "tool_calls": [_make_tool_call("screenshot")],
        }
        result = asyncio.run(env.step(msg))

        tool_log = str(result.logs["tool_result_0"])
        assert "<image>Image(10x10, RGB)</image>" in tool_log
        assert "Screenshot taken" in tool_log


class TestParseErrorPolicyContent:
    """Content parse failures under a configured ParseErrorPolicy: retry
    injection with the formatted detail, penalty on retried turns, terminal
    stop with terminal_reward once the consecutive budget is exceeded, counter
    reset on clean turns, and the mask marker."""

    def _make_env(
        self, policy: ParseErrorPolicy, **kwargs: Any
    ) -> tuple[AgentToolMessageEnv, RecordingReward]:
        reward_fn = kwargs.pop("reward_fn", RecordingReward())
        env = AgentToolMessageEnv(
            tools=[StubTool("search", "result")],
            initial_messages=[{"role": "user", "content": "hi"}],
            max_turns=kwargs.pop("max_turns", 10),
            reward_fn=reward_fn,
            parse_error_policy=policy,
            **kwargs,
        )
        asyncio.run(env.initial_observation())
        return env, reward_fn

    def _error_message(self, error: str = "Invalid JSON: Expecting value") -> Message:
        return {
            "role": "assistant",
            "content": "calling",
            "unparsed_tool_calls": [_make_unparsed(error)],
        }

    def test_retry_injects_corrective_user_message(self):
        policy = ParseErrorPolicy(max_consecutive=1)
        env, reward_fn = self._make_env(policy)

        result = asyncio.run(env.step(self._error_message("Invalid JSON: boom")))

        assert result.episode_done is False
        assert result.reward == 0.0  # penalty_per_error defaults to 0
        assert result.metrics["parse_error"] == 1.0
        assert "stop/parse_error" not in result.metrics
        assert result.logs["parse_failure_kind"] == "content"
        assert reward_fn.calls == 0
        # The corrective message is the next user message and carries the detail.
        injected = env.history[-1]
        assert injected["role"] == "user"
        assert isinstance(injected["content"], str)
        assert "formatting issue" in injected["content"]
        assert "Invalid JSON: boom" in injected["content"]

    def test_penalty_applied_to_retried_turn(self):
        policy = ParseErrorPolicy(max_consecutive=1, penalty_per_error=0.25)
        env, _ = self._make_env(policy)

        result = asyncio.run(env.step(self._error_message()))

        assert result.episode_done is False
        assert result.reward == -0.25

    def test_budget_exceeded_terminal_stop(self):
        """max_consecutive=1: the second consecutive error is terminal with
        terminal_reward, stop/parse_error, and reward_fn still skipped."""
        policy = ParseErrorPolicy(max_consecutive=1, terminal_reward=-0.5)
        env, reward_fn = self._make_env(policy)

        first = asyncio.run(env.step(self._error_message("err one")))
        assert first.episode_done is False

        second = asyncio.run(env.step(self._error_message("err two")))
        assert second.episode_done is True
        assert second.reward == -0.5
        assert second.metrics["stop/parse_error"] == 1.0
        assert reward_fn.calls == 0

    def test_max_consecutive_zero_stops_on_first_error(self):
        policy = ParseErrorPolicy(max_consecutive=0, terminal_reward=0.0)
        env, reward_fn = self._make_env(policy)

        result = asyncio.run(env.step(self._error_message()))

        assert result.episode_done is True
        assert result.reward == 0.0
        assert result.metrics["stop/parse_error"] == 1.0
        assert reward_fn.calls == 0

    def test_clean_turn_resets_consecutive_counter(self):
        """error -> valid tool turn -> error again is retried (not terminal)."""
        policy = ParseErrorPolicy(max_consecutive=1)
        env, _ = self._make_env(policy)

        first = asyncio.run(env.step(self._error_message()))
        assert first.episode_done is False

        tool_turn = asyncio.run(
            env.step(
                {
                    "role": "assistant",
                    "content": "trying properly",
                    "tool_calls": [_make_tool_call("search")],
                }
            )
        )
        assert tool_turn.episode_done is False

        third = asyncio.run(env.step(self._error_message()))
        assert third.episode_done is False  # counter reset: this is error 1 of 1

    def test_max_turns_wins_over_retry(self):
        policy = ParseErrorPolicy(max_consecutive=3, terminal_reward=-0.2)
        env, reward_fn = self._make_env(policy, max_turns=1)

        result = asyncio.run(env.step(self._error_message()))

        assert result.episode_done is True
        assert result.reward == -0.2
        assert result.metrics["max_turns"] == 1.0
        assert result.metrics["stop/parse_error"] == 1.0
        assert reward_fn.calls == 0

    def test_mask_error_turns_marks_retried_and_terminal_turns(self):
        policy = ParseErrorPolicy(max_consecutive=1, mask_error_turns=True)
        env, _ = self._make_env(policy)

        first = asyncio.run(env.step(self._error_message()))
        assert first.metrics[types.PARSE_ERROR_MASKED_METRIC_KEY] == 1.0

        second = asyncio.run(env.step(self._error_message()))
        assert second.episode_done is True
        assert second.metrics[types.PARSE_ERROR_MASKED_METRIC_KEY] == 1.0

    def test_mask_disabled_by_default(self):
        policy = ParseErrorPolicy(max_consecutive=1)
        env, _ = self._make_env(policy)

        result = asyncio.run(env.step(self._error_message()))
        assert types.PARSE_ERROR_MASKED_METRIC_KEY not in result.metrics

    def test_detail_truncated_in_injected_message(self):
        policy = ParseErrorPolicy(max_consecutive=1)
        env, _ = self._make_env(policy)

        long_error = "x" * (2 * PARSE_FAILURE_DETAIL_MAX_CHARS)
        asyncio.run(env.step(self._error_message(long_error)))

        injected = env.history[-1]
        assert isinstance(injected["content"], str)
        assert len(injected["content"]) < PARSE_FAILURE_DETAIL_MAX_CHARS + 200

    def test_mixed_valid_and_unparsed_not_retried(self):
        """Stage-1 semantics stand under a policy: mixed calls proceed normally."""
        policy = ParseErrorPolicy(max_consecutive=2, penalty_per_error=0.5)
        env, _ = self._make_env(policy)

        result = asyncio.run(
            env.step(
                {
                    "role": "assistant",
                    "content": "mixed",
                    "tool_calls": [_make_tool_call("search")],
                    "unparsed_tool_calls": [_make_unparsed()],
                }
            )
        )

        assert result.episode_done is False
        assert result.reward == 0.0  # no penalty: the turn made tool progress
        assert result.metrics["parse_error"] == 1.0  # error still surfaced
        # No corrective message was injected; the last history entries are tool results.
        assert env.history[-1]["role"] == "tool"
