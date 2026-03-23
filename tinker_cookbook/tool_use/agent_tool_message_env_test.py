"""Tests for AgentToolMessageEnv log population."""

import asyncio
from typing import Any

from tinker_cookbook.renderers.base import Message, ToolCall, ToolSpec
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
