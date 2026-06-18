"""Tests for AgentToolMessageEnv log population."""

import asyncio
from typing import Any

from PIL import Image

from tinker_cookbook.renderers.base import ImagePart, Message, TextPart, ToolCall, ToolSpec
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
        """When there are no tool calls, only assistant_content and conversation_history are logged."""
        env = AgentToolMessageEnv(
            tools=[],
            initial_messages=[{"role": "user", "content": "hi"}],
            max_turns=5,
            reward_fn=_noop_reward,
        )
        asyncio.run(env.initial_observation())

        result = asyncio.run(env.step({"role": "assistant", "content": "Just text."}))

        assert result.logs["assistant_content"] == "Just text."
        assert "conversation_history" in result.logs
        assert "tool_call_0" not in result.logs
        assert "tool_result_0" not in result.logs


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
