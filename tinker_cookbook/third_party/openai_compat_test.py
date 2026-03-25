"""Tests for OpenAI format compatibility utilities."""

from __future__ import annotations

from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.third_party.openai_compat import (
    openai_messages_to_tinker,
    openai_tools_to_tinker,
)

# ---------------------------------------------------------------------------
# openai_messages_to_tinker
# ---------------------------------------------------------------------------


class TestOpenAIMessagesToTinker:
    def test_basic_messages(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = openai_messages_to_tinker(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."
        assert result[1]["role"] == "user"

    def test_message_with_tool_call_id(self) -> None:
        messages = [
            {"role": "tool", "content": "result", "tool_call_id": "call_123"},
        ]
        result = openai_messages_to_tinker(messages)
        assert result[0].get("tool_call_id") == "call_123"

    def test_message_with_name(self) -> None:
        messages = [{"role": "user", "content": "hi", "name": "Alice"}]
        result = openai_messages_to_tinker(messages)
        assert result[0].get("name") == "Alice"

    def test_message_with_tool_calls(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {"name": "search", "arguments": '{"q": "test"}'},
                    }
                ],
            }
        ]
        result = openai_messages_to_tinker(messages)
        tcs = result[0].get("tool_calls")
        assert tcs is not None
        assert len(tcs) == 1
        assert isinstance(tcs[0], ToolCall)
        assert tcs[0].function.name == "search"

    def test_none_content_becomes_empty_string(self) -> None:
        messages = [{"role": "assistant", "content": None}]
        result = openai_messages_to_tinker(messages)
        assert result[0]["content"] == ""


# ---------------------------------------------------------------------------
# openai_tools_to_tinker
# ---------------------------------------------------------------------------


class TestOpenAIToolsToTinker:
    def test_basic_tool(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        result = openai_tools_to_tinker(tools)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get weather for a city"
        assert "properties" in result[0]["parameters"]

    def test_skips_non_function_tools(self) -> None:
        tools = [
            {"type": "retrieval"},
            {
                "type": "function",
                "function": {"name": "search", "description": "Search", "parameters": {}},
            },
        ]
        result = openai_tools_to_tinker(tools)
        assert len(result) == 1
        assert result[0]["name"] == "search"

    def test_missing_description(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {"name": "noop", "parameters": {}},
            }
        ]
        result = openai_tools_to_tinker(tools)
        assert result[0]["description"] == ""

    def test_empty_tools(self) -> None:
        assert openai_tools_to_tinker([]) == []
