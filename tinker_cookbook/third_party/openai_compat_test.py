"""Tests for OpenAI format compatibility utilities."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from tinker_cookbook.renderers.base import Message, ToolCall, ToolSpec, UnparsedToolCall
from tinker_cookbook.third_party.openai_compat import (
    openai_messages_to_tinker,
    openai_tools_to_tinker,
    tinker_messages_to_openai,
    tool_specs_to_openai_tools,
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

    def test_reasoning_with_string_content(self) -> None:
        messages = [{"role": "assistant", "content": "4", "reasoning_content": "2+2=4"}]
        result = openai_messages_to_tinker(messages)
        assert result[0]["content"] == [
            {"type": "thinking", "thinking": "2+2=4"},
            {"type": "text", "text": "4"},
        ]

    @pytest.mark.parametrize("content", ["", None])
    def test_reasoning_with_empty_content(self, content: str | None) -> None:
        messages = [{"role": "assistant", "content": content, "reasoning_content": "hmm"}]
        result = openai_messages_to_tinker(messages)
        assert result[0]["content"] == [{"type": "thinking", "thinking": "hmm"}]

    def test_reasoning_with_list_content(self) -> None:
        image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "a"}, image_part],
                "reasoning_content": "r",
            }
        ]
        result = openai_messages_to_tinker(messages)
        assert result[0]["content"] == [
            {"type": "thinking", "thinking": "r"},
            {"type": "text", "text": "a"},
            {"type": "image", "image": "data:image/png;base64,AAAA"},
        ]

    def test_openai_media_parts_become_cookbook_parts(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "image_url", "image_url": "https://example.com/cat.png"},
                    {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "WAV"}},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "AAAA",
                            "format": "mp3",
                            "num_frames": 9,
                            "sample_rate": 8000,
                        },
                    },
                ],
            }
        ]
        assert openai_messages_to_tinker(messages)[0]["content"] == [
            {"type": "image", "image": "data:image/png;base64,AAAA"},
            {"type": "image", "image": "https://example.com/cat.png"},
            {"type": "audio", "audio": b"RIFF", "format": "wav"},
            {
                "type": "audio",
                "audio": b"\x00\x00\x00",
                "format": "mp3",
                "num_frames": 9,
                "sample_rate": 8000,
            },
        ]

    @pytest.mark.parametrize("role", ["assistant", "user"])
    @pytest.mark.parametrize("reasoning", ["", None])
    def test_empty_reasoning_is_no_reasoning(self, role: str, reasoning: str | None) -> None:
        messages = [{"role": role, "content": "hi", "reasoning_content": reasoning}]
        result = openai_messages_to_tinker(messages)
        assert result[0]["content"] == "hi"

    def test_reasoning_on_non_assistant_raises(self) -> None:
        messages = [{"role": "user", "content": "hi", "reasoning_content": "r"}]
        with pytest.raises(ValueError, match="only valid on assistant"):
            openai_messages_to_tinker(messages)

    def test_non_string_reasoning_raises(self) -> None:
        messages = [{"role": "assistant", "content": "hi", "reasoning_content": ["r"]}]
        with pytest.raises(ValueError, match="must be a string"):
            openai_messages_to_tinker(messages)


# ---------------------------------------------------------------------------
# tinker_messages_to_openai
# ---------------------------------------------------------------------------


def _tool_call(call_id: str | None = "call_1") -> ToolCall:
    return ToolCall(
        id=call_id,
        function=ToolCall.FunctionBody(name="search", arguments='{"q": "test"}'),
    )


_FN_A: dict[str, Any] = {"type": "function", "function": {"name": "f", "description": "A"}}
_FN_B: dict[str, Any] = {"type": "function", "function": {"name": "f", "description": "B"}}


class TestTinkerMessagesToOpenAI:
    def test_string_content_and_tool_fields(self) -> None:
        messages: list[Message] = [
            {"role": "tool", "content": "result", "tool_call_id": "call_1", "name": "search"},
        ]
        assert tinker_messages_to_openai(messages) == {
            "messages": [
                {"role": "tool", "content": "result", "tool_call_id": "call_1", "name": "search"}
            ]
        }

    def test_thinking_becomes_reasoning_content(self) -> None:
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "first"},
                    {"type": "thinking", "thinking": "second"},
                    {"type": "text", "text": "answer"},
                ],
            }
        ]
        assert tinker_messages_to_openai(messages) == {
            "messages": [
                {"role": "assistant", "content": "answer", "reasoning_content": "first\nsecond"}
            ]
        }

    def test_media_parts_keep_list_content(self) -> None:
        image_uri = "data:image/png;base64,AAAA"
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image", "image": image_uri},
                    {"type": "audio", "audio": b"RIFF", "format": "wav"},
                ],
            }
        ]
        result = tinker_messages_to_openai(messages)["messages"]
        assert result[0]["content"] == [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": image_uri}},
            {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
        ]

    @pytest.mark.parametrize(
        ("tool_calls", "expected_content"), [([_tool_call()], None), (None, "")]
    )
    def test_no_parts_after_thinking(
        self, tool_calls: list[ToolCall] | None, expected_content: str | None
    ) -> None:
        message: Message = {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "r"}],
        }
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
        result = tinker_messages_to_openai([message])["messages"]
        assert result[0]["content"] == expected_content
        assert result[0]["reasoning_content"] == "r"

    def test_tool_calls_dump_without_none_and_dicts_pass_through(self) -> None:
        raw_call: dict[str, Any] = {
            "type": "function",
            "id": "call_2",
            "function": {"name": "f", "arguments": "{}"},
        }
        # Callers pass plain OpenAI dicts in tool_calls too.
        tool_calls = cast(list[ToolCall], [_tool_call(None), raw_call])
        message: Message = {"role": "assistant", "content": "", "tool_calls": tool_calls}
        result = tinker_messages_to_openai([message])["messages"]
        assert result[0]["tool_calls"] == [
            {"type": "function", "function": {"name": "search", "arguments": '{"q": "test"}'}},
            raw_call,
        ]

    def test_trainable_is_dropped(self) -> None:
        messages: list[Message] = [{"role": "assistant", "content": "a", "trainable": False}]
        assert tinker_messages_to_openai(messages) == {
            "messages": [{"role": "assistant", "content": "a"}]
        }

    @pytest.mark.parametrize(
        ("messages", "match"),
        [
            pytest.param(
                [{"role": "user", "content": [{"type": "thinking", "thinking": "r"}]}],
                "only valid on assistant",
                id="thinking-on-user",
            ),
            pytest.param(
                [{"role": "tool_declare", "content": "namespace functions {}"}],
                "JSON list of OpenAI tools",
                id="typescript-declaration",
            ),
            pytest.param(
                [{"role": "tool_declare", "content": json.dumps([1])}],
                "JSON list of OpenAI tools",
                id="non-dict-tool",
            ),
            pytest.param(
                [{"role": "tool_declare", "content": json.dumps([{"function": None}])}],
                "JSON list of OpenAI tools",
                id="null-function",
            ),
            pytest.param(
                [{"role": "tool_declare", "content": json.dumps([{}])}],
                "JSON list of OpenAI tools",
                id="missing-function",
            ),
            pytest.param(
                [{"role": "tool_declare", "content": json.dumps([{"function": {}}])}],
                "name must be a string",
                id="missing-name",
            ),
            pytest.param(
                [
                    {
                        "role": "tool_declare",
                        "content": json.dumps([{"type": "function", "function": {"name": ["f"]}}]),
                    }
                ],
                "name must be a string",
                id="list-name",
            ),
            pytest.param(
                [
                    {"role": "tool_declare", "content": json.dumps([_FN_A])},
                    {"role": "tool_declare", "content": json.dumps([_FN_B])},
                ],
                "declared twice with different specs",
                id="conflict-across-messages",
            ),
            pytest.param(
                [{"role": "tool_declare", "content": json.dumps([_FN_A, _FN_B])}],
                "declared twice with different specs",
                id="conflict-in-one-message",
            ),
        ],
    )
    def test_messages_without_openai_form_raise(self, messages: list[Message], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            tinker_messages_to_openai(messages)

    def test_round_trip(self) -> None:
        messages: list[Message] = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Search for test."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I should search."},
                    {"type": "text", "text": "Searching."},
                ],
                "tool_calls": [_tool_call()],
            },
            {"role": "tool", "content": "found", "tool_call_id": "call_1", "name": "search"},
        ]
        assert (
            openai_messages_to_tinker(tinker_messages_to_openai(messages)["messages"]) == messages
        )

    def test_tool_declare_messages_become_tools(self) -> None:
        search, fetch = tool_specs_to_openai_tools(
            [
                {"name": "search", "description": "Search.", "parameters": {}},
                {"name": "fetch", "description": "Fetch.", "parameters": {}},
            ]
        )
        messages: list[Message] = [
            {"role": "system", "content": "Be brief."},
            {"role": "tool_declare", "content": json.dumps([search])},
            {"role": "tool_declare", "content": json.dumps([search, fetch])},
            {"role": "user", "content": "Search for test."},
        ]
        assert tinker_messages_to_openai(messages) == {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Search for test."},
            ],
            "tools": [search, fetch],
        }

    def test_remote_image_url_passes_through(self) -> None:
        url = "https://example.com/cat.png"
        messages: list[Message] = [{"role": "user", "content": [{"type": "image", "image": url}]}]
        assert tinker_messages_to_openai(messages)["messages"][0]["content"] == [
            {"type": "image_url", "image_url": {"url": url}}
        ]

    def test_unknown_part_type_raises(self) -> None:
        messages = cast(list[Message], [{"role": "user", "content": [{"type": "video"}]}])
        with pytest.raises(ValueError, match="unsupported content part type"):
            tinker_messages_to_openai(messages)

    def test_permissive_shapes_convert(self) -> None:
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "thinking", "thinking": "r"},
                ],
                "unparsed_tool_calls": [UnparsedToolCall(raw_text="x", error="bad")],
            }
        ]
        assert tinker_messages_to_openai(messages) == {
            "messages": [{"role": "assistant", "content": "a", "reasoning_content": "r"}]
        }


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


# ---------------------------------------------------------------------------
# tool_specs_to_openai_tools
# ---------------------------------------------------------------------------


class TestToolSpecsToOpenAITools:
    def test_basic_tool(self) -> None:
        tools: list[ToolSpec] = [
            {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ]

        result = tool_specs_to_openai_tools(tools)

        assert result == [
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

    def test_round_trips_function_tools(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search docs",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        assert tool_specs_to_openai_tools(openai_tools_to_tinker(tools)) == tools
