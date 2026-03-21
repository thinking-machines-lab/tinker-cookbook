"""Tests for OpenAI format compatibility utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.third_party.openai_compat import (
    SamplingResult,
    extract_sampling_params,
    openai_messages_to_tinker,
    openai_tools_to_tinker,
    prepare_messages_with_tools,
    sample_chat_completion,
    sampling_result_to_openai_dict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeSampledSequence:
    tokens: list[int]
    logprobs: list[float] | None
    stop_reason: str = "stop"


@dataclass
class FakeSampleResponse:
    sequences: list[FakeSampledSequence]


def _make_sampling_result(
    *,
    prompt_tokens: list[int] | None = None,
    completion_tokens: list[int] | None = None,
    content: str = "Hello!",
    parse_success: bool = True,
    tool_calls: list[ToolCall] | None = None,
) -> SamplingResult:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return SamplingResult(
        prompt_token_ids=prompt_tokens or [1, 2, 3],
        completion_token_ids=completion_tokens or [4, 5, 6],
        logprobs=[0.1, 0.2, 0.3],
        parsed_message=msg,  # type: ignore[arg-type]
        parse_success=parse_success,
        model_name="tinker/test-model",
    )


# ---------------------------------------------------------------------------
# openai_messages_to_tinker
# ---------------------------------------------------------------------------


class TestOpenaiMessagesToTinker:
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


class TestOpenaiToolsToTinker:
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
# prepare_messages_with_tools
# ---------------------------------------------------------------------------


class TestPrepareMessagesWithTools:
    def test_extracts_system_message(self) -> None:
        renderer = MagicMock()
        renderer.create_conversation_prefix_with_tools.return_value = [
            {"role": "system", "content": "You have tools: [search]. Also: Be helpful."}
        ]

        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        tools = [
            {
                "type": "function",
                "function": {"name": "search", "description": "Search", "parameters": {}},
            }
        ]

        result = prepare_messages_with_tools(renderer, messages, tools)  # type: ignore[arg-type]

        renderer.create_conversation_prefix_with_tools.assert_called_once()
        args = renderer.create_conversation_prefix_with_tools.call_args
        assert args[0][1] == "Be helpful."  # system_prompt extracted
        # User message comes after the prefix
        assert result[-1]["role"] == "user"

    def test_no_system_message(self) -> None:
        renderer = MagicMock()
        renderer.create_conversation_prefix_with_tools.return_value = [
            {"role": "system", "content": "Tools: [search]"}
        ]

        messages = [{"role": "user", "content": "Hi"}]
        tools = [
            {
                "type": "function",
                "function": {"name": "search", "description": "Search", "parameters": {}},
            }
        ]

        prepare_messages_with_tools(renderer, messages, tools)  # type: ignore[arg-type]

        args = renderer.create_conversation_prefix_with_tools.call_args
        assert args[0][1] == ""  # no system prompt


# ---------------------------------------------------------------------------
# sampling_result_to_openai_dict
# ---------------------------------------------------------------------------


class TestSamplingResultToOpenaiDict:
    def test_basic_response(self) -> None:
        result = _make_sampling_result(content="Hi there!")
        d = sampling_result_to_openai_dict(result)

        assert d["object"] == "chat.completion"
        assert d["model"] == "tinker/test-model"
        assert len(d["choices"]) == 1
        assert d["choices"][0]["message"]["content"] == "Hi there!"
        assert d["choices"][0]["message"]["role"] == "assistant"
        assert d["choices"][0]["finish_reason"] == "stop"
        assert d["usage"]["prompt_tokens"] == 3
        assert d["usage"]["completion_tokens"] == 3

    def test_parse_failure_gives_length_finish(self) -> None:
        result = _make_sampling_result(parse_success=False)
        d = sampling_result_to_openai_dict(result)
        assert d["choices"][0]["finish_reason"] == "length"

    def test_tool_calls_in_response(self) -> None:
        tc = ToolCall(
            function=ToolCall.FunctionBody(name="search", arguments='{"q": "test"}'),
            id="call_abc",
        )
        result = _make_sampling_result(tool_calls=[tc])
        d = sampling_result_to_openai_dict(result)

        assert d["choices"][0]["finish_reason"] == "tool_calls"
        tool_calls = d["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "search"
        assert tool_calls[0]["id"] == "call_abc"

    def test_tool_call_without_id_gets_generated(self) -> None:
        tc = ToolCall(
            function=ToolCall.FunctionBody(name="search", arguments="{}"),
            id=None,
        )
        result = _make_sampling_result(tool_calls=[tc])
        d = sampling_result_to_openai_dict(result)
        assert d["choices"][0]["message"]["tool_calls"][0]["id"] == "call_0"

    def test_list_content_formatted_as_string(self) -> None:
        result = _make_sampling_result()
        result.parsed_message["content"] = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world!"},
        ]
        d = sampling_result_to_openai_dict(result)
        assert d["choices"][0]["message"]["content"] == "Hello \nworld!"


# ---------------------------------------------------------------------------
# sample_chat_completion
# ---------------------------------------------------------------------------


class TestSampleChatCompletion:
    @pytest.mark.asyncio
    async def test_basic_flow(self) -> None:
        fake_response = FakeSampleResponse(
            sequences=[FakeSampledSequence(tokens=[10, 20, 30], logprobs=[0.1, 0.2, 0.3])]
        )
        sampling_client = MagicMock()
        sampling_client.sample_async = AsyncMock(return_value=fake_response)

        renderer = MagicMock()
        renderer.build_generation_prompt.return_value = MagicMock()
        renderer.build_generation_prompt.return_value.to_ints.return_value = [1, 2, 3]
        renderer.get_stop_sequences.return_value = ["<|endoftext|>"]
        renderer.parse_response.return_value = (
            {"role": "assistant", "content": "response"},
            True,
        )

        result = await sample_chat_completion(
            sampling_client=sampling_client,
            renderer=renderer,
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
            max_tokens=64,
        )

        assert result.prompt_token_ids == [1, 2, 3]
        assert result.completion_token_ids == [10, 20, 30]
        assert result.parse_success is True
        assert result.parsed_message["content"] == "response"

        # Verify sampling params were passed correctly
        call_kwargs = sampling_client.sample_async.call_args.kwargs
        assert call_kwargs["sampling_params"].temperature == 0.5
        assert call_kwargs["sampling_params"].max_tokens == 64

    @pytest.mark.asyncio
    async def test_with_tools(self) -> None:
        fake_response = FakeSampleResponse(
            sequences=[FakeSampledSequence(tokens=[10], logprobs=[0.1])]
        )
        sampling_client = MagicMock()
        sampling_client.sample_async = AsyncMock(return_value=fake_response)

        renderer = MagicMock()
        renderer.build_generation_prompt.return_value = MagicMock()
        renderer.build_generation_prompt.return_value.to_ints.return_value = [1]
        renderer.get_stop_sequences.return_value = []
        renderer.create_conversation_prefix_with_tools.return_value = [
            {"role": "system", "content": "Tools: [search]"}
        ]
        renderer.parse_response.return_value = (
            {"role": "assistant", "content": "done"},
            True,
        )

        tools = [
            {
                "type": "function",
                "function": {"name": "search", "description": "Search", "parameters": {}},
            }
        ]
        result = await sample_chat_completion(
            sampling_client=sampling_client,
            renderer=renderer,
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
        )

        renderer.create_conversation_prefix_with_tools.assert_called_once()
        assert result.parse_success is True

    @pytest.mark.asyncio
    async def test_custom_stop_sequences(self) -> None:
        fake_response = FakeSampleResponse(
            sequences=[FakeSampledSequence(tokens=[10], logprobs=None)]
        )
        sampling_client = MagicMock()
        sampling_client.sample_async = AsyncMock(return_value=fake_response)

        renderer = MagicMock()
        renderer.build_generation_prompt.return_value = MagicMock()
        renderer.build_generation_prompt.return_value.to_ints.return_value = [1]
        renderer.parse_response.return_value = (
            {"role": "assistant", "content": "ok"},
            True,
        )

        await sample_chat_completion(
            sampling_client=sampling_client,
            renderer=renderer,
            messages=[{"role": "user", "content": "Hi"}],
            stop=["STOP"],
        )

        call_kwargs = sampling_client.sample_async.call_args.kwargs
        assert call_kwargs["sampling_params"].stop == ["STOP"]
        # get_stop_sequences should NOT be called when stop is explicit
        renderer.get_stop_sequences.assert_not_called()


# ---------------------------------------------------------------------------
# extract_sampling_params
# ---------------------------------------------------------------------------


class TestExtractSamplingParams:
    def test_all_params(self) -> None:
        params = extract_sampling_params(
            {
                "temperature": 0.5,
                "max_tokens": 256,
                "top_p": 0.9,
                "top_k": 50,
                "stop": ["STOP"],
                "irrelevant_param": True,
            }
        )
        assert params == {
            "temperature": 0.5,
            "max_tokens": 256,
            "top_p": 0.9,
            "top_k": 50,
            "stop": ["STOP"],
        }

    def test_max_completion_tokens(self) -> None:
        params = extract_sampling_params({"max_completion_tokens": 128})
        assert params == {"max_tokens": 128}

    def test_empty(self) -> None:
        assert extract_sampling_params({}) == {}
