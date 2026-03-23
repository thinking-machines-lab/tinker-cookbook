"""Tests for the LiteLLM integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.third_party.litellm.provider import (
    _extract_sampling_params,
    _prepare_messages_with_tools,
    _sample_chat_completion,
    _sampling_result_to_chat_completion_dict,
    _SamplingResult,
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
) -> _SamplingResult:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return _SamplingResult(
        prompt_token_ids=prompt_tokens or [1, 2, 3],
        completion_token_ids=completion_tokens or [4, 5, 6],
        logprobs=[0.1, 0.2, 0.3],
        parsed_message=msg,  # type: ignore[arg-type]
        parse_success=parse_success,
        model_name="tinker/test-model",
    )


# ---------------------------------------------------------------------------
# _extract_sampling_params
# ---------------------------------------------------------------------------


class TestExtractSamplingParams:
    def test_all_params(self) -> None:
        params = _extract_sampling_params(
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
        params = _extract_sampling_params({"max_completion_tokens": 128})
        assert params == {"max_tokens": 128}

    def test_empty(self) -> None:
        assert _extract_sampling_params({}) == {}


# ---------------------------------------------------------------------------
# _prepare_messages_with_tools
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

        result = _prepare_messages_with_tools(renderer, messages, tools)  # type: ignore[arg-type]

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

        _prepare_messages_with_tools(renderer, messages, tools)  # type: ignore[arg-type]

        args = renderer.create_conversation_prefix_with_tools.call_args
        assert args[0][1] == ""  # no system prompt


# ---------------------------------------------------------------------------
# _sampling_result_to_chat_completion_dict
# ---------------------------------------------------------------------------


class TestSamplingResultToDict:
    def test_basic_response(self) -> None:
        result = _make_sampling_result(content="Hi there!")
        d = _sampling_result_to_chat_completion_dict(result)

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
        d = _sampling_result_to_chat_completion_dict(result)
        assert d["choices"][0]["finish_reason"] == "length"

    def test_tool_calls_in_response(self) -> None:
        tc = ToolCall(
            function=ToolCall.FunctionBody(name="search", arguments='{"q": "test"}'),
            id="call_abc",
        )
        result = _make_sampling_result(tool_calls=[tc])
        d = _sampling_result_to_chat_completion_dict(result)

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
        d = _sampling_result_to_chat_completion_dict(result)
        assert d["choices"][0]["message"]["tool_calls"][0]["id"] == "call_0"

    def test_list_content_formatted_as_string(self) -> None:
        result = _make_sampling_result()
        result.parsed_message["content"] = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world!"},
        ]
        d = _sampling_result_to_chat_completion_dict(result)
        assert d["choices"][0]["message"]["content"] == "Hello \nworld!"


# ---------------------------------------------------------------------------
# _sample_chat_completion
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

        result = await _sample_chat_completion(
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
        result = await _sample_chat_completion(
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

        await _sample_chat_completion(
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
# LiteLLM provider
# ---------------------------------------------------------------------------


class TestTinkerLiteLLMProvider:
    def test_register_adds_to_provider_map(self) -> None:
        import litellm

        import tinker_cookbook.third_party.litellm.provider as provider_mod
        from tinker_cookbook.third_party.litellm import register_litellm_provider

        # Reset the singleton so we can test fresh registration
        old_registered = provider_mod._registered_provider
        provider_mod._registered_provider = None

        provider = None
        try:
            original_len = len(litellm.custom_provider_map)
            provider = register_litellm_provider()
            assert len(litellm.custom_provider_map) == original_len + 1
            entry = litellm.custom_provider_map[-1]
            assert entry["provider"] == "tinker"
            assert entry["custom_handler"] is provider

            # Calling again returns the same instance without adding a duplicate
            provider2 = register_litellm_provider()
            assert provider2 is provider
            assert len(litellm.custom_provider_map) == original_len + 1
        finally:
            # Clean up
            if provider is not None:
                litellm.custom_provider_map[:] = [
                    e
                    for e in litellm.custom_provider_map
                    if e.get("custom_handler") is not provider
                ]
            provider_mod._registered_provider = old_registered

    def test_set_client_creates_bundle(self) -> None:
        from tinker_cookbook.third_party.litellm.provider import TinkerLiteLLMProvider

        provider = TinkerLiteLLMProvider()
        mock_client = MagicMock()
        mock_client.get_base_model.return_value = "Qwen/Qwen3-8B"

        with (
            patch("tinker_cookbook.third_party.litellm.provider.get_tokenizer") as mock_get_tok,
            patch(
                "tinker_cookbook.third_party.litellm.provider.get_recommended_renderer_name",
                return_value="qwen3",
            ),
            patch("tinker_cookbook.third_party.litellm.provider.renderers.get_renderer"),
        ):
            mock_get_tok.return_value = MagicMock()
            provider.set_client(mock_client)

        assert "Qwen/Qwen3-8B" in provider._clients
        assert provider._clients["Qwen/Qwen3-8B"].sampling_client is mock_client

    def test_set_client_updates_existing_bundle(self) -> None:
        from tinker_cookbook.third_party.litellm.provider import (
            TinkerLiteLLMProvider,
            _ClientBundle,
        )

        provider = TinkerLiteLLMProvider()
        old_client = MagicMock()
        new_client = MagicMock()
        new_client.get_base_model.return_value = "Qwen/Qwen3-8B"

        provider._clients["Qwen/Qwen3-8B"] = _ClientBundle(
            sampling_client=old_client,
            renderer=MagicMock(),
            tokenizer=MagicMock(),
            base_model="Qwen/Qwen3-8B",
        )

        provider.set_client(new_client)
        assert provider._clients["Qwen/Qwen3-8B"].sampling_client is new_client

    @pytest.mark.asyncio
    async def test_acompletion_requires_base_model(self) -> None:
        from tinker_cookbook.third_party.litellm.provider import TinkerLiteLLMProvider

        provider = TinkerLiteLLMProvider()
        model_response = MagicMock()

        with pytest.raises(ValueError, match="base_model is required"):
            await provider.acompletion(
                model="tinker/test",
                messages=[],
                api_base="",
                custom_prompt_dict={},
                model_response=model_response,
                print_verbose=print,
                encoding=None,
                api_key=None,
                logging_obj=MagicMock(),
                optional_params={},
                litellm_params={},
            )

    @pytest.mark.asyncio
    async def test_acompletion_basic(self) -> None:
        from tinker_cookbook.third_party.litellm.provider import (
            TinkerLiteLLMProvider,
            _ClientBundle,
        )

        provider = TinkerLiteLLMProvider()

        fake_response = FakeSampleResponse(
            sequences=[FakeSampledSequence(tokens=[10, 20], logprobs=[0.1, 0.2])]
        )
        mock_sampling_client = MagicMock()
        mock_sampling_client.sample_async = AsyncMock(return_value=fake_response)

        mock_renderer = MagicMock()
        mock_renderer.build_generation_prompt.return_value = MagicMock()
        mock_renderer.build_generation_prompt.return_value.to_ints.return_value = [1, 2, 3]
        mock_renderer.get_stop_sequences.return_value = ["<|end|>"]
        mock_renderer.parse_response.return_value = (
            {"role": "assistant", "content": "Hello!"},
            True,
        )

        provider._clients["Qwen/Qwen3-8B"] = _ClientBundle(
            sampling_client=mock_sampling_client,
            renderer=mock_renderer,
            tokenizer=MagicMock(),
            base_model="Qwen/Qwen3-8B",
        )

        model_response = MagicMock()

        result = await provider.acompletion(
            model="tinker/my-model",
            messages=[{"role": "user", "content": "Hi"}],
            api_base="",
            custom_prompt_dict={},
            model_response=model_response,
            print_verbose=print,
            encoding=None,
            api_key=None,
            logging_obj=MagicMock(),
            optional_params={"temperature": 0.7, "max_tokens": 64},
            litellm_params={"base_model": "Qwen/Qwen3-8B"},
        )

        assert result is model_response
        # Verify the response was populated
        fields = result.choices[0].message.provider_specific_fields
        assert fields is not None
        assert fields["prompt_token_ids"] == [1, 2, 3]
        assert fields["completion_token_ids"] == [10, 20]
