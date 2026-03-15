"""
LiteLLM custom provider for Tinker sampling.

Enables using Tinker's native SamplingClient through LiteLLM's unified interface,
giving optimal sampling performance while exposing raw token IDs for training.

Usage::

    from tinker_cookbook.third_party.litellm import register_litellm_provider
    import litellm

    register_litellm_provider()

    response = await litellm.acompletion(
        model="tinker/my-model",
        messages=[{"role": "user", "content": "Hello!"}],
        base_model="Qwen/Qwen3-4B-Instruct-2507",
    )

    # Access raw tokens for training
    fields = response.choices[0].message.provider_specific_fields
    prompt_tokens = fields["prompt_token_ids"]
    completion_tokens = fields["completion_token_ids"]
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import httpx
import tinker

from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers.base import ToolCall, ToolSpec
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

try:
    from litellm.llms.custom_llm import CustomLLM
    from litellm.types.utils import Choices, Message, ModelResponse, Usage
except ImportError:
    raise ImportError(
        "litellm is required for the Tinker LiteLLM integration. "
        "Install it with: pip install 'tinker_cookbook[litellm]'"
    )


# ---------------------------------------------------------------------------
# Internal helpers: message/tool conversion, sampling, response building
# ---------------------------------------------------------------------------


@dataclass
class _SamplingResult:
    """Result of a Tinker sampling call with all data needed to build any response format."""

    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    logprobs: list[float] | None
    parsed_message: renderers.Message
    parse_success: bool
    model_name: str


def _convert_openai_messages(messages: list[dict[str, Any]]) -> list[renderers.Message]:
    """Convert OpenAI/LiteLLM message dicts to tinker-cookbook Messages."""
    out: list[renderers.Message] = []
    for msg in messages:
        tinker_msg: renderers.Message = {
            "role": msg["role"],
            "content": msg.get("content") or "",
        }
        if "name" in msg:
            tinker_msg["name"] = msg["name"]
        if "tool_call_id" in msg:
            tinker_msg["tool_call_id"] = msg["tool_call_id"]
        if "tool_calls" in msg:
            tinker_msg["tool_calls"] = [ToolCall.model_validate(tc) for tc in msg["tool_calls"]]
        out.append(tinker_msg)
    return out


def _convert_openai_tools(tools: list[dict[str, Any]]) -> list[ToolSpec]:
    """Convert OpenAI-format tool dicts to renderer ToolSpec."""
    out: list[ToolSpec] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool["function"]
        out.append(
            ToolSpec(
                name=func["name"],
                description=func.get("description", ""),
                parameters=func.get("parameters", {}),
            )
        )
    return out


def _prepare_messages_with_tools(
    renderer: renderers.Renderer,
    messages: list[renderers.Message],
    tools: list[dict[str, Any]],
) -> list[renderers.Message]:
    """Inject tool declarations into the message list via the renderer.

    Extracts the system message (if any), passes it to
    ``renderer.create_conversation_prefix_with_tools``, and prepends the
    resulting prefix messages to the remaining conversation.
    """
    tool_specs = _convert_openai_tools(tools)

    # Split out system message if present
    system_prompt = ""
    remaining: list[renderers.Message]
    if messages and messages[0]["role"] == "system":
        content = messages[0].get("content") or ""
        system_prompt = content if isinstance(content, str) else ""
        remaining = list(messages[1:])
    else:
        remaining = list(messages)

    prefix = renderer.create_conversation_prefix_with_tools(tool_specs, system_prompt)
    return prefix + remaining


async def _sample_chat_completion(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 1.0,
    max_tokens: int = 128,
    top_p: float = 1.0,
    top_k: int = -1,
    stop: list[str] | list[int] | None = None,
    tools: list[dict[str, Any]] | None = None,
    model_name: str = "tinker",
) -> _SamplingResult:
    """Run the full render -> sample -> parse pipeline."""
    tinker_messages = _convert_openai_messages(messages)

    if tools:
        tinker_messages = _prepare_messages_with_tools(renderer, tinker_messages, tools)

    model_input = renderer.build_generation_prompt(tinker_messages)
    prompt_token_ids: list[int] = model_input.to_ints()

    if stop is None:
        stop = renderer.get_stop_sequences()

    sample_response = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        ),
    )

    seq = sample_response.sequences[0]
    completion_token_ids: list[int] = seq.tokens
    logprobs: list[float] | None = seq.logprobs

    parsed_message, parse_success = renderer.parse_response(completion_token_ids)

    return _SamplingResult(
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
        logprobs=logprobs,
        parsed_message=parsed_message,
        parse_success=parse_success,
        model_name=model_name,
    )


def _sampling_result_to_chat_completion_dict(result: _SamplingResult) -> dict[str, Any]:
    """Convert a _SamplingResult to an OpenAI ChatCompletion-compatible dict."""
    content = result.parsed_message.get("content", "")
    if isinstance(content, list):
        content = renderers.format_content_as_string(content)

    # Build tool_calls list if present
    tool_calls_out: list[dict[str, Any]] | None = None
    raw_tool_calls: list[ToolCall] | None = result.parsed_message.get("tool_calls")
    if raw_tool_calls:
        tool_calls_out = [
            {
                "id": tc.id or f"call_{i}",
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for i, tc in enumerate(raw_tool_calls)
        ]

    if tool_calls_out:
        finish_reason = "tool_calls"
    elif result.parse_success:
        finish_reason = "stop"
    else:
        finish_reason = "length"

    message_dict: dict[str, Any] = {
        "role": "assistant",
        "content": content or None,
    }
    if tool_calls_out:
        message_dict["tool_calls"] = tool_calls_out

    return {
        "id": f"chatcmpl-tinker-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.model_name,
        "choices": [
            {
                "index": 0,
                "message": message_dict,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": len(result.prompt_token_ids),
            "completion_tokens": len(result.completion_token_ids),
            "total_tokens": len(result.prompt_token_ids) + len(result.completion_token_ids),
        },
    }


def _extract_sampling_params(optional_params: dict[str, Any]) -> dict[str, Any]:
    """Extract Tinker-compatible sampling parameters from LiteLLM optional_params."""
    params: dict[str, Any] = {}
    if "temperature" in optional_params:
        params["temperature"] = float(optional_params["temperature"])
    if "max_tokens" in optional_params:
        params["max_tokens"] = int(optional_params["max_tokens"])
    elif "max_completion_tokens" in optional_params:
        params["max_tokens"] = int(optional_params["max_completion_tokens"])
    if "top_p" in optional_params:
        params["top_p"] = float(optional_params["top_p"])
    if "top_k" in optional_params:
        params["top_k"] = int(optional_params["top_k"])
    if "stop" in optional_params:
        params["stop"] = optional_params["stop"]
    return params


def _build_model_response(
    result: _SamplingResult,
    model_response: ModelResponse,
) -> ModelResponse:
    """Populate a LiteLLM ModelResponse from a _SamplingResult."""
    completion_dict = _sampling_result_to_chat_completion_dict(result)

    choice_data = completion_dict["choices"][0]
    message_data = choice_data["message"]

    model_response.choices = [
        Choices(
            finish_reason=choice_data["finish_reason"],
            index=0,
            message=Message(
                content=message_data.get("content"),
                role="assistant",
                tool_calls=message_data.get("tool_calls"),
                provider_specific_fields={
                    "prompt_token_ids": result.prompt_token_ids,
                    "completion_token_ids": result.completion_token_ids,
                },
            ),
        )
    ]

    usage_data = completion_dict["usage"]
    model_response.usage = Usage(  # type: ignore[assignment]
        prompt_tokens=usage_data["prompt_tokens"],
        completion_tokens=usage_data["completion_tokens"],
        total_tokens=usage_data["total_tokens"],
    )
    model_response.model = result.model_name

    return model_response


def _map_tinker_error(exc: Exception) -> Exception:
    """Map Tinker SDK exceptions to LiteLLM-compatible errors."""
    import litellm.exceptions

    if isinstance(exc, tinker.AuthenticationError):
        return litellm.exceptions.AuthenticationError(
            message=str(exc),
            llm_provider="tinker",
            model="",
        )
    if isinstance(exc, tinker.RateLimitError):
        return litellm.exceptions.RateLimitError(
            message=str(exc),
            llm_provider="tinker",
            model="",
        )
    if isinstance(exc, tinker.APITimeoutError):
        return litellm.exceptions.Timeout(
            message=str(exc),
            llm_provider="tinker",
            model="",
        )
    if isinstance(exc, tinker.APIConnectionError):
        return litellm.exceptions.APIConnectionError(
            message=str(exc),
            llm_provider="tinker",
            model="",
        )
    if isinstance(exc, tinker.BadRequestError):
        return litellm.exceptions.BadRequestError(
            message=str(exc),
            llm_provider="tinker",
            model="",
        )
    # Fallback: re-raise as-is
    return exc


# ---------------------------------------------------------------------------
# Client bundle and provider
# ---------------------------------------------------------------------------


@dataclass
class _ClientBundle:
    """Cached group of objects needed to sample from a specific model."""

    sampling_client: tinker.SamplingClient
    renderer: renderers.Renderer
    tokenizer: Tokenizer
    base_model: str


class TinkerLiteLLMProvider(CustomLLM):
    """LiteLLM custom provider that routes calls through Tinker's native SamplingClient."""

    def __init__(self) -> None:
        super().__init__()
        self._clients: dict[str, _ClientBundle] = {}
        self._service_client: tinker.ServiceClient | None = None

    def _get_service_client(self) -> tinker.ServiceClient:
        if self._service_client is None:
            self._service_client = tinker.ServiceClient()
        return self._service_client

    def _get_or_create_client(self, base_model: str) -> _ClientBundle:
        """Get or lazily create a client bundle for the given base model."""
        if base_model not in self._clients:
            tokenizer = get_tokenizer(base_model)
            renderer_name = get_recommended_renderer_name(base_model)
            renderer = renderers.get_renderer(renderer_name, tokenizer)
            sampling_client = self._get_service_client().create_sampling_client(
                base_model=base_model
            )
            self._clients[base_model] = _ClientBundle(
                sampling_client=sampling_client,
                renderer=renderer,
                tokenizer=tokenizer,
                base_model=base_model,
            )
        return self._clients[base_model]

    def set_client(
        self,
        base_model: str,
        sampling_client: tinker.SamplingClient,
    ) -> None:
        """Inject a custom SamplingClient for a base model.

        Use this after saving new weights to point at a specific checkpoint,
        or to share a SamplingClient with other parts of your training loop.
        """
        if base_model in self._clients:
            self._clients[base_model].sampling_client = sampling_client
        else:
            tokenizer = get_tokenizer(base_model)
            renderer_name = get_recommended_renderer_name(base_model)
            renderer = renderers.get_renderer(renderer_name, tokenizer)
            self._clients[base_model] = _ClientBundle(
                sampling_client=sampling_client,
                renderer=renderer,
                tokenizer=tokenizer,
                base_model=base_model,
            )

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client=None,
    ) -> ModelResponse:
        base_model: str = (litellm_params or {}).get("base_model", "")
        if not base_model:
            raise ValueError(
                "base_model is required for the Tinker provider. "
                "Pass it as: litellm.acompletion(..., base_model='Qwen/Qwen3-8B')"
            )

        bundle = self._get_or_create_client(base_model)
        sampling_params = _extract_sampling_params(optional_params)

        try:
            result = await _sample_chat_completion(
                sampling_client=bundle.sampling_client,
                renderer=bundle.renderer,
                messages=messages,
                tools=optional_params.get("tools"),
                model_name=model,
                **sampling_params,
            )
        except tinker.TinkerError as exc:
            raise _map_tinker_error(exc) from exc

        return _build_model_response(result, model_response)

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client=None,
    ) -> ModelResponse:
        base_model: str = (litellm_params or {}).get("base_model", "")
        if not base_model:
            raise ValueError(
                "base_model is required for the Tinker provider. "
                "Pass it as: litellm.completion(..., base_model='Qwen/Qwen3-8B')"
            )

        bundle = self._get_or_create_client(base_model)
        sampling_params = _extract_sampling_params(optional_params)

        try:
            result = asyncio.get_event_loop().run_until_complete(
                _sample_chat_completion(
                    sampling_client=bundle.sampling_client,
                    renderer=bundle.renderer,
                    messages=messages,
                    tools=optional_params.get("tools"),
                    model_name=model,
                    **sampling_params,
                )
            )
        except tinker.TinkerError as exc:
            raise _map_tinker_error(exc) from exc

        return _build_model_response(result, model_response)


def register_litellm_provider() -> TinkerLiteLLMProvider:
    """Register the Tinker provider with LiteLLM. Call once at startup.

    Returns the provider instance, which can be used to inject custom
    SamplingClients via ``provider.set_client(base_model, client)``.
    """
    import litellm

    provider = TinkerLiteLLMProvider()
    litellm.custom_provider_map.append({"provider": "tinker", "custom_handler": provider})
    return provider
