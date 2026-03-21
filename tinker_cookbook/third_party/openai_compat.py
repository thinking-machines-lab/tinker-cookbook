"""OpenAI format compatibility utilities for tinker-cookbook renderers.

Provides conversion between OpenAI API message/tool formats and tinker-cookbook's
internal Message/ToolSpec/ToolCall types, plus a generic render-sample-parse pipeline.

The reverse direction (tinker -> OpenAI) is handled by ``Renderer.to_openai_message()``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

import tinker

from tinker_cookbook.renderers.base import (
    Message,
    Renderer,
    ToolCall,
    ToolSpec,
    format_content_as_string,
)


@dataclass
class SamplingResult:
    """Result of a Tinker sampling call with all data needed to build any response format."""

    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    logprobs: list[float] | None
    parsed_message: Message
    parse_success: bool
    model_name: str


def openai_messages_to_tinker(messages: list[dict[str, Any]]) -> list[Message]:
    """Convert OpenAI/LiteLLM message dicts to tinker-cookbook Messages."""
    out: list[Message] = []
    for msg in messages:
        tinker_msg: Message = {
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


def openai_tools_to_tinker(tools: list[dict[str, Any]]) -> list[ToolSpec]:
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


def prepare_messages_with_tools(
    renderer: Renderer,
    messages: list[Message],
    tools: list[dict[str, Any]],
) -> list[Message]:
    """Inject tool declarations into the message list via the renderer.

    Extracts the system message (if any), passes it to
    ``renderer.create_conversation_prefix_with_tools``, and prepends the
    resulting prefix messages to the remaining conversation.
    """
    tool_specs = openai_tools_to_tinker(tools)

    # Split out system message if present
    system_prompt = ""
    remaining: list[Message]
    if messages and messages[0]["role"] == "system":
        content = messages[0].get("content") or ""
        system_prompt = content if isinstance(content, str) else ""
        remaining = list(messages[1:])
    else:
        remaining = list(messages)

    prefix = renderer.create_conversation_prefix_with_tools(tool_specs, system_prompt)
    return prefix + remaining


def extract_sampling_params(optional_params: dict[str, Any]) -> dict[str, Any]:
    """Extract Tinker-compatible sampling parameters from an OpenAI-style params dict."""
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


def sampling_result_to_openai_dict(result: SamplingResult) -> dict[str, Any]:
    """Convert a SamplingResult to an OpenAI ChatCompletion-compatible dict."""
    content = result.parsed_message.get("content", "")
    if isinstance(content, list):
        content = format_content_as_string(content)

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


async def sample_chat_completion(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 1.0,
    max_tokens: int = 128,
    top_p: float = 1.0,
    top_k: int = -1,
    stop: list[str] | list[int] | None = None,
    tools: list[dict[str, Any]] | None = None,
    model_name: str = "tinker",
) -> SamplingResult:
    """Run the full render -> sample -> parse pipeline."""
    tinker_messages = openai_messages_to_tinker(messages)

    if tools:
        tinker_messages = prepare_messages_with_tools(renderer, tinker_messages, tools)

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

    return SamplingResult(
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
        logprobs=logprobs,
        parsed_message=parsed_message,
        parse_success=parse_success,
        model_name=model_name,
    )
