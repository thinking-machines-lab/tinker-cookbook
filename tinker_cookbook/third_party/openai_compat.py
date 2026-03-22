"""OpenAI format compatibility utilities for tinker-cookbook.

Stateless conversion between OpenAI API message/tool formats and tinker-cookbook's
internal Message/ToolSpec/ToolCall types.

The reverse direction (tinker -> OpenAI) is handled by ``Renderer.to_openai_message()``.
"""

from __future__ import annotations

from typing import Any

from tinker_cookbook.renderers.base import (
    Message,
    ToolCall,
    ToolSpec,
)


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
