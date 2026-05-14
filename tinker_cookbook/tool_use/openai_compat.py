"""Helpers for OpenAI-compatible tool-call interop.

Some OpenAI-compatible sampling APIs return model-native tool calls as text in
``message.content`` instead of filling the OpenAI ``message.tool_calls`` field.
These helpers convert XML-style Qwen/Tinker tool-call text into the cookbook's
structured ``ToolCall`` objects and can linearize structured tool history back
into text for providers that reject OpenAI ``role=tool`` history.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from tinker_cookbook.renderers.base import (
    Message,
    TextPart,
    ThinkingPart,
    ToolCall,
    UnparsedToolCall,
)

_THINK_RE = re.compile(r"(?is)<think\b[^>]*>(.*?)</think>\s*")
_TOOL_CALL_RE = re.compile(r"(?is)<tool_call\b[^>]*>(.*?)</tool_call>")
_FUNCTION_RE = re.compile(r"(?is)<function\s*=\s*([^>\s]+)\s*>(.*?)</function>")
_PARAMETER_RE = re.compile(r"(?is)<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>")


def strip_xml_thinking(content: str) -> tuple[str, str | None]:
    """Strip XML-style thinking blocks from visible content.

    Returns the visible content and the removed thinking text, if any.
    """

    thinking_parts: list[str] = []

    def collect_thinking(match: re.Match[str]) -> str:
        thinking = match.group(1).strip()
        if thinking:
            thinking_parts.append(thinking)
        return ""

    visible_content = _THINK_RE.sub(collect_thinking, content).strip()
    hidden_thinking = "\n\n".join(thinking_parts) or None
    return visible_content, hidden_thinking


def parse_xml_tool_calls(
    content: str,
    *,
    allowed_tool_names: set[str] | None = None,
) -> tuple[list[ToolCall], list[UnparsedToolCall]]:
    """Parse Qwen-style XML tool calls from text.

    Supports blocks like::

        <tool_call>
        <function=search>
        <parameter=query>
        weather in NYC
        </parameter>
        </function>
        </tool_call>

    Parameter values are JSON-decoded when possible and otherwise kept as
    strings. If ``allowed_tool_names`` is provided, calls to other tools are
    reported as unparsed instead of being returned as executable tool calls.
    """

    tool_calls: list[ToolCall] = []
    unparsed_tool_calls: list[UnparsedToolCall] = []

    for block_match in _TOOL_CALL_RE.finditer(content):
        raw_text = block_match.group(0)
        block = block_match.group(1)
        function_match = _FUNCTION_RE.search(block)
        if function_match is None:
            unparsed_tool_calls.append(
                UnparsedToolCall(raw_text=raw_text, error="Missing <function=...> block")
            )
            continue

        function_name = function_match.group(1).strip().strip("\"'")
        body = function_match.group(2)
        if not function_name:
            unparsed_tool_calls.append(
                UnparsedToolCall(raw_text=raw_text, error="Missing function name")
            )
            continue
        if allowed_tool_names is not None and function_name not in allowed_tool_names:
            unparsed_tool_calls.append(
                UnparsedToolCall(
                    raw_text=raw_text,
                    error=f"Tool name {function_name!r} is not in allowed_tool_names",
                )
            )
            continue

        arguments: dict[str, Any] = {}
        for parameter_match in _PARAMETER_RE.finditer(body):
            parameter_name = parameter_match.group(1).strip().strip("\"'")
            if not parameter_name:
                unparsed_tool_calls.append(
                    UnparsedToolCall(raw_text=raw_text, error="Empty parameter name")
                )
                break

            raw_value = parameter_match.group(2).strip()
            try:
                value: Any = json.loads(raw_value)
            except json.JSONDecodeError:
                value = raw_value
            arguments[parameter_name] = value
        else:
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex}",
                    function=ToolCall.FunctionBody(
                        name=function_name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
            )

    return tool_calls, unparsed_tool_calls


def normalize_xml_tool_call_message(
    message: Message,
    *,
    allowed_tool_names: set[str] | None = None,
) -> Message:
    """Convert assistant XML tool-call text into structured ``tool_calls``.

    If the message already contains structured tool calls, it is returned
    unchanged. Otherwise, assistant string content is scanned for XML-style
    ``<tool_call>`` blocks. Parsed tool calls are moved into
    ``message["tool_calls"]`` and visible text outside the XML blocks is kept in
    ``message["content"]``.
    """

    if message["role"] != "assistant" or "tool_calls" in message:
        return message.copy()
    if not isinstance(message["content"], str):
        return message.copy()

    content, hidden_thinking = strip_xml_thinking(message["content"])
    tool_calls, unparsed_tool_calls = parse_xml_tool_calls(
        content,
        allowed_tool_names=allowed_tool_names,
    )
    if not tool_calls and not unparsed_tool_calls:
        return {**message, "content": content}

    visible_content = _TOOL_CALL_RE.sub("", content).strip()
    normalized: Message = {**message, "content": visible_content}
    if hidden_thinking:
        normalized["content"] = [
            ThinkingPart(type="thinking", thinking=hidden_thinking),
            TextPart(type="text", text=visible_content),
        ]
    if tool_calls:
        normalized["tool_calls"] = tool_calls
    if unparsed_tool_calls:
        normalized["unparsed_tool_calls"] = unparsed_tool_calls
    return normalized


def _tool_call_as_text(tool_call: ToolCall) -> str:
    return f"Tool call: {tool_call.function.name}\nArguments: {tool_call.function.arguments}"


def linearize_tool_history_for_text_chat(messages: list[Message]) -> list[Message]:
    """Represent OpenAI-style tool history as plain assistant text.

    This is useful when sending a multi-turn history to a provider that accepts
    normal chat messages but rejects OpenAI ``assistant.tool_calls`` and
    ``role=tool`` messages.
    """

    converted: list[Message] = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if message["role"] == "assistant" and tool_calls:
            converted.append(
                {
                    "role": "assistant",
                    "content": "\n\n".join(_tool_call_as_text(tc) for tc in tool_calls),
                }
            )
        elif message["role"] == "tool":
            converted.append(
                {
                    "role": "assistant",
                    "content": f"Tool result:\n{message['content']}",
                }
            )
        else:
            converted.append(message.copy())
    return converted
