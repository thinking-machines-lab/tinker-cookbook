"""OpenAI format compatibility utilities for tinker-cookbook.

Stateless, model-neutral conversion between OpenAI Chat Completions message/tool dicts
and tinker-cookbook's Message/ToolSpec/ToolCall types, in both directions.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from typing import Any, Literal, NotRequired, TypedDict, cast
from urllib.parse import urlparse

from tinker_cookbook.audio_processing_utils import audio_part_to_input_audio
from tinker_cookbook.image_processing_utils import image_to_data_uri
from tinker_cookbook.renderers.base import (
    AudioPart,
    ContentPart,
    ImagePart,
    Message,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolSpec,
)


def _reasoning_content(msg: dict[str, Any]) -> str | None:
    reasoning = msg.get("reasoning_content")
    if reasoning is None or reasoning == "":
        return None
    if not isinstance(reasoning, str):
        raise ValueError(f"reasoning_content must be a string; got {type(reasoning)!r}")
    if msg["role"] != "assistant":
        raise ValueError(
            f"reasoning_content is only valid on assistant messages; got role {msg['role']!r}"
        )
    return reasoning


def _openai_part_to_tinker(part: dict[str, Any]) -> ContentPart:
    if part.get("type") == "image_url":
        image_url = part["image_url"]
        return ImagePart(
            type="image", image=image_url if isinstance(image_url, str) else image_url["url"]
        )
    if part.get("type") == "input_audio":
        input_audio = part["input_audio"]
        audio = AudioPart(
            type="audio",
            audio=base64.b64decode(input_audio["data"], validate=True),
            format=cast(Literal["wav", "mp3", "flac"], input_audio["format"].lower()),
        )
        if "num_frames" in input_audio:
            audio["num_frames"] = input_audio["num_frames"]
        if "sample_rate" in input_audio:
            audio["sample_rate"] = input_audio["sample_rate"]
        return audio
    return cast(ContentPart, part)


def openai_messages_to_tinker(messages: list[dict[str, Any]]) -> list[Message]:
    """Convert OpenAI/LiteLLM message dicts to tinker-cookbook Messages.

    A non-empty ``reasoning_content`` on an assistant message becomes a leading ThinkingPart.
    An ``image_url`` part becomes an ImagePart, and an ``input_audio`` part an AudioPart.
    """
    out: list[Message] = []
    for msg in messages:
        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = [_openai_part_to_tinker(part) for part in content]
        reasoning = _reasoning_content(msg)
        if reasoning is not None:
            thinking: ThinkingPart = {"type": "thinking", "thinking": reasoning}
            if isinstance(content, str):
                parts: list[ContentPart] = [thinking]
                if content:
                    parts.append(TextPart(type="text", text=content))
                content = parts
            else:
                content = [thinking, *content]
        tinker_msg: Message = {
            "role": msg["role"],
            "content": content,
        }
        if "name" in msg:
            tinker_msg["name"] = msg["name"]
        if "tool_call_id" in msg:
            tinker_msg["tool_call_id"] = msg["tool_call_id"]
        if "tool_calls" in msg:
            tinker_msg["tool_calls"] = [ToolCall.model_validate(tc) for tc in msg["tool_calls"]]
        out.append(tinker_msg)
    return out


class OpenAIChatInput(TypedDict):
    """The ``messages`` and ``tools`` of an OpenAI Chat Completions request."""

    messages: list[dict[str, Any]]
    tools: NotRequired[list[dict[str, Any]]]


def tinker_messages_to_openai(messages: Sequence[Message]) -> OpenAIChatInput:
    """Convert tinker-cookbook Messages to the ``messages`` and ``tools`` of an OpenAI Chat
    Completions request.

    - All ``tool_declare`` messages merge into ``tools``. Without one, the result has no ``tools``.
    - The thinking parts of a message join with ``"\\n"`` into one ``reasoning_content``.
    - Local images and audio become inline ``image_url`` and ``input_audio`` parts. An http(s)
      image URL stays unchanged.
    - The conversion drops ``trainable`` and ``unparsed_tool_calls``: OpenAI Chat has no field
      for them.
    - OpenAI's own API rejects a tool call without an ``id``, FLAC audio, and
      ``num_frames``/``sample_rate`` on audio.

    Raises:
        ValueError: If a message has thinking on a non-assistant role, or a ``tool_declare``
            message does not hold a JSON list of OpenAI tools (Kimi K2.5 writes TypeScript),
            or two declared tools share a name but differ, or a content part has an unknown type.
    """
    named: dict[str, dict[str, Any]] = {}
    tools: list[dict[str, Any]] = []
    for message in messages:
        if message["role"] != "tool_declare":
            continue
        for tool in _declared_tools(message):
            name = tool["function"]["name"]
            if name not in named:
                named[name] = tool
                tools.append(tool)
            elif named[name] != tool:
                raise ValueError(f"tool {name!r} is declared twice with different specs")
    result: OpenAIChatInput = {
        "messages": [
            _tinker_message_to_openai(message)
            for message in messages
            if message["role"] != "tool_declare"
        ]
    }
    if tools:
        result["tools"] = tools
    return result


def _declared_tools(message: Message) -> list[dict[str, Any]]:
    content = message["content"]
    try:
        tools = json.loads(content) if isinstance(content, str) else None
    except json.JSONDecodeError:
        tools = None
    if not isinstance(tools, list) or not all(
        isinstance(tool, dict) and isinstance(tool.get("function"), dict) for tool in tools
    ):
        raise ValueError(
            "a tool_declare message must hold a JSON list of OpenAI tools, as "
            "tool_specs_to_openai_tools returns (Kimi K2.5 writes TypeScript, which has no "
            "OpenAI form)"
        )
    for tool in tools:
        name = tool["function"].get("name")
        if not isinstance(name, str):
            raise ValueError(f"a declared tool name must be a string; got {name!r}")
    return tools


def _tinker_message_to_openai(message: Message) -> dict[str, Any]:
    role = message["role"]
    result: dict[str, Any] = {"role": role}
    content = message["content"]
    if isinstance(content, str):
        result["content"] = content
    else:
        thinking: list[str] = []
        parts: list[dict[str, Any]] = []
        for part in content:
            if part["type"] == "thinking":
                if role != "assistant":
                    raise ValueError(
                        f"thinking is only valid on assistant messages; got role {role!r}"
                    )
                # OpenAI has one reasoning string per message, so this join loses the order of
                # thinking relative to text.
                thinking.append(part["thinking"])
            elif part["type"] == "text":
                parts.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image":
                image = part["image"]
                # OpenAI accepts a remote URL unchanged; only a local image needs inlining.
                remote = isinstance(image, str) and urlparse(image).scheme in ("http", "https")
                url = image if remote else image_to_data_uri(image)
                parts.append({"type": "image_url", "image_url": {"url": url}})
            elif part["type"] == "audio":
                parts.append(
                    {"type": "input_audio", "input_audio": audio_part_to_input_audio(part)}
                )
            else:
                raise ValueError(f"unsupported content part type: {part['type']!r}")
        if thinking:
            result["reasoning_content"] = "\n".join(thinking)
        if len(parts) == 1 and parts[0].get("type") == "text":
            result["content"] = parts[0]["text"]
        elif not parts:
            result["content"] = None if message.get("tool_calls") else ""
        else:
            result["content"] = parts

    if "tool_calls" in message:
        result["tool_calls"] = [
            tc.model_dump(mode="json", exclude_none=True) if isinstance(tc, ToolCall) else tc
            for tc in message["tool_calls"]
        ]
    if "tool_call_id" in message:
        result["tool_call_id"] = message["tool_call_id"]
    if "name" in message:
        result["name"] = message["name"]
    return result


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


def tool_specs_to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Convert renderer ToolSpec values to OpenAI-format function tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in tools
    ]
