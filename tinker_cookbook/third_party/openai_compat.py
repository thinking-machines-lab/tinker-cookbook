"""OpenAI format compatibility utilities for tinker-cookbook.

Stateless conversion between OpenAI API message/tool formats and tinker-cookbook's
Message/ToolSpec/ToolCall types.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from tinker_cookbook.image_processing_utils import image_to_data_uri
from tinker_cookbook.renderers.base import (
    AudioPart,
    Message,
    ToolCall,
    ToolSpec,
)

_AUDIO_FORMAT_BY_MIME = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
}
_SUPPORTED_AUDIO_FORMATS = ("wav", "mp3", "flac")


def _decode_audio_data_uri(source: str, explicit_format: str | None) -> tuple[bytes, str]:
    header, separator, payload = source.partition(",")
    if not separator:
        raise ValueError("audio data URI must contain a comma-separated payload")
    if ";base64" not in header.lower():
        raise ValueError("audio data URI must use base64 encoding")

    mime = header[5:].split(";", 1)[0].lower()
    inferred_format = _AUDIO_FORMAT_BY_MIME.get(mime)
    if explicit_format and inferred_format and explicit_format != inferred_format:
        raise ValueError(
            f"AudioPart.format {explicit_format!r} disagrees with data URI MIME type {mime!r}"
        )
    audio_format = explicit_format or inferred_format
    if audio_format is None:
        raise ValueError(
            f"cannot infer audio format from data URI MIME type {mime!r}; "
            "set AudioPart.format explicitly"
        )
    try:
        return base64.b64decode(payload, validate=True), audio_format
    except ValueError as exc:
        raise ValueError("audio data URI payload is not valid base64") from exc


def _read_local_audio(source: str, explicit_format: str | None) -> tuple[bytes, str]:
    parsed = urlparse(source)
    if parsed.scheme == "":
        path = Path(source).expanduser()
    elif parsed.scheme == "file" and parsed.netloc in ("", "localhost"):
        path = Path(unquote(parsed.path)).expanduser()
    else:
        raise ValueError(
            f"OpenAI conversion does not fetch remote audio URLs (scheme {parsed.scheme!r}); "
            "provide encoded bytes, a local path, or a base64 data: URI"
        )

    suffix = path.suffix.lower().removeprefix(".")
    audio_format = explicit_format or {"wave": "wav", "mpeg": "mp3"}.get(suffix, suffix)
    if not audio_format:
        raise ValueError(f"cannot infer audio format from path {str(path)!r}; set AudioPart.format")
    return path.read_bytes(), audio_format


def _audio_part_to_openai(part: AudioPart) -> dict[str, Any]:
    source = part["audio"]
    explicit_format = part.get("format")

    if isinstance(source, bytes):
        raw, audio_format = source, explicit_format or "wav"
    elif source.startswith("data:"):
        raw, audio_format = _decode_audio_data_uri(source, explicit_format)
    else:
        raw, audio_format = _read_local_audio(source, explicit_format)

    if audio_format not in _SUPPORTED_AUDIO_FORMATS:
        raise ValueError(
            f"unsupported audio format {audio_format!r}; expected 'wav', 'mp3', or 'flac'"
        )

    input_audio: dict[str, Any] = {
        "data": base64.b64encode(raw).decode("ascii"),
        "format": audio_format,
    }
    has_num_frames = "num_frames" in part
    has_sample_rate = "sample_rate" in part
    if has_num_frames != has_sample_rate:
        raise ValueError("AudioPart must provide num_frames and sample_rate together")
    if audio_format != "wav" and not has_num_frames:
        raise ValueError(f"{audio_format} AudioPart must provide num_frames and sample_rate")
    if has_num_frames and has_sample_rate:
        num_frames = part["num_frames"]
        sample_rate = part["sample_rate"]
        if num_frames <= 0 or sample_rate <= 0:
            raise ValueError("AudioPart num_frames and sample_rate must be positive")
        input_audio.update(num_frames=num_frames, sample_rate=sample_rate)
    return {"type": "input_audio", "input_audio": input_audio}


def tinker_messages_to_openai(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Convert Cookbook messages, including typed tools and media, to OpenAI dicts."""
    converted_messages: list[dict[str, Any]] = []
    for message in messages:
        converted: dict[str, Any] = dict(message)
        if tool_calls := message.get("tool_calls"):
            converted["tool_calls"] = [
                tool_call.model_dump(mode="json") for tool_call in tool_calls
            ]
        if unparsed_tool_calls := message.get("unparsed_tool_calls"):
            converted["unparsed_tool_calls"] = [
                tool_call.model_dump(mode="json") for tool_call in unparsed_tool_calls
            ]

        content = message["content"]
        if isinstance(content, list):
            converted_content: list[Any] = []
            for part in content:
                if part["type"] == "image":
                    converted_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_data_uri(part["image"])},
                        }
                    )
                elif part["type"] == "audio":
                    converted_content.append(_audio_part_to_openai(part))
                else:
                    converted_content.append(part)
            converted["content"] = converted_content

        converted_messages.append(converted)
    return converted_messages


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
