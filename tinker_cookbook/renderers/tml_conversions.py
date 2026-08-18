"""Shared conversion helpers for public ``tml_renderers`` adapters."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, TypeGuard, cast
from urllib.parse import unquote, urlparse

from tinker_cookbook.image_processing_utils import image_to_data_uri
from tinker_cookbook.renderers.base import (
    AudioPart,
    ImagePart,
    Message,
    ToolCall,
    TrainOnWhat,
)

if TYPE_CHECKING:
    from tml_renderers import chat as tml_chat  # pyright: ignore[reportMissingImports]


TmlRenderInput: TypeAlias = (
    "Sequence[tml_chat.Message] | Sequence[tml_chat.OpenAIMessage] | tml_chat.MessageList"
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


def _is_tml_renderers_input(messages: object) -> TypeGuard[TmlRenderInput]:
    chat = import_module("tml_renderers.chat")
    if isinstance(messages, chat.MessageList):
        return True
    if isinstance(messages, list):
        return all(isinstance(message, chat.Message | chat.OpenAIMessage) for message in messages)
    return False


def _jsonable_cookbook_message(message: Message | Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(message, dict):
        return message

    result: dict[str, Any] = dict(message)
    for field in ("tool_calls", "unparsed_tool_calls"):
        if field in result:
            result[field] = [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                for item in result[field]
            ]
    return result


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
            f"tml_v0 does not fetch remote audio URLs (scheme {parsed.scheme!r}); "
            "provide encoded bytes, a local path, or a base64 data: URI"
        )

    suffix = path.suffix.lower().removeprefix(".")
    audio_format = explicit_format or {"wave": "wav", "mpeg": "mp3"}.get(suffix, suffix)
    if not audio_format:
        raise ValueError(f"cannot infer audio format from path {str(path)!r}; set AudioPart.format")
    return path.read_bytes(), audio_format


def _audio_part_to_openai(part: AudioPart) -> dict[str, Any]:
    """Convert a cookbook ``AudioPart`` to OpenAI's inline ``input_audio`` shape."""
    source = part["audio"]
    explicit_format = part.get("format")

    if isinstance(source, bytes):
        raw, audio_format = source, explicit_format or "wav"
    elif isinstance(source, str) and source.startswith("data:"):
        raw, audio_format = _decode_audio_data_uri(source, explicit_format)
    elif isinstance(source, str):
        raw, audio_format = _read_local_audio(source, explicit_format)
    else:
        raise TypeError(f"audio must be bytes or str; got {type(source)!r}")

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


def _normalize_cookbook_media(messages: Sequence[Message]) -> Sequence[Mapping[str, Any]]:
    """Rewrite cookbook image/audio parts to OpenAI-compatible content parts."""
    if not isinstance(messages, list):
        return messages

    normalized: list[Any] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            normalized.append(_jsonable_cookbook_message(message))
            continue

        new_content: list[Any] = []
        changed = False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                new_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_uri(cast(ImagePart, part)["image"])},
                    }
                )
                changed = True
            elif isinstance(part, dict) and part.get("type") == "audio":
                new_content.append(_audio_part_to_openai(cast(AudioPart, part)))
                changed = True
            else:
                new_content.append(part)
        normalized.append(
            _jsonable_cookbook_message({**message, "content": new_content} if changed else message)
        )
    return normalized


def _messages_to_render_input(messages: Sequence[Message] | TmlRenderInput) -> TmlRenderInput:
    if _is_tml_renderers_input(messages):
        return messages
    chat = import_module("tml_renderers.chat")
    return chat.OpenAIMessage.from_oss_messages(
        _normalize_cookbook_media(cast("Sequence[Message]", messages))
    )


def _assistant_target_indices(messages: Sequence[Message], train_on_what: TrainOnWhat) -> set[int]:
    assistant_indices = {i for i, message in enumerate(messages) if message["role"] == "assistant"}
    if train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES:
        return assistant_indices
    if train_on_what == TrainOnWhat.LAST_ASSISTANT_MESSAGE:
        return {max(assistant_indices)} if assistant_indices else set()
    if train_on_what == TrainOnWhat.CUSTOMIZED:
        return {
            i
            for i, message in enumerate(messages)
            if message["role"] == "assistant" and message.get("trainable", True)
        }
    raise NotImplementedError(
        f"tml_v0 currently supports {TrainOnWhat.ALL_ASSISTANT_MESSAGES.value}, "
        f"{TrainOnWhat.LAST_ASSISTANT_MESSAGE.value}, and {TrainOnWhat.CUSTOMIZED.value}; "
        f"got {train_on_what.value!r}"
    )


def _cookbook_messages_to_sft_input(
    messages: Sequence[Message] | TmlRenderInput, train_on_what: TrainOnWhat
) -> TmlRenderInput:
    chat = import_module("tml_renderers.chat")
    if train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES:
        return _messages_to_render_input(messages)

    if _is_tml_renderers_input(messages):
        raise NotImplementedError(
            "tml_v0 only supports selective train_on_what modes for cookbook/OpenAI "
            "message dictionaries. Pass train_on_what=ALL_ASSISTANT_MESSAGES when using "
            "native tml_renderers.chat.Message, OpenAIMessage, or MessageList inputs."
        )
    cookbook_messages = cast("Sequence[Message]", messages)

    openai_messages = chat.OpenAIMessage.from_oss_messages(
        _normalize_cookbook_media(cookbook_messages)
    )
    target_indices = _assistant_target_indices(cookbook_messages, train_on_what)
    zero_metadata = chat.MessageMetadata(training_metadata=chat.TrainingMetadata(0.0, False))
    flattened: list[tml_chat.Message] = []
    for idx, (cookbook_message, openai_message) in enumerate(
        zip(cookbook_messages, openai_messages, strict=True)
    ):
        rendered_messages = list(openai_message.to_messages())
        if cookbook_message["role"] == "assistant" and idx not in target_indices:
            rendered_messages = [
                (
                    message.copy(message_metadata=zero_metadata)
                    if message.author.kind == chat.AuthorKind.Model
                    else message
                )
                for message in rendered_messages
            ]
        flattened.extend(rendered_messages)
    return flattened


def _parsed_messages_to_cookbook(parsed: list[tml_chat.Message]) -> Message | None:
    chat = import_module("tml_renderers.chat")
    openai_messages = chat.OpenAIMessage.from_messages(parsed)
    if not openai_messages:
        return None
    openai_dicts = chat.OpenAIMessage.to_oss_messages(openai_messages)
    message = dict(openai_dicts[-1])
    if tool_calls := message.get("tool_calls"):
        message["tool_calls"] = [
            ToolCall(
                id=tool_call.get("id"),
                function=ToolCall.FunctionBody(
                    name=tool_call["function"]["name"],
                    arguments=tool_call["function"]["arguments"],
                ),
            )
            for tool_call in tool_calls
        ]
    return Message(**message)
