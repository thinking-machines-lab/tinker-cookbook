"""Tests for tml-renderers compatibility utilities."""

from __future__ import annotations

import base64
import io
import json
import wave
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

import pytest
from PIL import Image

from tinker_cookbook.audio_processing_utils import audio_part_to_input_audio
from tinker_cookbook.image_processing_utils import image_to_data_uri
from tinker_cookbook.renderers.base import (
    AudioPart,
    ImagePart,
    Message,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
)
from tinker_cookbook.third_party.openai_compat import tool_specs_to_openai_tools
from tinker_cookbook.third_party.tml_renderers_compat import (
    message_list_to_tinker_messages,
    tinker_messages_to_message_list,
    tinker_messages_to_sft_message_list,
)

if TYPE_CHECKING:
    from tml_renderers import chat as tml_chat  # pyright: ignore[reportMissingImports]

chat = import_module("tml_renderers.chat")

# Sorted keys: tml-renderers stores tool parameters with sorted keys.
_TOOL: ToolSpec = {
    "name": "get_weather",
    "description": "Get the weather for a city.",
    "parameters": {"properties": {"city": {"type": "string"}}, "type": "object"},
}


def _declare(tool: ToolSpec) -> Message:
    return {
        "role": "tool_declare",
        "content": json.dumps(tool_specs_to_openai_tools([tool]), separators=(",", ":")),
    }


def _call(call_id: str | None, arguments: str) -> ToolCall:
    return ToolCall(
        id=call_id, function=ToolCall.FunctionBody(name="get_weather", arguments=arguments)
    )


def _wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 160)
    return buffer.getvalue()


def _png_data_uri() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (3, 2)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _weights(message_list: tml_chat.MessageList) -> list[float | None]:
    return [
        m.message_metadata.training_metadata.weight
        if m.message_metadata and m.message_metadata.training_metadata
        else None
        for m in message_list.messages
    ]


# The dicts tml_v0 passed to the wheel's from_oss_messages before tml_renderers_compat.
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


def _normalize_cookbook_media(messages: Sequence[Message]) -> Sequence[Mapping[str, Any]]:
    """Rewrite cookbook image/audio parts to OpenAI-compatible content parts.

    Images become inline ``image_url`` data URIs. Audio becomes inline
    ``input_audio`` base64. Message count and ordering are preserved.
    """
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
                new_content.append(
                    {
                        "type": "input_audio",
                        "input_audio": audio_part_to_input_audio(cast(AudioPart, part)),
                    }
                )
                changed = True
            else:
                new_content.append(part)
        normalized.append(
            _jsonable_cookbook_message({**message, "content": new_content} if changed else message)
        )
    return normalized


_CORPUS: dict[str, list[Message]] = {
    "interleaved thinking": [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "a"},
                {"type": "text", "text": "x"},
                {"type": "thinking", "thinking": "b"},
                {"type": "text", "text": "y"},
            ],
        },
    ],
    "declaration after the system prompt": [
        {"role": "system", "content": "S"},
        _declare(_TOOL),
        {"role": "user", "content": "SF?"},
    ],
    "tool calls and results": [
        {"role": "assistant", "content": "", "tool_calls": [_call("c1", '{"city":"SF"}')]},
        {"role": "tool", "content": "Sunny", "tool_call_id": "c1", "name": "get_weather"},
        {"role": "tool", "content": "Rain", "tool_call_id": "c2"},
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "t"}],
            "tool_calls": [_call(None, '{"city": "é", "n": [1.50, null]}')],
        },
    ],
    "media": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (7, 5))},
                {"type": "audio", "audio": _wav()},
                {
                    "type": "audio",
                    "audio": b"mp3",
                    "format": "mp3",
                    "num_frames": 9,
                    "sample_rate": 8000,
                },
            ],
        }
    ],
    "openai media parts": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _png_data_uri()}},  # type: ignore[list-item]
                {
                    "type": "input_audio",  # type: ignore[list-item]
                    "input_audio": {
                        "data": "AAAA",
                        "format": "mp3",
                        "num_frames": 1,
                        "sample_rate": 8000,
                    },
                },
            ],
        }
    ],
    "remote image with a size": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",  # type: ignore[list-item]
                    "image_url": {"url": "https://example.com/a.png", "width": 4, "height": 3},
                }
            ],
        }
    ],
    "shapes outside the cookbook type": [
        {"role": "user", "content": [{"type": "thinking", "thinking": "t"}]},
        {
            "role": "tool_declare",
            "content": '[{"type":"function","function":{"name":"f","description":null}}]',
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [  # type: ignore[list-item]
                {"id": "c1", "function": {"name": "f"}},
                {"id": "c2", "function": {"name": "f", "arguments": {"a": 1}}},
            ],
        },
    ],
}


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("messages", _CORPUS.values(), ids=_CORPUS.keys())
def test_message_list_matches_the_wheel_cookbook_reader(messages: list[Message]) -> None:
    turns = chat.OpenAIMessage.from_oss_messages(_normalize_cookbook_media(messages))
    expected = chat.MessageList([m for turn in turns for m in turn.to_messages()])

    assert tinker_messages_to_message_list(messages) == expected


def test_empty_content_keeps_an_empty_text() -> None:
    # tml-renderers 0.1.0 drops a turn with `content: []`; later readers keep an empty text.
    message_list = tinker_messages_to_message_list([{"role": "user", "content": []}])

    assert [m.content for m in message_list.messages] == [chat.Text("")]


def test_messages_round_trip_through_message_list() -> None:
    messages: list[Message] = [
        {"role": "system", "content": "Be brief."},
        _declare(_TOOL),
        {"role": "user", "content": "Weather in SF?"},
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "Look it up."}],
            "tool_calls": [_call("call_1", '{"city": "SF"}')],
        },
        {"role": "tool", "content": "Sunny.", "tool_call_id": "call_1", "name": "get_weather"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Report it."},
                {"type": "text", "text": "It is sunny."},
                {"type": "thinking", "thinking": "Done."},
            ],
        },
    ]

    message_list = tinker_messages_to_message_list(messages)

    assert message_list_to_tinker_messages(message_list) == messages


# TINK-351: a cookbook ImagePart has no size.
_ROUND_TRIP = {
    name: messages for name, messages in _CORPUS.items() if name != "remote image with a size"
}


@pytest.mark.parametrize("messages", _ROUND_TRIP.values(), ids=_ROUND_TRIP.keys())
def test_message_list_survives_a_round_trip_through_cookbook_messages(
    messages: list[Message],
) -> None:
    message_list = tinker_messages_to_message_list(messages)

    assert (
        tinker_messages_to_message_list(message_list_to_tinker_messages(message_list))
        == message_list
    )


def test_remote_audio_round_trips_as_a_pointer() -> None:
    messages: list[Message] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio": "gs://bucket/clip.wav",
                    "format": "wav",
                    "num_frames": 16000,
                    "sample_rate": 16000,
                }
            ],
        }
    ]

    message_list = tinker_messages_to_message_list(messages)

    assert message_list.messages[0].content == chat.AudioPointer(
        "gs://bucket/clip.wav", chat.AudioFormat.Wav, 16000, 16000
    )
    assert message_list_to_tinker_messages(message_list) == messages


def test_round_trip_losses() -> None:
    # TINK-351: the cookbook has no tool-error part and no image size.
    message_list = chat.MessageList(
        [
            chat.Message(chat.ToolError("city not found"), chat.Author(chat.AuthorKind.Tool, "f")),
            chat.Message(
                chat.ImagePointer("https://example.com/a.png", chat.ImageFormat.Png, 4, 3),
                chat.Author(chat.AuthorKind.User),
            ),
        ]
    )

    assert message_list_to_tinker_messages(message_list) == [
        {"role": "tool", "content": "city not found", "name": "f"},
        {"role": "user", "content": [{"type": "image", "image": "https://example.com/a.png"}]},
    ]


_SFT_MESSAGES: list[Message] = [
    {"role": "user", "content": "Prompt."},
    {"role": "assistant", "content": "First.", "trainable": False},
    {"role": "assistant", "content": "Second.", "trainable": True},
]


@pytest.mark.parametrize(
    ("train_on_what", "weights"),
    [
        # user; "First." and its end of turn; "Second." and its end of turn
        (TrainOnWhat.ALL_ASSISTANT_MESSAGES, [None, None, None, None, None]),
        (TrainOnWhat.LAST_ASSISTANT_MESSAGE, [None, 0.0, 0.0, None, None]),
        (TrainOnWhat.CUSTOMIZED, [None, 0.0, 0.0, None, None]),
    ],
)
def test_sft_message_list_weights_the_selected_turns(
    train_on_what: TrainOnWhat, weights: list[float | None]
) -> None:
    message_list = tinker_messages_to_sft_message_list(_SFT_MESSAGES, train_on_what)

    assert _weights(message_list) == weights


def test_sft_message_list_rejects_modes_tml_renderers_cannot_weight() -> None:
    with pytest.raises(NotImplementedError, match="all_messages"):
        tinker_messages_to_sft_message_list(_SFT_MESSAGES, TrainOnWhat.ALL_MESSAGES)


@pytest.mark.parametrize(
    "message",
    [
        {
            "role": "tool_declare",
            "content": '[{"type":"function","function":{"name":"f","parameters":null}}]',
        },
        {"role": "developer", "content": "hi"},
        {"role": "tool_declare", "content": "function get_weather(city: string): string"},
    ],
    ids=["null tool parameters", "unknown role", "TypeScript tool declaration"],
)
def test_unconvertible_messages_raise(message: Message) -> None:
    with pytest.raises(ValueError):
        tinker_messages_to_message_list([message])


def test_a_text_tool_call_has_no_tinker_form() -> None:
    message_list = chat.MessageList(
        [chat.Message(chat.InvokeTool("raw call"), chat.Author(chat.AuthorKind.Model))]
    )

    with pytest.raises(ValueError, match="text tool call"):
        message_list_to_tinker_messages(message_list)
