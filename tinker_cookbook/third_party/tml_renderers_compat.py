"""tml-renderers compatibility utilities for tinker-cookbook.

Conversion between tinker-cookbook Messages and ``tml_renderers.chat.MessageList``, built with
the tml-renderers constructors:

- Thinking and text parts keep their order.
- A ``tool_declare`` message keeps its place.
- The conversion drops ``unparsed_tool_calls``: tml-renderers has no content for it.

``tinker_messages_to_sft_message_list`` also sets the training weights that a
``TrainOnWhat`` selects.
"""

from __future__ import annotations

import base64
import io
import json
import wave
from collections.abc import Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import urlparse

from PIL import Image

from tinker_cookbook.audio_processing_utils import audio_part_to_input_audio
from tinker_cookbook.image_processing_utils import image_to_data_uri
from tinker_cookbook.renderers.base import (
    AudioPart,
    ContentPart,
    Message,
    ToolCall,
    TrainOnWhat,
)
from tinker_cookbook.third_party.openai_compat import (
    _declared_tools,
)
from tinker_cookbook.tokenizer_utils import ensure_tml_renderers_importable

if TYPE_CHECKING:
    # tml_renderers is imported lazily at runtime; import it here for annotations only.
    from tml_renderers import chat as tml_chat  # pyright: ignore[reportMissingImports]

_AUDIO_MIME = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}


def _chat() -> Any:
    ensure_tml_renderers_importable()
    return import_module("tml_renderers.chat")


def _tool_declare(message: Message) -> tml_chat.ToolDeclareJson:
    chat = _chat()
    specs = []
    for tool in _declared_tools(message):
        function = tool["function"]
        parameters = function.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("a declared tool's parameters must be a dict")
        type_name, description = tool.get("type"), function.get("description")
        specs.append(
            chat.ToolSpecJson(
                type_name if isinstance(type_name, str) else "function",
                function["name"],
                description if isinstance(description, str) else "",
                parameters,
            )
        )
    return chat.ToolDeclareJson(specs)


def _image(url: str, image_url: dict[str, Any]) -> tml_chat.ImagePointer:
    # tml-renderers reads the size and format of a data URI from its bytes; any other URL
    # must give them.
    chat = _chat()
    formats = {"png": chat.ImageFormat.Png, "jpeg": chat.ImageFormat.Jpeg}
    formats["webp"] = chat.ImageFormat.Webp
    header, _, payload = url.partition(",")
    if url.startswith("data:") and "base64" in header.lower().split(";")[1:]:
        with Image.open(io.BytesIO(base64.b64decode(payload))) as image:
            width, height = image.size
            image_format = formats.get((image.format or "").lower(), chat.ImageFormat.Png)
    elif "width" in image_url and "height" in image_url:
        width, height = image_url["width"], image_url["height"]
        image_format = formats[str(image_url.get("format", "png")).lower()]
    else:
        raise ValueError(f"image {url[:64]!r} must be a base64 data URI or give width and height")
    if width <= 0 or height <= 0:
        raise ValueError(f"image {url[:64]!r} has size {width}x{height}")
    return chat.ImagePointer(url, image_format, width, height)


def _audio(input_audio: dict[str, Any]) -> tml_chat.AudioPointer:
    chat = _chat()
    data, audio_format = input_audio["data"], input_audio["format"].lower()
    if audio_format not in _AUDIO_MIME:
        raise ValueError(f"unsupported audio format {audio_format!r}")
    if audio_format == "wav":
        # tml-renderers reads WAV metadata from the header, not from the part.
        try:
            with wave.open(io.BytesIO(base64.b64decode(data, validate=True)), "rb") as wav:
                num_frames, sample_rate = wav.getnframes(), wav.getframerate()
        except (EOFError, wave.Error) as e:
            raise ValueError(f"could not read WAV header from input_audio data: {e!r}") from e
    elif "num_frames" in input_audio and "sample_rate" in input_audio:
        num_frames, sample_rate = input_audio["num_frames"], input_audio["sample_rate"]
    else:
        raise ValueError(f"{audio_format} input_audio must carry num_frames and sample_rate")
    if sample_rate == 0:
        raise ValueError(f"{audio_format} audio has sample_rate 0")
    formats = {
        "wav": chat.AudioFormat.Wav,
        "mp3": chat.AudioFormat.Mp3,
        "flac": chat.AudioFormat.Flac,
    }
    return chat.AudioPointer(
        f"data:{_AUDIO_MIME[audio_format]};base64,{data}",
        formats[audio_format],
        num_frames,
        sample_rate,
    )


def _is_remote(source: object) -> bool:
    return isinstance(source, str) and urlparse(source).scheme not in ("", "data", "file")


def _remote_audio(part: AudioPart) -> tml_chat.AudioPointer:
    # The renderer decides whether it can load the location; the converter keeps it.
    chat = _chat()
    if "format" not in part or "num_frames" not in part or "sample_rate" not in part:
        raise ValueError("an AudioPart with a URL must give format, num_frames, and sample_rate")
    formats = {
        "wav": chat.AudioFormat.Wav,
        "mp3": chat.AudioFormat.Mp3,
        "flac": chat.AudioFormat.Flac,
    }
    return chat.AudioPointer(
        cast(str, part["audio"]), formats[part["format"]], part["num_frames"], part["sample_rate"]
    )


def _content(part: dict[str, Any]) -> Any:
    chat = _chat()
    match part["type"]:
        case "text":
            return chat.Text(part["text"])
        case "thinking":
            return chat.Thinking(part["thinking"])
        case "image":
            return _image(image_to_data_uri(part["image"]), {})
        case "audio":
            if _is_remote(part["audio"]):
                return _remote_audio(cast(AudioPart, part))
            return _audio(audio_part_to_input_audio(cast(AudioPart, part)))
        case "tool_error":
            return chat.ToolError(part["text"])
        # OpenAI parts, which tml_v0 accepted through the wheel's reader.
        case "image_url":
            image_url = part["image_url"]
            if isinstance(image_url, str):
                return _image(image_url, {})
            return _image(image_url["url"], image_url)
        case "input_audio":
            return _audio(part["input_audio"])
    raise ValueError(f"unsupported content part type: {part['type']!r}")


def _invoke_tool(call: ToolCall | dict[str, Any]) -> tml_chat.InvokeTool:
    chat = _chat()
    if not isinstance(call, ToolCall):
        # A dict may omit arguments or give them as an object, as tml-renderers accepts.
        function = dict(call["function"])
        arguments = function.get("arguments")
        function["arguments"] = "{}" if arguments is None else arguments
        if not isinstance(function["arguments"], str):
            function["arguments"] = json.dumps(function["arguments"])
        call = ToolCall.model_validate({**call, "function": function})
    arguments = json.loads(call.function.arguments)
    if not isinstance(arguments, dict):
        raise ValueError("tool call arguments must decode to a JSON object")
    # Each argument value is its own JSON text, in json.dumps's default spacing.
    args = [chat.ToolArg(name, json.dumps(value)) for name, value in arguments.items()]
    return chat.InvokeTool(
        chat.StructuredToolCall(call.function.name, args=args, tool_call_id=call.id)
    )


def _tml_renderers_openai_message(message: Message) -> tml_chat.OpenAIMessage:
    """One tinker-cookbook Message as one ``tml_renderers.chat.OpenAIMessage``."""
    chat = _chat()
    role = message["role"]
    if role == "tool_declare":
        return chat.OpenAIMessage(
            author=chat.Author(chat.AuthorKind.System, "tool_declare"),
            content=[_tool_declare(message)],
        )
    kinds = {
        "user": chat.AuthorKind.User,
        "assistant": chat.AuthorKind.Model,
        "system": chat.AuthorKind.System,
        "tool": chat.AuthorKind.Tool,
    }
    if role not in kinds:
        raise ValueError(f"unsupported role: {role!r}")
    content = message.get("content")
    parts: list[dict[str, Any]] = (
        [{"type": "text", "text": content}]
        if isinstance(content, str)
        else [dict(part) for part in content or []]
    )
    items = [_content(part) for part in parts] or [chat.Text("")]
    calls = [_invoke_tool(call) for call in message.get("tool_calls", [])]
    if calls:
        # A tool-call turn has no empty text part.
        items = [item for item in items if not (isinstance(item, chat.Text) and not item.text)]
    tool_name = message.get("name") if role == "tool" else None
    return chat.OpenAIMessage(
        author=chat.Author(kinds[role], tool_name),
        content=items + calls,
        tool_call_id=message.get("tool_call_id"),
        tool_name=tool_name,
    )


def tinker_messages_to_message_list(messages: Sequence[Message]) -> tml_chat.MessageList:
    """Convert tinker-cookbook Messages to a ``tml_renderers.chat.MessageList``.

    Each Message becomes the Messages of its turn, and an assistant turn ends with
    ``ModelEndSampling``. The conversion ignores ``trainable``.

    Raises ValueError for an unknown role or content part, or a ``tool_declare`` message
    that does not hold a JSON list of OpenAI tools.
    """
    chat = _chat()
    return chat.MessageList(
        [
            part
            for message in messages
            for part in _tml_renderers_openai_message(message).to_messages()
        ]
    )


def _trained_assistant_indices(messages: Sequence[Message], train_on_what: TrainOnWhat) -> set[int]:
    assistant_indices = {i for i, message in enumerate(messages) if message["role"] == "assistant"}
    if train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES:
        return assistant_indices
    if train_on_what == TrainOnWhat.LAST_ASSISTANT_MESSAGE:
        return {max(assistant_indices)} if assistant_indices else set()
    if train_on_what == TrainOnWhat.CUSTOMIZED:
        return {i for i in assistant_indices if messages[i].get("trainable", True)}
    raise NotImplementedError(
        f"tml-renderers SFT supports {TrainOnWhat.ALL_ASSISTANT_MESSAGES.value}, "
        f"{TrainOnWhat.LAST_ASSISTANT_MESSAGE.value}, and {TrainOnWhat.CUSTOMIZED.value}; "
        f"got {train_on_what.value!r}"
    )


def tinker_messages_to_sft_message_list(
    messages: Sequence[Message], train_on_what: TrainOnWhat
) -> tml_chat.MessageList:
    """Convert tinker-cookbook Messages to a ``MessageList`` for ``render_for_sft``.

    tml-renderers trains every model Message unless its weight is 0. So each assistant turn
    that ``train_on_what`` does not select gets weight 0 on its model Messages. Under
    ``CUSTOMIZED``, an assistant message without ``trainable`` is trained.
    """
    chat = _chat()
    trained = _trained_assistant_indices(messages, train_on_what)
    zero = chat.TrainingMetadata(0.0, False)
    parts: list[tml_chat.Message] = []
    for i, message in enumerate(messages):
        turn = _tml_renderers_openai_message(message).to_messages()
        if message["role"] == "assistant" and i not in trained:
            turn = [
                part.copy(message_metadata=chat.MessageMetadata(training_metadata=zero))
                if part.author.kind == chat.AuthorKind.Model
                else part
                for part in turn
            ]
        parts += turn
    return chat.MessageList(parts)


def _tinker_part(content: Any) -> ContentPart:
    chat = _chat()
    if isinstance(content, chat.Text | chat.ToolError):
        return {"type": "text", "text": content.text}
    if isinstance(content, chat.Thinking):
        return {"type": "thinking", "thinking": content.text}
    if isinstance(content, chat.ImagePointer):
        return {"type": "image", "image": content.location}
    if isinstance(content, chat.AudioPointer):
        formats: dict[Any, Literal["wav", "mp3", "flac"]] = {
            chat.AudioFormat.Wav: "wav",
            chat.AudioFormat.Mp3: "mp3",
            chat.AudioFormat.Flac: "flac",
        }
        audio_format = formats[content.format]
        part: AudioPart = {"type": "audio", "audio": content.location, "format": audio_format}
        # Only an inline WAV gives its frame counts back from its header.
        if audio_format != "wav" or not content.location.startswith("data:"):
            part["num_frames"] = content.num_frames
            part["sample_rate"] = content.sample_rate
        return part
    raise ValueError(f"no tinker-cookbook content part for {type(content).__name__}")


def _tool_call(invoke: tml_chat.InvokeTool) -> ToolCall:
    if invoke.text is not None:
        # tml-renderers' writers refuse a text tool call too: it has no name and arguments.
        raise ValueError("a text tool call has no tinker-cookbook form")
    arguments = {arg.name: json.loads(arg.value) for arg in invoke.args}
    return ToolCall(
        id=invoke.tool_call_id,
        function=ToolCall.FunctionBody(name=invoke.name, arguments=json.dumps(arguments)),
    )


def _tinker_message(turn: tml_chat.OpenAIMessage) -> Message:
    chat = _chat()
    match turn.content:
        case [chat.ToolDeclareJson() as declare]:
            tools = [
                {
                    "type": s.type_name,
                    "function": {
                        "name": s.name,
                        "description": s.description,
                        "parameters": json.loads(s.parameters),
                    },
                }
                for s in declare.tool_specs
            ]
            return Message(role="tool_declare", content=json.dumps(tools, separators=(",", ":")))
    roles = {
        chat.AuthorKind.User: "user",
        chat.AuthorKind.Model: "assistant",
        chat.AuthorKind.System: "system",
        chat.AuthorKind.Tool: "tool",
    }
    calls = [
        _tool_call(cast("tml_chat.InvokeTool", c))
        for c in turn.content
        if isinstance(c, chat.InvokeTool)
    ]
    parts = [_tinker_part(c) for c in turn.content if not isinstance(c, chat.InvokeTool)]
    message = Message(
        role=roles[turn.author.kind],
        content=parts[0]["text"] if len(parts) == 1 and parts[0]["type"] == "text" else parts,
    )
    if calls:
        message["tool_calls"] = calls
    if turn.tool_call_id is not None:
        message["tool_call_id"] = turn.tool_call_id
    if turn.tool_name is not None:
        message["name"] = turn.tool_name
    return message


def message_list_to_tinker_messages(messages: tml_chat.MessageList) -> list[Message]:
    """Convert a ``tml_renderers.chat.MessageList`` to tinker-cookbook Messages.

    ``OpenAIMessage.from_messages`` groups the Messages into turns, and each turn becomes one
    Message. Consecutive Messages from one author become one Message. The Messages hold
    what tml-renderers stores: images and audio as data URIs, tool parameters with sorted
    keys, and tool-call arguments in ``json.dumps`` form. The conversion does not read
    training weights back.

    Raises ValueError for a text tool call or content with no tinker-cookbook part.
    """
    chat = _chat()
    return [_tinker_message(turn) for turn in chat.OpenAIMessage.from_messages(messages.messages)]
