"""Cookbook renderer backed by the public ``tml_renderers.v0`` renderer.

This is intentionally a thin integration layer. ``tml_renderers`` owns TMLv0
framing and unshifted SFT masks; cookbook owns final Datum construction through
``datum_from_model_input_weights``.
"""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, TypeGuard, cast
from urllib.parse import unquote, urlparse

import tinker
import torch

from tinker_cookbook.image_processing_utils import image_to_data_uri
from tinker_cookbook.renderers.base import (
    AudioPart,
    ImagePart,
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
)
from tinker_cookbook.third_party.openai_compat import tool_specs_to_openai_tools
from tinker_cookbook.tokenizer_utils import (
    SupportsTmlTokenizer,
    TmlTokenizer,
    Tokenizer,
    ensure_tml_renderers_importable,
)

if TYPE_CHECKING:
    # tml_renderers is an optional dependency (the `inkling` extra) imported
    # lazily at runtime; import it here for annotations only. It ships py.typed
    # stubs, so pyright checks these types whenever the package is installed.
    from tml_renderers import chat as tml_chat  # pyright: ignore[reportMissingImports]
    from tml_renderers import v0 as tml_v0  # pyright: ignore[reportMissingImports]

# What tml_renderers.v0.Renderer accepts as conversation input.
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

_MINIMUM_TORCH_VERSION = (2, 10)

# Effort values must be in [0.0, 1.0).
DEFAULT_EFFORT: float = 0.9


def _validate_torch_version() -> None:
    try:
        major, minor = (int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"TmlV0Renderer could not determine the installed PyTorch version: "
            f"{torch.__version__!r}"
        ) from exc

    if (major, minor) < _MINIMUM_TORCH_VERSION:
        raise RuntimeError(
            f"TmlV0Renderer requires PyTorch 2.10 or newer; found {torch.__version__}. "
            "Install a supported version via the inkling extra "
            "(uv pip install 'tinker-cookbook[inkling]') or directly with "
            'pip install "torch>=2.10".'
        )


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


def _validate_effort(effort: float) -> None:
    if not math.isfinite(effort) or not 0.0 <= effort < 1.0:
        raise ValueError(f"thinking effort must be a finite number in [0, 1), got {effort}")


def _ensure_model_end_sampling(
    messages: list[tml_chat.Message],
) -> list[tml_chat.Message]:
    """Terminate every model turn with ``ModelEndSampling``.

    ``ModelEndSampling`` is the stop-token supervision: without it the model
    never learns where to end its turn, and rendering silently drops that
    weighted token. ``from_oss_messages`` adds the boundary for dict inputs;
    this makes native-message SFT input behave the same. Idempotent, so
    callers that already include the boundary are unchanged.
    """
    chat = import_module("tml_renderers.chat")
    result: list[tml_chat.Message] = []
    for i, message in enumerate(messages):
        result.append(message)
        if message.author.kind != chat.AuthorKind.Model or isinstance(
            message.content, chat.ModelEndSampling
        ):
            continue
        # A model turn can span several messages (thinking, text, tool calls);
        # only close the turn when the run of model messages ends.
        next_message = messages[i + 1] if i + 1 < len(messages) else None
        if next_message is None or next_message.author.kind != chat.AuthorKind.Model:
            result.append(
                chat.Message(
                    content=chat.ModelEndSampling(),
                    author=chat.Author(chat.AuthorKind.Model),
                )
            )
    return result


def _prepare_sft_input(messages: TmlRenderInput, effort: float) -> list[tml_chat.Message]:
    """Normalize SFT input to native messages ready for ``render_for_sft``.

    Expands ``OpenAIMessage`` inputs, terminates model turns with
    ``ModelEndSampling``, and inserts the same effort message used by
    completion rendering.
    """
    _validate_effort(effort)
    chat = import_module("tml_renderers.chat")

    # isinstance against the lazily imported classes can't narrow for pyright,
    # so the checked branches restate the types with casts.
    source: Sequence[tml_chat.Message | tml_chat.OpenAIMessage]
    if isinstance(messages, chat.MessageList):
        source = cast("tml_chat.MessageList", messages).messages
    else:
        source = cast("Sequence[tml_chat.Message | tml_chat.OpenAIMessage]", messages)
    native_messages: list[tml_chat.Message] = []
    for message in source:
        if isinstance(message, chat.OpenAIMessage):
            native_messages.extend(cast("tml_chat.OpenAIMessage", message).to_messages())
        else:
            native_messages.append(cast("tml_chat.Message", message))
    native_messages = _ensure_model_end_sampling(native_messages)

    # ThinkingEffort stores thousandths; tml-renderers owns its display rounding.
    effort_message = chat.Message(
        content=chat.ThinkingEffort(round(effort * 1000)),
        author=chat.Author(chat.AuthorKind.System),
    )
    insertion_index = 0
    while (
        insertion_index < len(native_messages)
        and native_messages[insertion_index].author.kind == chat.AuthorKind.System
    ):
        insertion_index += 1
    native_messages.insert(insertion_index, effort_message)
    return native_messages


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


def _unwrap_tml_tokenizer(tokenizer: Tokenizer) -> TmlTokenizer:
    if isinstance(tokenizer, SupportsTmlTokenizer):
        return tokenizer.tml_tokenizer
    raise TypeError(
        "TmlV0Renderer requires the TML tokenizer adapter. "
        "Use get_tokenizer('thinkingmachines/Inkling') or another "
        "tml-renderers-backed model name."
    )


class TmlV0Renderer(Renderer):
    """Renderer adapter for Inkling models."""

    supports_streaming = False

    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer)
        _validate_torch_version()
        ensure_tml_renderers_importable()
        self._tml_tokenizer = _unwrap_tml_tokenizer(tokenizer)
        self._tml_renderer: tml_v0.Renderer = import_module("tml_renderers.v0").Renderer(
            self._tml_tokenizer
        )

    @property
    def has_extension_property(self) -> bool:
        """TMLv0 frames each message independently, so nothing is stripped or re-headered
        by position. Shorter prompts stay token-prefixes of longer ones.
        """
        return True

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        raise NotImplementedError(
            "TmlV0Renderer renders whole conversations through tml_renderers.v0.Renderer"
        )

    @staticmethod
    def _validate_generation_options(role: Role, prefill: str | None) -> None:
        if role != "assistant":
            raise NotImplementedError("tml_v0 only supports assistant generation")
        if prefill is not None:
            raise NotImplementedError(
                "TMLv0 sampling does not accept partial assistant messages. "
                "Pass complete messages and let the model start a new assistant response."
            )

    @staticmethod
    def _token_spans_to_model_input(spans: list[tml_chat.TokenSpan]) -> tinker.ModelInput:
        return import_module("tml_renderers.tinker").token_spans_to_tinker_model_input(spans)

    def build_generation_prompt(
        self,
        messages: list[Message] | TmlRenderInput,
        role: Role = "assistant",
        prefill: str | None = None,
        effort: float = DEFAULT_EFFORT,
    ) -> tinker.ModelInput:
        """Build a generation prompt with reasoning-effort conditioning.

        ``effort`` must be a finite value in ``[0.0, 1.0)`` and defaults to
        high. Insertion of the system-level effort directive is delegated to
        ``tml-renderers``.
        """
        self._validate_generation_options(role, prefill)
        _validate_effort(effort)
        render_input = _messages_to_render_input(messages)
        spans, _parser = self._tml_renderer.render_for_completion_with_effort(render_input, effort)
        return self._token_spans_to_model_input(spans)

    def _render_sft_examples(
        self, render_input: TmlRenderInput
    ) -> list[tuple[tinker.ModelInput, torch.Tensor]]:
        tml_tinker = import_module("tml_renderers.tinker")
        examples = self._tml_renderer.render_for_sft(render_input)
        rendered: list[tuple[tinker.ModelInput, torch.Tensor]] = []
        for example in examples:
            model_input, weights = tml_tinker.training_example_to_tinker_model_input_and_weights(
                example
            )
            rendered.append((model_input, torch.tensor(weights, dtype=torch.float32)))
        return rendered

    def build_supervised_examples(
        self,
        messages: list[Message] | TmlRenderInput,
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        effort: float = DEFAULT_EFFORT,
    ) -> list[tuple[tinker.ModelInput, torch.Tensor]]:
        """Build SFT examples with the same effort conditioning used for generation.

        The inserted effort message is token-identical to the one
        ``build_generation_prompt`` renders, so supervised data matches sampling.

        Generic supervised dataset builders currently use the default effort
        (``0.9``). We plan to expose per-example effort through those builders;
        until then, call this method directly to render a conversation at a
        specific effort level.
        """
        render_input = _cookbook_messages_to_sft_input(messages, train_on_what)
        return self._render_sft_examples(_prepare_sft_input(render_input, effort))

    def build_supervised_example(
        self,
        messages: list[Message] | TmlRenderInput,
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        effort: float = DEFAULT_EFFORT,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        examples = self.build_supervised_examples(messages, train_on_what, effort=effort)
        return self._single_example(examples)

    @staticmethod
    def _single_example(
        examples: list[tuple[tinker.ModelInput, torch.Tensor]],
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        if len(examples) != 1:
            raise NotImplementedError(
                "tml_v0 produced multiple SFT examples; use build_supervised_examples"
            )
        return examples[0]

    def get_stop_sequences(self) -> list[int]:
        return self._tml_renderer.stop()

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        prefix: list[Message] = []
        if system_prompt:
            prefix.append(Message(role="system", content=system_prompt))
        if tools:
            prefix.append(
                Message(
                    role="tool_declare",
                    content=json.dumps(
                        tool_specs_to_openai_tools(tools),
                        separators=(",", ":"),
                    ),
                )
            )
        return prefix

    def _decode_or_empty(self, response: list[int]) -> str:
        try:
            decoded = self.tokenizer.decode(response)
        except Exception:
            return ""
        return decoded if isinstance(decoded, str) else "".join(decoded)

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        # TMLv0 sampling emits its own message header, so parse the response as-is.
        _spans, parser = self._tml_renderer.render_for_completion([])
        parse_error = import_module("tml_renderers.v0").ParseError
        try:
            parsed = parser.parse_tokens(response)
        except parse_error:
            # The native parser raises ParseError (e.g. JsonIncomplete) when the
            # tokens end mid-structure — typically a max_tokens truncation at an
            # unlucky cut point. The cookbook renderer contract (see qwen3
            # parse_response) is to RETURN a MALFORMED termination, never raise;
            # fall back to the decoded raw text, matching the fallback below.
            return (
                Message(role="assistant", content=self._decode_or_empty(response)),
                ParseTermination.MALFORMED,
            )
        chat = import_module("tml_renderers.chat")
        content_messages = [
            message for message in parsed if not isinstance(message.content, chat.ModelEndSampling)
        ]
        # A dropped ModelEndSampling is the expected stop signal; without it the response
        # was truncated. Any unparseable content falls back to the decoded text.
        saw_stop = len(content_messages) != len(parsed)
        termination = ParseTermination.STOP_SEQUENCE if saw_stop else ParseTermination.MALFORMED

        message = _parsed_messages_to_cookbook(content_messages) if content_messages else None
        if message is None:
            message = Message(role="assistant", content=self._decode_or_empty(response))
        return message, termination
