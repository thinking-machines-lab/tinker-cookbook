"""Cookbook ``Renderer`` base for renderers that the ``tml_renderers`` wheel implements."""

from __future__ import annotations

import abc
import json
from collections.abc import Sequence
from importlib import import_module
from typing import TYPE_CHECKING, TypeAlias, TypeGuard, cast

import tinker
import torch

from tinker_cookbook.renderers.base import (
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    ToolSpec,
    TrainOnWhat,
)
from tinker_cookbook.third_party.openai_compat import tool_specs_to_openai_tools
from tinker_cookbook.third_party.tml_renderers_compat import (
    message_list_to_tinker_messages,
    tinker_messages_to_message_list,
    tinker_messages_to_sft_message_list,
)
from tinker_cookbook.tokenizer_utils import (
    SupportsTmlTokenizer,
    TmlTokenizer,
    Tokenizer,
    ensure_tml_renderers_importable,
)

if TYPE_CHECKING:
    # tml_renderers is imported lazily at runtime; import it here for
    # annotations only. It ships py.typed stubs, so pyright checks these types.
    from tml_renderers import chat as tml_chat  # pyright: ignore[reportMissingImports]
    from tml_renderers import renderer as tml_renderer  # pyright: ignore[reportMissingImports]

# What a tml_renderers renderer accepts as conversation input.
TmlRenderInput: TypeAlias = (
    "Sequence[tml_chat.Message] | Sequence[tml_chat.OpenAIMessage] | tml_chat.MessageList"
)

_MINIMUM_TORCH_VERSION = (2, 10)


def _validate_torch_version() -> None:
    try:
        major, minor = (int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"tml-renderers could not determine the installed PyTorch version: "
            f"{torch.__version__!r}"
        ) from exc

    if (major, minor) < _MINIMUM_TORCH_VERSION:
        raise RuntimeError(
            f"tml-renderers requires PyTorch 2.10 or newer; found {torch.__version__}. "
            'Reinstall tinker-cookbook or run `pip install "torch>=2.10"`.'
        )


def _is_tml_renderers_input(messages: object) -> TypeGuard[TmlRenderInput]:
    chat = import_module("tml_renderers.chat")
    if isinstance(messages, chat.MessageList):
        return True
    if isinstance(messages, Sequence) and not isinstance(messages, str):
        return all(isinstance(message, chat.Message | chat.OpenAIMessage) for message in messages)
    return False


def _messages_to_render_input(messages: Sequence[Message] | TmlRenderInput) -> TmlRenderInput:
    if _is_tml_renderers_input(messages):
        return messages
    return tinker_messages_to_message_list(cast("Sequence[Message]", messages))


def _cookbook_messages_to_sft_input(
    messages: Sequence[Message] | TmlRenderInput, train_on_what: TrainOnWhat, renderer_name: str
) -> TmlRenderInput:
    if not _is_tml_renderers_input(messages):
        return tinker_messages_to_sft_message_list(
            cast("Sequence[Message]", messages), train_on_what
        )
    if train_on_what != TrainOnWhat.ALL_ASSISTANT_MESSAGES:
        raise NotImplementedError(
            f"{renderer_name} only supports selective train_on_what modes for cookbook/OpenAI "
            "message dictionaries. Pass train_on_what=ALL_ASSISTANT_MESSAGES when using "
            "native tml_renderers.chat.Message, OpenAIMessage, or MessageList inputs."
        )
    return messages


def _ensure_model_end_sampling(
    messages: list[tml_chat.Message],
) -> list[tml_chat.Message]:
    """Terminate every model turn with ``ModelEndSampling``.

    ``ModelEndSampling`` is the stop-token supervision: without it the model
    never learns where to end its turn, and rendering silently drops that
    weighted token. ``OpenAIMessage.to_messages`` adds the boundary for dict inputs;
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


def _native_sft_messages(messages: TmlRenderInput) -> list[tml_chat.Message]:
    """Expand ``OpenAIMessage`` inputs and terminate model turns with ``ModelEndSampling``."""
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
    return _ensure_model_end_sampling(native_messages)


def _parsed_messages_to_cookbook(parsed: list[tml_chat.Message]) -> Message | None:
    chat = import_module("tml_renderers.chat")
    messages = message_list_to_tinker_messages(chat.MessageList(parsed))
    return messages[-1] if messages else None


def _unwrap_tml_tokenizer(tokenizer: Tokenizer, renderer_class: str) -> TmlTokenizer:
    if isinstance(tokenizer, SupportsTmlTokenizer):
        return tokenizer.tml_tokenizer
    raise TypeError(
        f"{renderer_class} requires the TML tokenizer adapter. "
        "Use get_tokenizer() with a tml-renderers-backed model name, such as "
        "'thinkingmachines/Inkling'."
    )


class TmlRenderersRenderer(Renderer, abc.ABC):
    """Adapts a ``tml_renderers`` renderer to the cookbook ``Renderer`` interface.

    Subclasses build the wheel renderer and name the exceptions its parser raises.
    """

    supports_streaming = False

    # The cookbook renderer name, used in error messages.
    renderer_name: str

    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer)
        _validate_torch_version()
        ensure_tml_renderers_importable()
        self._tml_tokenizer = _unwrap_tml_tokenizer(tokenizer, type(self).__name__)
        self._tml_renderer = self._build_tml_renderer(self._tml_tokenizer)

    @abc.abstractmethod
    def _build_tml_renderer(self, tml_tokenizer: TmlTokenizer) -> tml_renderer.Renderer: ...

    @abc.abstractmethod
    def _parse_error_types(self) -> tuple[type[Exception], ...]: ...

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        raise NotImplementedError(
            f"{type(self).__name__} renders whole conversations through tml_renderers"
        )

    def _validate_generation_options(self, role: Role, prefill: str | None) -> None:
        if role != "assistant":
            raise NotImplementedError(f"{self.renderer_name} only supports assistant generation")
        if prefill is not None:
            raise NotImplementedError(
                f"{self.renderer_name} sampling does not accept partial assistant messages. "
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
    ) -> tinker.ModelInput:
        self._validate_generation_options(role, prefill)
        render_input = _messages_to_render_input(messages)
        spans, _parser = self._tml_renderer.render_for_completion(render_input)
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
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_TURN,
    ) -> list[tuple[tinker.ModelInput, torch.Tensor]]:
        render_input = _cookbook_messages_to_sft_input(messages, train_on_what, self.renderer_name)
        return self._render_sft_examples(_native_sft_messages(render_input))

    def build_supervised_example(
        self,
        messages: list[Message] | TmlRenderInput,
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        return self._single_example(self.build_supervised_examples(messages, train_on_what))

    def _single_example(
        self,
        examples: list[tuple[tinker.ModelInput, torch.Tensor]],
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        if len(examples) != 1:
            raise NotImplementedError(
                f"{self.renderer_name} produced multiple SFT examples; "
                "use build_supervised_examples"
            )
        return examples[0]

    def get_stop_sequences(self) -> list[int]:
        # tml_renderers >= 0.2 types stop() as list[int] | list[str] | str | None for
        # every renderer; the wheel-backed renderers here stop on token ids.
        stop = self._tml_renderer.stop()
        if not isinstance(stop, list) or not all(isinstance(token, int) for token in stop):
            raise TypeError(
                f"Expected the {self.renderer_name} stop() to return list[int], got {stop!r}"
            )
        return cast(list[int], stop)

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

    def _response_stopped(self, response: list[int], saw_end: bool) -> bool:
        """Return whether the model ended its turn, rather than being cut off.

        ``saw_end`` is whether the parser emitted ``ModelEndSampling``. Without it the
        response was truncated.
        """
        return saw_end

    def _completion_parser(self) -> tml_renderer.Parser:
        _spans, parser = self._tml_renderer.render_for_completion([])
        return parser

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        parser = self._completion_parser()
        try:
            parsed = parser.parse_tokens(response)
        except self._parse_error_types():
            # The native parser raises when the tokens end mid-structure — typically a
            # max_tokens truncation at an unlucky cut point. The cookbook renderer
            # contract (see qwen3 parse_response) is to RETURN a MALFORMED termination,
            # never raise; fall back to the decoded raw text, matching the fallback below.
            return (
                Message(role="assistant", content=self._decode_or_empty(response)),
                ParseTermination.MALFORMED,
            )
        chat = import_module("tml_renderers.chat")
        content_messages = [
            message for message in parsed if not isinstance(message.content, chat.ModelEndSampling)
        ]
        saw_end = len(content_messages) != len(parsed)
        termination = (
            ParseTermination.STOP_SEQUENCE
            if self._response_stopped(response, saw_end)
            else ParseTermination.MALFORMED
        )

        try:
            message = _parsed_messages_to_cookbook(content_messages) if content_messages else None
        except ValueError:
            # A text tool call parses but has no tinker-cookbook form.
            return (
                Message(role="assistant", content=self._decode_or_empty(response)),
                ParseTermination.MALFORMED,
            )
        if message is None:
            message = Message(role="assistant", content=self._decode_or_empty(response))
        return message, termination
