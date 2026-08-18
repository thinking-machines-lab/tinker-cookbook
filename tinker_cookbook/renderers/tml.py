from __future__ import annotations

import json
from collections.abc import Iterator
from importlib import import_module
from typing import TYPE_CHECKING, cast

import tinker
import torch

from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.renderers.base import (
    Message,
    MessageDelta,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    StreamingMessageHeader,
    StreamingTextDelta,
    StreamingThinkingDelta,
    ToolSpec,
    TrainOnWhat,
)
from tinker_cookbook.renderers.tml_conversions import (
    TmlRenderInput,
    _cookbook_messages_to_sft_input,
    _messages_to_render_input,
    _parsed_messages_to_cookbook,
)
from tinker_cookbook.third_party.openai_compat import tool_specs_to_openai_tools
from tinker_cookbook.tokenizer_utils import Tokenizer

if TYPE_CHECKING:
    from tml_renderers import chat as tml_chat  # pyright: ignore[reportMissingImports]
    from tml_renderers.renderer import (  # pyright: ignore[reportMissingImports]
        Parser as PublicParser,
    )
    from tml_renderers.renderer import (  # pyright: ignore[reportMissingImports]
        Renderer as PublicRenderer,
    )


class TmlRendererAdapter(Renderer):
    supports_streaming = True

    def __init__(self, renderer: PublicRenderer):
        self._tml_renderer = renderer
        self._pending_parser: PublicParser | None = None
        super().__init__(cast(Tokenizer, renderer.tokenizer))

    def __reduce__(self) -> tuple:
        if self._pending_parser is not None:
            raise RuntimeError("cannot pickle a TML renderer adapter with an unparsed completion")
        return super().__reduce__()

    @property
    def has_extension_property(self) -> bool:
        return bool(getattr(self._tml_renderer, "has_extension_property", False))

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        del message, ctx
        raise NotImplementedError(
            "TML renderer adapters render complete conversations, not context-free messages"
        )

    def _take_parser(self) -> PublicParser:
        parser = self._pending_parser
        self._pending_parser = None
        if parser is None:
            raise RuntimeError("render a completion prompt before parsing its response")
        return parser

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        if role != "assistant":
            raise NotImplementedError("TML renderers only support assistant generation")
        if self._pending_parser is not None:
            raise RuntimeError("parse the pending completion before rendering another prompt")
        try:
            spans, self._pending_parser = self._tml_renderer.render_for_completion(
                _messages_to_render_input(messages)
            )
        except ValueError as error:
            raise RendererError(str(error)) from error
        model_input = import_module("tml_renderers.tinker").token_spans_to_tinker_model_input(spans)
        if not prefill:
            return model_input
        return tinker.ModelInput(
            chunks=[
                *model_input.chunks,
                tinker.EncodedTextChunk(
                    tokens=list(self._tml_renderer.tokenizer.encode_ordinary(prefill))
                ),
            ]
        )

    def build_supervised_examples(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    ) -> list[tuple[tinker.ModelInput, torch.Tensor]]:
        bridge = import_module("tml_renderers.tinker")
        rendered: list[tuple[tinker.ModelInput, torch.Tensor]] = []
        render_input = cast(
            TmlRenderInput,
            _cookbook_messages_to_sft_input(messages, train_on_what),
        )
        try:
            examples = self._tml_renderer.render_for_sft(render_input)
        except ValueError as error:
            raise RendererError(str(error)) from error
        for example in examples:
            model_input, weights = bridge.training_example_to_tinker_model_input_and_weights(
                example
            )
            rendered.append((model_input, torch.tensor(weights, dtype=torch.float32)))
        return rendered

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        bridge = import_module("tml_renderers.tinker")
        render_input = cast(
            TmlRenderInput,
            _cookbook_messages_to_sft_input(messages, train_on_what),
        )
        try:
            examples = self._tml_renderer.render_for_sft(
                render_input,
                split_non_extension_history=False,
            )
        except ValueError as error:
            raise RendererError(str(error)) from error
        if len(examples) != 1:
            raise NotImplementedError(
                "TML renderer produced multiple SFT examples; use build_supervised_examples"
            )
        model_input, weights = bridge.training_example_to_tinker_model_input_and_weights(
            examples[0]
        )
        return model_input, torch.tensor(weights, dtype=torch.float32)

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
                    content=json.dumps(tool_specs_to_openai_tools(tools), separators=(",", ":")),
                )
            )
        return prefix

    def _decode_or_empty(self, response: list[int]) -> str:
        try:
            return self._tml_renderer.tokenizer.decode(response)
        except ValueError:
            return ""

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        chat = import_module("tml_renderers.chat")
        try:
            parsed = self._take_parser().parse_tokens(response)
        except ValueError:
            return (
                Message(role="assistant", content=self._decode_or_empty(response)),
                ParseTermination.MALFORMED,
            )
        content = [
            message for message in parsed if not isinstance(message.content, chat.ModelEndSampling)
        ]
        saw_stop = len(content) != len(parsed)
        message = (
            _parsed_messages_to_cookbook(cast("list[tml_chat.Message]", content))
            if content
            else None
        )
        return (
            message or Message(role="assistant", content=self._decode_or_empty(response)),
            ParseTermination.STOP_SEQUENCE if saw_stop else ParseTermination.MALFORMED,
        )

    def parse_response_streaming(self, response: list[int]) -> Iterator[MessageDelta]:
        chat = import_module("tml_renderers.chat")
        parser = self._take_parser()
        try:
            updates = [update for token in response for update in parser.parse_token(token)]
            updates.extend(parser.flush_updates())
        except ValueError:
            yield StreamingMessageHeader(role="assistant")
            yield Message(role="assistant", content=self._decode_or_empty(response))
            return

        parsed: list[tml_chat.Message] = []
        emitted_header = False
        content_index = -1
        for update in updates:
            event = update.update
            if isinstance(event, chat.StreamingMessageHeader):
                content_index += 1
                if not emitted_header:
                    emitted_header = True
                    yield StreamingMessageHeader(role="assistant")
            elif isinstance(event, chat.StreamingContent):
                if isinstance(event.content, chat.Thinking):
                    yield StreamingThinkingDelta(
                        thinking=event.content.text,
                        content_index=max(content_index, 0),
                    )
                else:
                    yield StreamingTextDelta(
                        text=event.content.text,
                        content_index=max(content_index, 0),
                    )
            elif not isinstance(event.content, chat.ModelEndSampling):
                parsed.append(event)

        if not emitted_header:
            yield StreamingMessageHeader(role="assistant")
        yield _parsed_messages_to_cookbook(parsed) or Message(
            role="assistant", content=self._decode_or_empty(response)
        )


__all__ = ["TmlRendererAdapter"]
