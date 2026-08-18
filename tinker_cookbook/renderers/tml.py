"""Cookbook compatibility for renderers from the public ``tml_renderers`` package."""

from __future__ import annotations

import json
from collections.abc import Iterator
from importlib import import_module
from typing import TYPE_CHECKING, cast

import tinker
import torch

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
from tinker_cookbook.renderers.tml_v0 import (
    TmlRenderInput,
    _cookbook_messages_to_sft_input,
    _messages_to_render_input,
    _parsed_messages_to_cookbook,
)
from tinker_cookbook.third_party.openai_compat import tool_specs_to_openai_tools
from tinker_cookbook.tokenizer_utils import Tokenizer

if TYPE_CHECKING:
    from tml_renderers import Parser as PublicParser  # pyright: ignore[reportMissingImports]
    from tml_renderers import Renderer as PublicRenderer  # pyright: ignore[reportMissingImports]
    from tml_renderers import chat as tml_chat  # pyright: ignore[reportMissingImports]


class TmlRendererAdapter(Renderer):
    """Expose one self-contained ``tml_renderers`` renderer through Cookbook's API.

    The wrapped renderer owns tokenization, rendering, parsing, and SFT masks. This class owns
    only the conversion between Cookbook/OpenAI messages and Tinker transport objects.
    """

    supports_streaming = True

    def __init__(self, renderer: PublicRenderer):
        self._tml_renderer = renderer
        self._pending_parser: PublicParser | None = None
        # Cookbook's base class exposes ``tokenizer`` to legacy callers. It is the wrapped
        # renderer's tokenizer, not a second tokenizer supplied by Cookbook.
        super().__init__(cast(Tokenizer, renderer.tokenizer))

    def __reduce__(self) -> tuple[type[TmlRendererAdapter], tuple[PublicRenderer]]:
        return type(self), (self._tml_renderer,)

    @property
    def has_extension_property(self) -> bool:
        return False

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Provide Cookbook's legacy single-message view from the public renderer."""
        del ctx
        [converted] = cast(
            list[object],
            self._preserve_tool_parameter_order(_messages_to_render_input([message]), [message]),
        )
        header_spans, output_spans = self._tml_renderer.render_message(converted)
        bridge = import_module("tml_renderers.tinker")
        header_input = bridge.token_spans_to_tinker_model_input(header_spans)
        output_input = bridge.token_spans_to_tinker_model_input(output_spans)
        if len(header_input.chunks) != 1 or not isinstance(
            header_input.chunks[0], tinker.EncodedTextChunk
        ):
            raise ValueError("renderer message header must be one encoded-text chunk")
        return RenderedMessage(header=header_input.chunks[0], output=output_input.chunks)

    def _render_completion_input(self, messages: list[Message]) -> tinker.ModelInput:
        spans, self._pending_parser = self._tml_renderer.render_for_completion(
            self._preserve_tool_parameter_order(_messages_to_render_input(messages), messages)
        )
        return import_module("tml_renderers.tinker").token_spans_to_tinker_model_input(spans)

    def _take_parser(self) -> PublicParser:
        parser = self._pending_parser
        self._pending_parser = None
        if parser is None:
            _spans, parser = self._tml_renderer.render_for_completion([])
        return parser

    @staticmethod
    def _preserve_tool_parameter_order(
        render_input: TmlRenderInput,
        cookbook_messages: list[Message],
    ) -> TmlRenderInput:
        """Restore schema key order lost by the canonical public chat conversion."""
        chat = import_module("tml_renderers.chat")
        declaration_type = getattr(chat, "ToolDeclareJson", ())
        raw_declarations = [
            json.loads(message["content"])
            for message in cookbook_messages
            if message["role"] == "tool_declare" and isinstance(message["content"], str)
        ]

        if not isinstance(render_input, list):
            return render_input
        restored: list[object] = []
        declaration_index = 0
        for message in render_input:
            contents = message.content if hasattr(message, "content") else None
            content_list = contents if isinstance(contents, list) else [contents]
            new_contents: list[object] = []
            for content in content_list:
                if content is None or not isinstance(content, declaration_type):
                    new_contents.append(content)
                    continue
                raw_specs = raw_declarations[declaration_index]
                declaration_index += 1
                specs = [
                    spec.copy(
                        parameters=json.dumps(
                            raw["function"].get("parameters", {}),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                    for spec, raw in zip(content.tool_specs, raw_specs, strict=True)
                ]
                new_contents.append(content.copy(tool_specs=specs))
            if isinstance(contents, list):
                message.content = new_contents
                restored.append(message)
            elif contents is not None:
                restored.append(message.copy(content=new_contents[0]))
            else:
                restored.append(message)
        return cast(TmlRenderInput, restored)

    @staticmethod
    def _validate_generation_options(role: Role, prefill: str | None) -> None:
        if role != "assistant":
            raise NotImplementedError("TML renderers only support assistant generation")

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        self._validate_generation_options(role, prefill)
        model_input = self._render_completion_input(messages)
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
        render_input = self._preserve_tool_parameter_order(render_input, messages)
        for example in self._tml_renderer.render_for_sft(render_input):
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
        examples = self.build_supervised_examples(messages, train_on_what)
        if len(examples) != 1:
            raise NotImplementedError(
                "TML renderer produced multiple SFT examples; use build_supervised_examples"
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
