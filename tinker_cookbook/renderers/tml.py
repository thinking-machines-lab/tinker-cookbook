from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import cast

import tinker
import torch
from tml_renderers import chat as tml_chat
from tml_renderers.renderer import Parser as PublicParser
from tml_renderers.renderer import Renderer as PublicRenderer
from tml_renderers.tinker import (
    token_spans_to_tinker_model_input,
    training_example_to_tinker_model_input_and_weights,
)

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
from tinker_cookbook.tokenizer_utils import TmlRenderersTokenizerAdapter


class TmlRendererAdapter(Renderer):
    supports_streaming = True

    def __init__(self, renderer: PublicRenderer):
        self._tml_renderer = renderer
        self._pending_parser: PublicParser | None = None
        self._pending_parser_factory: Callable[[], PublicParser] | None = None
        super().__init__(TmlRenderersTokenizerAdapter.from_tokenizer(renderer.tokenizer))

    def __reduce__(self) -> tuple:
        if self._pending_parser is not None:
            raise RuntimeError("cannot pickle a TML renderer adapter with an unparsed completion")
        return super().__reduce__()

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        del message, ctx
        raise NotImplementedError(
            "TML renderer adapters render complete conversations, not context-free messages"
        )

    def _take_parser(self) -> PublicParser:
        parser = self._pending_parser
        self._pending_parser = None
        self._pending_parser_factory = None
        if parser is None:
            raise RuntimeError("render a completion prompt before parsing its response")
        return parser

    def _take_parsers(self, count: int) -> list[PublicParser]:
        if count == 0:
            return []
        parser = self._pending_parser
        parser_factory = self._pending_parser_factory
        self._pending_parser = None
        self._pending_parser_factory = None
        if parser is None or parser_factory is None:
            raise RuntimeError("render a completion prompt before parsing its responses")
        try:
            parsers = [parser]
            parsers.extend(parser_factory() for _ in range(count - 1))
            return parsers
        except ValueError as error:
            raise RendererError(str(error)) from error

    @staticmethod
    def _validate_generation_options(role: Role, prefill: str | None) -> None:
        if role != "assistant":
            raise NotImplementedError("TML renderers only support assistant generation")
        if prefill:
            raise NotImplementedError("TML renderers do not support caller-provided prefill")

    def _build_generation_prompt(
        self,
        messages: list[Message] | TmlRenderInput,
        role: Role,
        prefill: str | None,
        render_for_completion: Callable[
            [TmlRenderInput], tuple[list[tml_chat.TokenSpan], PublicParser]
        ],
    ) -> tinker.ModelInput:
        self._validate_generation_options(role, prefill)
        if self._pending_parser is not None:
            raise RuntimeError("parse the pending completion before rendering another prompt")
        try:
            render_input = _messages_to_render_input(messages)
            spans, self._pending_parser = render_for_completion(render_input)
            self._pending_parser_factory = lambda: render_for_completion(render_input)[1]
        except ValueError as error:
            raise RendererError(str(error)) from error
        return token_spans_to_tinker_model_input(spans)

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        return self._build_generation_prompt(
            messages,
            role,
            prefill,
            self._tml_renderer.render_for_completion,
        )

    def _build_supervised_examples(
        self, render_input: TmlRenderInput
    ) -> list[tuple[tinker.ModelInput, torch.Tensor]]:
        try:
            examples = self._tml_renderer.render_for_sft(render_input)
        except ValueError as error:
            raise RendererError(str(error)) from error
        converted = (
            training_example_to_tinker_model_input_and_weights(example) for example in examples
        )
        return [
            (model_input, torch.tensor(weights, dtype=torch.float32))
            for model_input, weights in converted
        ]

    def build_supervised_examples(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    ) -> list[tuple[tinker.ModelInput, torch.Tensor]]:
        render_input = cast(
            TmlRenderInput, _cookbook_messages_to_sft_input(messages, train_on_what)
        )
        return self._build_supervised_examples(render_input)

    @staticmethod
    def _single_supervised_example(
        examples: list[tuple[tinker.ModelInput, torch.Tensor]],
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        if len(examples) != 1:
            raise NotImplementedError(
                "TML renderer produced multiple SFT examples; use build_supervised_examples"
            )
        return examples[0]

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        return self._single_supervised_example(
            self.build_supervised_examples(messages, train_on_what)
        )

    def get_stop_sequences(self) -> list[int] | list[str]:
        stop = self._tml_renderer.stop()
        if stop is None:
            return []
        if isinstance(stop, str):
            return [stop]
        return stop

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

    def _parse_response(
        self, parser: PublicParser, response: list[int]
    ) -> tuple[Message, ParseTermination]:
        try:
            parsed = parser.parse_tokens(response)
        except ValueError:
            return (
                Message(role="assistant", content=self._decode_or_empty(response)),
                ParseTermination.MALFORMED,
            )
        content = [
            message
            for message in parsed
            if not isinstance(message.content, tml_chat.ModelEndSampling)
        ]
        saw_stop = len(content) != len(parsed)
        message = _parsed_messages_to_cookbook(content) if content else None
        return (
            message or Message(role="assistant", content=self._decode_or_empty(response)),
            ParseTermination.STOP_SEQUENCE if saw_stop else ParseTermination.MALFORMED,
        )

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        return self._parse_response(self._take_parser(), response)

    def parse_responses(
        self, responses: list[list[int]]
    ) -> list[tuple[Message, ParseTermination]]:
        return [
            self._parse_response(parser, response)
            for parser, response in zip(self._take_parsers(len(responses)), responses, strict=True)
        ]

    def parse_response_streaming(self, response: list[int]) -> Iterator[MessageDelta]:
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
            if isinstance(event, tml_chat.StreamingMessageHeader):
                content_index += 1
                if not emitted_header:
                    emitted_header = True
                    yield StreamingMessageHeader(role="assistant")
            elif isinstance(event, tml_chat.StreamingContent):
                if isinstance(event.content, tml_chat.Thinking):
                    yield StreamingThinkingDelta(
                        thinking=event.content.text,
                        content_index=max(content_index, 0),
                    )
                else:
                    yield StreamingTextDelta(
                        text=event.content.text,
                        content_index=max(content_index, 0),
                    )
            elif not isinstance(event.content, tml_chat.ModelEndSampling):
                parsed.append(event)

        if not emitted_header:
            yield StreamingMessageHeader(role="assistant")
        yield _parsed_messages_to_cookbook(parsed) or Message(
            role="assistant", content=self._decode_or_empty(response)
        )


__all__ = ["TmlRendererAdapter"]
