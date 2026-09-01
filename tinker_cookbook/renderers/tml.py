from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from typing import TypeAlias, cast

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
from tinker_cookbook.third_party.openai_compat import (
    openai_messages_to_tinker,
    tinker_messages_to_openai,
    tool_specs_to_openai_tools,
)
from tinker_cookbook.tokenizer_utils import TmlRenderersTokenizerAdapter

TmlRenderInput: TypeAlias = (
    Sequence[tml_chat.Message] | Sequence[tml_chat.OpenAIMessage] | tml_chat.MessageList
)


class TmlRendererAdapter(Renderer):
    supports_streaming = True

    def __init__(self, renderer: PublicRenderer):
        self._tml_renderer = renderer
        super().__init__(TmlRenderersTokenizerAdapter.from_tokenizer(renderer.tokenizer))

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        del message, ctx
        raise NotImplementedError(
            "TML renderer adapters render complete conversations, not context-free messages"
        )

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
        render_input = self._native_messages(messages)
        if render_input is None:
            render_input = tml_chat.MessageList.from_oss_messages(
                tinker_messages_to_openai(cast("list[Message]", messages))
            ).messages
        try:
            spans, _parser = render_for_completion(render_input)
        except ValueError as error:
            raise RendererError(str(error)) from error
        return token_spans_to_tinker_model_input(spans)

    @staticmethod
    def _native_messages(messages: object) -> list[tml_chat.Message] | None:
        if isinstance(messages, tml_chat.MessageList):
            return messages.messages
        if (
            isinstance(messages, Sequence)
            and messages
            and all(
                isinstance(message, (tml_chat.Message, tml_chat.OpenAIMessage))
                for message in messages
            )
        ):
            return tml_chat.MessageList.from_messages(messages).messages
        return None

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

    def _sft_render_input(
        self,
        messages: list[Message] | TmlRenderInput,
        train_on_what: TrainOnWhat,
    ) -> TmlRenderInput:
        if (native := self._native_messages(messages)) is not None:
            if train_on_what != TrainOnWhat.ALL_ASSISTANT_MESSAGES:
                raise NotImplementedError(
                    "native tml_renderers messages require train_on_what=ALL_ASSISTANT_MESSAGES"
                )
            return native

        supported = {
            TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            TrainOnWhat.CUSTOMIZED,
        }
        if train_on_what not in supported:
            raise NotImplementedError(
                "TML renderer adapters support ALL_ASSISTANT_MESSAGES, "
                "LAST_ASSISTANT_MESSAGE, and CUSTOMIZED; "
                f"got {train_on_what.value!r}"
            )

        cookbook_messages = cast("list[Message]", messages)
        training_mask = [
            should_train
            for _ctx, should_train in self._training_plan(cookbook_messages, train_on_what)
        ]
        if any(
            should_train and message["role"] != "assistant"
            for message, should_train in zip(cookbook_messages, training_mask, strict=True)
        ):
            raise NotImplementedError(
                "TML renderers cannot train non-assistant messages with CUSTOMIZED"
            )
        openai_messages = tml_chat.OpenAIMessage.from_oss_messages(
            tinker_messages_to_openai(cookbook_messages)
        )
        zero_training = tml_chat.TrainingMetadata(0.0, False)
        render_input: list[tml_chat.Message] = []
        for should_train, openai_message in zip(training_mask, openai_messages, strict=True):
            for message in openai_message.to_messages():
                if not should_train and message.author.kind == tml_chat.AuthorKind.Model:
                    metadata = (
                        message.message_metadata.copy(training_metadata=zero_training)
                        if message.message_metadata is not None
                        else tml_chat.MessageMetadata(training_metadata=zero_training)
                    )
                    message = message.copy(message_metadata=metadata)
                render_input.append(message)
        return render_input

    def build_supervised_examples(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    ) -> list[tuple[tinker.ModelInput, torch.Tensor]]:
        return self._build_supervised_examples(self._sft_render_input(messages, train_on_what))

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

    @staticmethod
    def _parsed_message(parsed: list[tml_chat.Message]) -> Message | None:
        if not parsed:
            return None
        openai_messages = tml_chat.MessageList(parsed).to_oss_messages()
        return openai_messages_to_tinker(openai_messages)[-1]

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        try:
            parsed = self._tml_renderer.parser_for_completion().parse_tokens(response)
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
        message = self._parsed_message(content)
        return (
            message or Message(role="assistant", content=self._decode_or_empty(response)),
            ParseTermination.STOP_SEQUENCE if saw_stop else ParseTermination.MALFORMED,
        )

    def parse_response_streaming(self, response: list[int]) -> Iterator[MessageDelta]:
        parser = self._tml_renderer.parser_for_completion()
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
        yield self._parsed_message(parsed) or Message(
            role="assistant", content=self._decode_or_empty(response)
        )


__all__ = ["TmlRendererAdapter"]
