from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, cast

from tml_renderers import chat as tml_chat

from tinker_cookbook.renderers.base import (
    Message,
    ToolCall,
)
from tinker_cookbook.third_party.openai_compat import tinker_messages_to_openai

TmlRenderInput: TypeAlias = (
    Sequence[tml_chat.Message] | Sequence[tml_chat.OpenAIMessage] | tml_chat.MessageList
)


def _native_messages(messages: object) -> list[tml_chat.Message] | None:
    if isinstance(messages, tml_chat.MessageList):
        return messages.messages
    if (
        isinstance(messages, Sequence)
        and messages
        and all(
            isinstance(message, (tml_chat.Message, tml_chat.OpenAIMessage)) for message in messages
        )
    ):
        return tml_chat.MessageList.from_messages(messages).messages
    return None


def _messages_to_render_input(messages: Sequence[Message] | TmlRenderInput) -> TmlRenderInput:
    if (native := _native_messages(messages)) is not None:
        return native
    return tml_chat.MessageList.from_oss_messages(
        tinker_messages_to_openai(cast("Sequence[Message]", messages))
    ).messages


def _cookbook_messages_to_sft_input(
    messages: Sequence[Message], training_mask: Sequence[bool]
) -> list[tml_chat.Message]:
    openai_messages = tml_chat.OpenAIMessage.from_oss_messages(tinker_messages_to_openai(messages))
    zero_training = tml_chat.TrainingMetadata(0.0, False)
    flattened: list[tml_chat.Message] = []
    for should_train, openai_message in zip(training_mask, openai_messages, strict=True):
        rendered_messages = list(openai_message.to_messages())
        if not should_train:
            rendered_messages = [
                (
                    message.copy(
                        message_metadata=(
                            message.message_metadata.copy(training_metadata=zero_training)
                            if message.message_metadata is not None
                            else tml_chat.MessageMetadata(training_metadata=zero_training)
                        )
                    )
                    if message.author.kind == tml_chat.AuthorKind.Model
                    else message
                )
                for message in rendered_messages
            ]
        flattened.extend(rendered_messages)
    return flattened


def _parsed_messages_to_cookbook(parsed: list[tml_chat.Message]) -> Message | None:
    if not parsed:
        return None
    openai_dicts = tml_chat.MessageList(parsed).to_oss_messages()
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
