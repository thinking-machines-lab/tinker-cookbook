"""Tests for inspect_utils conversion functions."""

from inspect_ai.model import ChatMessage as InspectAIChatMessage
from inspect_ai.model import ChatMessageAssistant as InspectAIChatMessageAssistant
from inspect_ai.model import ChatMessageUser as InspectAIChatMessageUser
from inspect_ai.model import ContentReasoning as InspectAIContentReasoning
from inspect_ai.model import ContentText as InspectAIContentText

from tinker_cookbook import renderers
from tinker_cookbook.eval.inspect_utils import _message_to_inspect_content, convert_inspect_messages

# --- Output: _message_to_inspect_content ---


def test_message_to_inspect_content_with_thinking():
    message = renderers.Message(
        role="assistant",
        content=[
            renderers.ThinkingPart(type="thinking", thinking="let me think"),
            renderers.TextPart(type="text", text="the answer"),
        ],
    )
    result = _message_to_inspect_content(message)
    assert len(result) == 2
    assert isinstance(result[0], InspectAIContentReasoning)
    assert result[0].reasoning == "let me think"
    assert isinstance(result[1], InspectAIContentText)
    assert result[1].text == "the answer"


def test_message_to_inspect_content_string_content():
    message = renderers.Message(role="assistant", content="plain answer")
    result = _message_to_inspect_content(message)
    assert len(result) == 1
    assert isinstance(result[0], InspectAIContentText)
    assert result[0].text == "plain answer"


def test_message_to_inspect_content_text_only_parts():
    message = renderers.Message(
        role="assistant",
        content=[renderers.TextPart(type="text", text="just text")],
    )
    result = _message_to_inspect_content(message)
    assert len(result) == 1
    assert isinstance(result[0], InspectAIContentText)
    assert result[0].text == "just text"


def test_message_to_inspect_content_empty_thinking():
    message = renderers.Message(
        role="assistant",
        content=[
            renderers.ThinkingPart(type="thinking", thinking=""),
            renderers.TextPart(type="text", text="answer"),
        ],
    )
    result = _message_to_inspect_content(message)
    assert len(result) == 2
    assert isinstance(result[0], InspectAIContentReasoning)
    assert result[0].reasoning == ""


# --- Input: convert_inspect_messages ---


def test_convert_inspect_messages_string_content():
    messages: list[InspectAIChatMessage] = [
        InspectAIChatMessageUser(content="hello"),
        InspectAIChatMessageAssistant(content="hi there"),
    ]
    result = convert_inspect_messages(messages)
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "hello"
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == "hi there"


def test_convert_inspect_messages_structured_assistant():
    messages: list[InspectAIChatMessage] = [
        InspectAIChatMessageAssistant(
            content=[
                InspectAIContentReasoning(reasoning="thinking..."),
                InspectAIContentText(text="answer"),
            ]
        ),
    ]
    result = convert_inspect_messages(messages)
    assert len(result) == 1
    content = result[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0]["type"] == "thinking"
    assert content[0]["thinking"] == "thinking..."  # type: ignore[typeddict-item]
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "answer"  # type: ignore[typeddict-item]


def test_convert_inspect_messages_structured_non_assistant_flattens():
    messages: list[InspectAIChatMessage] = [
        InspectAIChatMessageUser(
            content=[
                InspectAIContentText(text="hello"),
                InspectAIContentText(text="world"),
            ]
        ),
    ]
    result = convert_inspect_messages(messages)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "hello world"
