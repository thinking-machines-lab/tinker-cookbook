"""Tests for inspect_utils conversion functions."""

from typing import cast

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai.model import ChatMessage as InspectAIChatMessage
from inspect_ai.model import ChatMessageAssistant as InspectAIChatMessageAssistant
from inspect_ai.model import ChatMessageTool as InspectAIChatMessageTool
from inspect_ai.model import ChatMessageUser as InspectAIChatMessageUser
from inspect_ai.model import ContentReasoning as InspectAIContentReasoning
from inspect_ai.model import ContentText as InspectAIContentText
from inspect_ai.tool import ToolCall as InspectAIToolCall
from inspect_ai.tool import ToolFunction as InspectAIToolFunction
from inspect_ai.tool import ToolInfo as InspectAIToolInfo

from tinker_cookbook import renderers
from tinker_cookbook.eval.inspect_utils import (
    _conversation_with_tool_declarations,
    _message_to_inspect_content,
    _message_to_inspect_tool_calls,
    convert_inspect_messages,
)


class FakeToolRenderer:
    def __init__(self):
        self.received_tools: list[renderers.ToolSpec] | None = None
        self.received_system_prompt: str | None = None

    def create_conversation_prefix_with_tools(
        self, tools: list[renderers.ToolSpec], system_prompt: str = ""
    ) -> list[renderers.Message]:
        self.received_tools = tools
        self.received_system_prompt = system_prompt
        return [
            renderers.Message(role="tool_declare", content="tool specs"),
            renderers.Message(role="system", content=system_prompt or "default system"),
        ]


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


def test_message_to_inspect_tool_calls():
    message = renderers.Message(
        role="assistant",
        content="",
        tool_calls=[
            renderers.ToolCall(
                id="call_123",
                function=renderers.ToolCall.FunctionBody(
                    name="lookup", arguments='{"query":"GDP"}'
                ),
            )
        ],
    )

    result = _message_to_inspect_tool_calls(message, choice_index=0)

    assert result is not None
    assert len(result) == 1
    assert result[0].id == "call_123"
    assert result[0].function == "lookup"
    assert result[0].arguments == {"query": "GDP"}


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


def test_convert_inspect_messages_preserves_tool_calls_and_tool_results():
    messages: list[InspectAIChatMessage] = [
        InspectAIChatMessageAssistant(
            content="",
            tool_calls=[
                InspectAIToolCall(
                    id="call_123",
                    function="lookup",
                    arguments={"query": "GDP"},
                )
            ],
        ),
        InspectAIChatMessageTool(
            content="tool result",
            tool_call_id="call_123",
            function="lookup",
        ),
    ]

    result = convert_inspect_messages(messages)

    assert result[0]["role"] == "assistant"
    tool_calls = result[0].get("tool_calls")
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_123"
    assert tool_calls[0].function.name == "lookup"
    assert tool_calls[0].function.arguments == '{"query":"GDP"}'
    assert result[1]["role"] == "tool"
    assert result[1].get("tool_call_id") == "call_123"
    assert result[1].get("name") == "lookup"


def test_conversation_with_tool_declarations_passes_tools_and_system_prompt():
    fake_renderer = FakeToolRenderer()
    convo = [
        renderers.Message(role="system", content="custom system"),
        renderers.Message(role="user", content="use a tool"),
    ]
    tools = [
        InspectAIToolInfo(
            name="lookup",
            description="Look up a value",
        )
    ]

    result = _conversation_with_tool_declarations(
        cast(renderers.Renderer, fake_renderer), convo, tools, "auto"
    )

    assert fake_renderer.received_tools == [
        {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        }
    ]
    assert fake_renderer.received_system_prompt == "custom system"
    assert [message["role"] for message in result] == ["tool_declare", "system", "user"]


def test_conversation_with_tool_declarations_respects_tool_choice_none():
    fake_renderer = FakeToolRenderer()
    convo = [renderers.Message(role="user", content="do not use tools")]
    tools = [InspectAIToolInfo(name="lookup", description="Look up a value")]

    result = _conversation_with_tool_declarations(
        cast(renderers.Renderer, fake_renderer), convo, tools, "none"
    )

    assert result == convo
    assert fake_renderer.received_tools is None


def test_conversation_with_tool_declarations_filters_specific_tool_choice():
    fake_renderer = FakeToolRenderer()
    convo = [renderers.Message(role="user", content="use the calculator")]
    tools = [
        InspectAIToolInfo(name="lookup", description="Look up a value"),
        InspectAIToolInfo(name="calculate", description="Calculate a value"),
    ]

    _conversation_with_tool_declarations(
        cast(renderers.Renderer, fake_renderer),
        convo,
        tools,
        InspectAIToolFunction(name="calculate"),
    )

    assert fake_renderer.received_tools is not None
    assert [tool["name"] for tool in fake_renderer.received_tools] == ["calculate"]
