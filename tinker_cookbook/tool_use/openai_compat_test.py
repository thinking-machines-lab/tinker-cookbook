"""Tests for OpenAI-compatible tool-call interop helpers."""

import json

from tinker_cookbook.renderers.base import Message, ToolCall
from tinker_cookbook.tool_use.openai_compat import (
    linearize_tool_history_for_text_chat,
    normalize_xml_tool_call_message,
    parse_xml_tool_calls,
    strip_xml_thinking,
)


def test_strip_xml_thinking_removes_think_blocks():
    content, thinking = strip_xml_thinking("<think>step one</think>\nFinal answer")

    assert content == "Final answer"
    assert thinking == "step one"


def test_parse_xml_tool_calls_with_json_and_string_parameters():
    content = """
<tool_call>
<function=search>
<parameter=query>
weather in NYC
</parameter>
<parameter=limit>
3
</parameter>
</function>
</tool_call>
"""

    tool_calls, unparsed = parse_xml_tool_calls(content, allowed_tool_names={"search"})

    assert unparsed == []
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "search"
    assert tool_calls[0].id is not None
    assert json.loads(tool_calls[0].function.arguments) == {
        "query": "weather in NYC",
        "limit": 3,
    }


def test_parse_xml_tool_calls_rejects_unallowed_tools():
    content = """
<tool_call>
<function=delete_all>
<parameter=confirm>
true
</parameter>
</function>
</tool_call>
"""

    tool_calls, unparsed = parse_xml_tool_calls(content, allowed_tool_names={"search"})

    assert tool_calls == []
    assert len(unparsed) == 1
    assert "allowed_tool_names" in unparsed[0].error


def test_normalize_xml_tool_call_message_moves_tool_call_out_of_content():
    message: Message = {
        "role": "assistant",
        "content": """
<think>Need to search.</think>
I'll check.
<tool_call>
<function=search>
<parameter=query>
"weather in NYC"
</parameter>
</function>
</tool_call>
""",
    }

    normalized = normalize_xml_tool_call_message(message, allowed_tool_names={"search"})

    assert "tool_calls" in normalized
    assert len(normalized["tool_calls"]) == 1
    assert normalized["tool_calls"][0].function.name == "search"
    assert "<tool_call>" not in str(normalized["content"])
    assert isinstance(normalized["content"], list)
    assert normalized["content"][0]["type"] == "thinking"
    assert normalized["content"][0]["thinking"] == "Need to search."
    assert normalized["content"][1]["type"] == "text"
    assert normalized["content"][1]["text"] == "I'll check."


def test_normalize_xml_tool_call_message_preserves_existing_tool_calls():
    tool_call = ToolCall(function=ToolCall.FunctionBody(name="search", arguments="{}"))
    message: Message = {"role": "assistant", "content": "", "tool_calls": [tool_call]}

    normalized = normalize_xml_tool_call_message(message)

    assert normalized == message


def test_linearize_tool_history_for_text_chat():
    tool_call = ToolCall(
        id="call_1",
        function=ToolCall.FunctionBody(name="search", arguments='{"query": "weather"}'),
    )
    messages: list[Message] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        {"role": "tool", "content": '{"ok": true}', "tool_call_id": "call_1", "name": "search"},
    ]

    converted = linearize_tool_history_for_text_chat(messages)

    assert converted == [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": 'Tool call: search\nArguments: {"query": "weather"}',
        },
        {"role": "assistant", "content": 'Tool result:\n{"ok": true}'},
    ]
