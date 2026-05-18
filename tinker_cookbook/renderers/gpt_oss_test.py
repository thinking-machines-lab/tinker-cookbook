"""Tests specific to GptOss renderer (parse_response, tool calls, channel parsing)."""

from tinker_cookbook.renderers import (
    TextPart,
    ThinkingPart,
)
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

# =============================================================================
# GptOss parse_response Tests
# =============================================================================


def test_gptoss_parse_response_extracts_thinking():
    """Test GptOssRenderer.parse_response extracts analysis channel as thinking."""
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")

    # GptOss format: analysis channel then final channel
    response_str = "<|channel|>analysis<|message|>Let me think about this.<|end|><|start|>assistant<|channel|>final<|message|>The answer is 42.<|return|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success.is_clean
    content = message["content"]
    assert isinstance(content, list)

    thinking_parts = [p for p in content if p["type"] == "thinking"]
    text_parts = [p for p in content if p["type"] == "text"]

    assert len(thinking_parts) == 1
    assert thinking_parts[0]["thinking"] == "Let me think about this."
    assert len(text_parts) == 1
    assert text_parts[0]["text"] == "The answer is 42."


def test_gptoss_parse_response_multiple_analysis():
    """Test GptOssRenderer.parse_response handles multiple analysis messages."""
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")

    response_str = "<|channel|>analysis<|message|>First thought.<|end|><|start|>assistant<|channel|>analysis<|message|>Second thought.<|end|><|start|>assistant<|channel|>final<|message|>Done.<|return|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success.is_clean
    content = message["content"]
    assert isinstance(content, list)
    assert len(content) == 3

    assert content[0] == ThinkingPart(type="thinking", thinking="First thought.")
    assert content[1] == ThinkingPart(type="thinking", thinking="Second thought.")
    assert content[2] == TextPart(type="text", text="Done.")


def test_gptoss_parse_response_final_only():
    """Test GptOssRenderer.parse_response with only final channel."""
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")

    response_str = "<|channel|>final<|message|>Simple answer.<|return|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success.is_clean
    content = message["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0] == TextPart(type="text", text="Simple answer.")


def test_gptoss_parse_response_no_channels():
    """Test GptOssRenderer.parse_response returns string when no channel markers."""
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")

    response_str = "Plain response without channels.<|return|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success.is_clean
    # No channel markers, so content stays as string
    assert isinstance(message["content"], str)
    assert message["content"] == "Plain response without channels."


def test_gptoss_parse_response_tool_call():
    """Test GptOssRenderer.parse_response extracts tool calls from commentary channel."""
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")

    # Tool call format: commentary channel with to=functions.name and <|call|> stop token
    response_str = '<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location": "San Francisco"}<|call|>'
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success.is_clean
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "get_weather"
    assert '"location"' in message["tool_calls"][0].function.arguments


def test_gptoss_parse_response_tool_call_with_analysis():
    """Test GptOssRenderer.parse_response extracts both thinking and tool calls."""
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")

    response_str = '<|channel|>analysis<|message|>I need to check the weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city": "NYC"}<|call|>'
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success.is_clean
    content = message["content"]
    assert isinstance(content, list)

    # Should have thinking from analysis channel
    thinking_parts = [p for p in content if p["type"] == "thinking"]
    assert len(thinking_parts) >= 1
    assert "check the weather" in thinking_parts[0]["thinking"]

    # Should have tool call
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "get_weather"


def test_gptoss_parse_response_invalid_tool_call_json():
    """Test GptOssRenderer.parse_response handles invalid JSON in tool calls."""
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")

    response_str = "<|channel|>commentary to=functions.broken <|constrain|>json<|message|>not valid json<|call|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success.is_clean
    assert "unparsed_tool_calls" in message
    assert len(message["unparsed_tool_calls"]) == 1
    assert "Invalid JSON" in message["unparsed_tool_calls"][0].error


def test_gptoss_parse_response_tool_call_recipient_before_channel():
    """Test GptOssRenderer.parse_response handles recipient before channel."""
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")

    response_str = '<|start|>assistant to=functions.get_weather<|channel|>commentary<|constrain|>json<|message|>{"location": "Tokyo"}<|call|>'
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success.is_clean
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "get_weather"


def test_gptoss_parse_response_commentary_preamble():
    """Test GptOssRenderer.parse_response keeps commentary preamble text."""
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")

    response_str = (
        "<|channel|>commentary<|message|>Checking now.<|end|>"
        '<|start|>assistant to=functions.get_weather<|channel|>commentary <|constrain|>json<|message|>{"location": "SF"}<|call|>'
    )
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success.is_clean
    content = message["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0] == TextPart(type="text", text="Checking now.")
    assert "tool_calls" in message and len(message["tool_calls"]) == 1
