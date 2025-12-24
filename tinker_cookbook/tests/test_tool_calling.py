"""
Tests for tool calling bug fixes in renderers.

These tests verify:
1. Tool response messages are correctly rendered (Issue #74)
2. Multiple tool calls are correctly parsed
3. Tool call content is stripped from parsed messages
4. Kimi K2 tool name extraction works correctly
"""

import pytest
import tinker

from tinker_cookbook.renderers import Message, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer


# =============================================================================
# Tool Response Rendering Tests
# =============================================================================


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [
        ("Qwen/Qwen3-8B", "qwen3"),
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", "qwen3_instruct"),
    ],
)
def test_qwen3_tool_response_rendering(model_name: str, renderer_name: str):
    """Test that Qwen3 renders tool responses with user role and tool_response tags.

    This verifies the fix for Issue #74: tool messages should render as
    <|im_start|>user with content wrapped in <tool_response> tags.
    """
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    tool_message: Message = {"role": "tool", "content": '{"weather": "sunny", "high": 72}'}

    rendered = renderer.render_message(0, tool_message)
    prefix = rendered.get("prefix")
    assert prefix is not None, "Expected prefix in rendered message"
    content = rendered["content"]
    assert len(content) > 0, "Expected content in rendered message"

    prefix_str = tokenizer.decode(list(prefix.tokens))
    # Content[0] is an EncodedTextChunk for text-only messages
    content_chunk = content[0]
    assert isinstance(content_chunk, tinker.EncodedTextChunk), "Expected EncodedTextChunk"
    content_str = tokenizer.decode(list(content_chunk.tokens))

    # Tool messages should be rendered as "user" role
    assert "<|im_start|>user" in prefix_str
    # Content should be wrapped in tool_response tags
    assert "<tool_response>" in content_str
    assert "</tool_response>" in content_str
    assert '"weather": "sunny"' in content_str


# =============================================================================
# Tool Call Parsing Tests
# =============================================================================


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [
        ("Qwen/Qwen3-8B", "qwen3"),
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", "qwen3_instruct"),
    ],
)
def test_qwen3_parse_single_tool_call(model_name: str, renderer_name: str):
    """Test parsing a single tool call from Qwen3 response."""
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # Simulate model response with tool call
    response_text = """I'll search for that information.
<tool_call>
{"name": "search", "args": {"query": "weather in NYC"}}
</tool_call><|im_end|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    assert message["role"] == "assistant"
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "search"
    # Content should have tool_call block stripped
    assert "<tool_call>" not in message["content"]


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [
        ("Qwen/Qwen3-8B", "qwen3"),
    ],
)
def test_qwen3_parse_multiple_tool_calls(model_name: str, renderer_name: str):
    """Test parsing multiple tool calls from Qwen3 response.

    This verifies the fix for the bug where only the first tool call was parsed.
    """
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # Simulate model response with multiple tool calls
    response_text = """I'll get the weather for both cities.
<tool_call>
{"name": "get_weather", "args": {"location": "NYC"}}
</tool_call>
<tool_call>
{"name": "get_weather", "args": {"location": "LA"}}
</tool_call><|im_end|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 2
    assert message["tool_calls"][0].function.name == "get_weather"
    assert message["tool_calls"][1].function.name == "get_weather"
    # Verify different arguments
    assert "NYC" in message["tool_calls"][0].function.arguments
    assert "LA" in message["tool_calls"][1].function.arguments


def test_kimi_k2_parse_tool_call():
    """Test parsing tool call from Kimi K2 response.

    This verifies the fix for extracting function name from tool_id format
    "functions.{name}:{idx}".
    """
    model_name = "moonshotai/Kimi-K2-Thinking"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("kimi_k2", tokenizer)

    # Simulate model response with tool call (Kimi K2 format)
    response_text = """<think></think>I'll search for that.
<|tool_calls_section_begin|><|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{"query": "weather NYC"}<|tool_call_end|><|tool_calls_section_end|><|im_end|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    # Verify function name is extracted from tool_id
    assert message["tool_calls"][0].function.name == "search"
    assert message["tool_calls"][0].id == "functions.search:0"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


def test_qwen3_parse_invalid_tool_call_json():
    """Test that invalid JSON in tool call returns parse failure."""
    model_name = "Qwen/Qwen3-8B"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("qwen3", tokenizer)

    # Invalid JSON in tool call
    response_text = """<tool_call>
{invalid json here}
</tool_call><|im_end|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is False


def test_qwen3_parse_no_tool_call():
    """Test parsing response without tool calls."""
    model_name = "Qwen/Qwen3-8B"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("qwen3", tokenizer)

    response_text = "I don't need to use any tools for this.<|im_end|>"
    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    assert "tool_calls" not in message
    assert message["content"] == "I don't need to use any tools for this."
