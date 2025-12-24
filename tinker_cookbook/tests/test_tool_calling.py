"""
Tests for tool calling support across different model families.

These tests verify that:
1. Tool specifications are correctly rendered for each model family
2. Tool calls are correctly parsed from model responses
3. Tool response messages are correctly rendered
4. Multiple tool calls are handled properly
"""

import json
from typing import Any

import pytest

from tinker_cookbook.renderers import (
    Message,
    ToolCall,
    ToolSpec,
    render_tools_for_deepseek_v3,
    render_tools_for_kimi_k2,
    render_tools_for_llama3,
    render_tools_for_qwen3,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_tools() -> list[ToolSpec]:
    """Sample tool specifications for testing."""
    return [
        {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
        {
            "name": "search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    ]


@pytest.fixture
def sample_tool_call() -> ToolCall:
    """Sample tool call for testing."""
    return ToolCall(
        function=ToolCall.FunctionBody(
            name="get_weather",
            arguments='{"location": "San Francisco", "unit": "celsius"}',
        ),
        id="call_123",
    )


# =============================================================================
# Tool Specification Rendering Tests
# =============================================================================


class TestQwen3ToolRendering:
    """Tests for Qwen3 tool specification rendering."""

    def test_render_tools_empty(self):
        """Empty tools list returns empty string."""
        result = render_tools_for_qwen3([])
        assert result == ""

    def test_render_tools_contains_tools_tags(self, sample_tools: list[ToolSpec]):
        """Tool specs are wrapped in <tools> tags."""
        result = render_tools_for_qwen3(sample_tools)
        assert "<tools>" in result
        assert "</tools>" in result

    def test_render_tools_contains_tool_call_example(self, sample_tools: list[ToolSpec]):
        """Output includes tool_call example format."""
        result = render_tools_for_qwen3(sample_tools)
        assert "<tool_call>" in result
        assert "</tool_call>" in result
        assert '{"name": <function-name>, "arguments": <args-json-object>}' in result

    def test_render_tools_contains_all_tools(self, sample_tools: list[ToolSpec]):
        """All tool names appear in the output."""
        result = render_tools_for_qwen3(sample_tools)
        assert "get_weather" in result
        assert "search" in result


class TestLlama3ToolRendering:
    """Tests for Llama3 tool specification rendering."""

    def test_render_tools_empty(self):
        """Empty tools list returns empty string."""
        result = render_tools_for_llama3([])
        assert result == ""

    def test_render_tools_contains_environment_ipython(self, sample_tools: list[ToolSpec]):
        """Output starts with Environment: ipython."""
        result = render_tools_for_llama3(sample_tools)
        assert result.startswith("Environment: ipython")

    def test_render_tools_contains_all_tools(self, sample_tools: list[ToolSpec]):
        """All tool names appear in the output."""
        result = render_tools_for_llama3(sample_tools)
        assert "get_weather" in result
        assert "search" in result


class TestKimiK2ToolRendering:
    """Tests for Kimi K2 tool specification rendering."""

    def test_render_tools_returns_message(self, sample_tools: list[ToolSpec]):
        """Returns a Message dict with tool_declare name."""
        result = render_tools_for_kimi_k2(sample_tools)
        assert result["role"] == "system"
        assert result["name"] == "tool_declare"

    def test_render_tools_content_is_json(self, sample_tools: list[ToolSpec]):
        """Content is valid JSON containing all tools."""
        result = render_tools_for_kimi_k2(sample_tools)
        content = result["content"]
        assert isinstance(content, str)
        parsed = json.loads(content)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "get_weather"


class TestDeepSeekV3ToolRendering:
    """Tests for DeepSeek V3 tool specification rendering."""

    def test_render_tools_empty(self):
        """Empty tools list returns empty string."""
        result = render_tools_for_deepseek_v3([])
        assert result == ""

    def test_render_tools_contains_all_tools(self, sample_tools: list[ToolSpec]):
        """All tool names appear in the output."""
        result = render_tools_for_deepseek_v3(sample_tools)
        assert "get_weather" in result
        assert "search" in result


# =============================================================================
# Tool Response Rendering Tests (Renderer-specific)
# =============================================================================


def _load_tokenizer(model_name: str) -> Any:
    """Load tokenizer with special handling for models that need trust_remote_code."""
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    kwargs: dict[str, Any] = {}
    if model_name == "moonshotai/Kimi-K2-Thinking":
        kwargs["trust_remote_code"] = True
        kwargs["revision"] = "612681931a8c906ddb349f8ad0f582cb552189cd"

    return AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [
        ("Qwen/Qwen3-8B", "qwen3"),
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", "qwen3_instruct"),
    ],
)
def test_qwen3_tool_response_rendering(model_name: str, renderer_name: str):
    """Test that Qwen3 renders tool responses with user role and tool_response tags."""
    from tinker_cookbook.renderers import get_renderer

    tokenizer = _load_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    tool_message: Message = {"role": "tool", "content": '{"temperature": 72}'}

    rendered = renderer.render_message(0, tool_message)
    prefix_str = tokenizer.decode(rendered["prefix"].tokens)
    content_str = tokenizer.decode(rendered["content"][0].tokens)

    # Tool messages should be rendered as "user" role
    assert "<|im_start|>user" in prefix_str
    # Content should be wrapped in tool_response tags
    assert "<tool_response>" in content_str
    assert "</tool_response>" in content_str
    assert '{"temperature": 72}' in content_str


@pytest.mark.parametrize("model_name", ["meta-llama/Llama-3.2-1B-Instruct"])
def test_llama3_tool_response_rendering(model_name: str):
    """Test that Llama3 renders tool responses with ipython role."""
    from tinker_cookbook.renderers import get_renderer

    tokenizer = _load_tokenizer(model_name)
    renderer = get_renderer("llama3", tokenizer)

    tool_message: Message = {"role": "tool", "content": "sunny, 72F"}

    rendered = renderer.render_message(0, tool_message)
    prefix_str = tokenizer.decode(rendered["prefix"].tokens)
    content_str = tokenizer.decode(rendered["content"][0].tokens)

    # Tool messages should be rendered as "ipython" role
    assert "<|start_header_id|>ipython<|end_header_id|>" in prefix_str
    # Content should be wrapped in output JSON
    assert '"output"' in content_str
    assert "sunny, 72F" in content_str


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
    from tinker_cookbook.renderers import get_renderer

    tokenizer = _load_tokenizer(model_name)
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
    """Test parsing multiple tool calls from Qwen3 response."""
    from tinker_cookbook.renderers import get_renderer

    tokenizer = _load_tokenizer(model_name)
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
    """Test parsing tool call from Kimi K2 response."""
    from tinker_cookbook.renderers import get_renderer

    model_name = "moonshotai/Kimi-K2-Thinking"
    tokenizer = _load_tokenizer(model_name)
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
# End-to-end Tool Use Flow Tests
# =============================================================================


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [
        ("Qwen/Qwen3-8B", "qwen3"),
    ],
)
def test_qwen3_full_tool_use_flow(model_name: str, renderer_name: str):
    """Test complete tool use flow: render tools, parse call, render response."""
    from tinker_cookbook.renderers import get_renderer

    tokenizer = _load_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # 1. Create system message with tools
    tools: list[ToolSpec] = [
        {
            "name": "search",
            "description": "Search for information",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]
    tools_text = render_tools_for_qwen3(tools)
    system_content = "You are a helpful assistant." + tools_text

    # 2. Create conversation with tool call
    messages: list[Message] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "Search for Python tutorials"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="search", arguments='{"query": "Python tutorials"}'
                    ),
                    id="call_1",
                )
            ],
        },
        {"role": "tool", "content": '{"results": ["tutorial1", "tutorial2"]}'},
    ]

    # 3. Build generation prompt for next assistant turn
    prompt = renderer.build_generation_prompt(messages)
    prompt_str = tokenizer.decode(prompt.to_ints())

    # Verify the conversation is properly rendered
    assert "<tools>" in prompt_str  # Tools in system message
    assert "<tool_call>" in prompt_str  # Tool call rendered
    assert "<tool_response>" in prompt_str  # Tool response rendered
    assert "Python tutorials" in prompt_str


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


def test_qwen3_parse_invalid_tool_call_json():
    """Test that invalid JSON in tool call returns parse failure."""
    from tinker_cookbook.renderers import get_renderer

    model_name = "Qwen/Qwen3-8B"
    tokenizer = _load_tokenizer(model_name)
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
    from tinker_cookbook.renderers import get_renderer

    model_name = "Qwen/Qwen3-8B"
    tokenizer = _load_tokenizer(model_name)
    renderer = get_renderer("qwen3", tokenizer)

    response_text = "I don't need to use any tools for this.<|im_end|>"
    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    assert "tool_calls" not in message
    assert message["content"] == "I don't need to use any tools for this."
