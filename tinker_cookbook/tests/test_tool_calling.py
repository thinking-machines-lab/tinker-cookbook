"""
Tests for tool calling support in renderers.

These tests verify that renderers correctly handle:
1. Tool response message rendering (role mapping and content wrapping)
2. Parsing of single and multiple tool calls from model output
3. Stripping tool call blocks from parsed message content
4. Extracting function names from model-specific tool call formats
"""

import pytest
import tinker

from tinker_cookbook.renderers import Message, RenderContext, get_renderer, get_text_content
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

    Per the Qwen3 chat template, tool messages should render as
    <|im_start|>user with content wrapped in <tool_response> tags.
    """
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    tool_message: Message = {"role": "tool", "content": '{"weather": "sunny", "high": 72}'}

    ctx = RenderContext(idx=0, is_last=False, prev_message=None)
    rendered = renderer.render_message(tool_message, ctx)
    header = rendered.header
    assert header is not None, "Expected header in rendered message"
    output = rendered.output
    assert len(output) > 0, "Expected output in rendered message"

    header_str = tokenizer.decode(list(header.tokens))
    # output[0] is an EncodedTextChunk for text-only messages
    output_chunk = output[0]
    assert isinstance(output_chunk, tinker.EncodedTextChunk), "Expected EncodedTextChunk"
    output_str = tokenizer.decode(list(output_chunk.tokens))

    # Tool messages should be rendered as "user" role
    assert "<|im_start|>user" in header_str
    # Content should be wrapped in tool_response tags
    assert "<tool_response>" in output_str
    assert "</tool_response>" in output_str
    assert '"weather": "sunny"' in output_str


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
{"name": "search", "arguments": {"query": "weather in NYC"}}
</tool_call><|im_end|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    assert message["role"] == "assistant"
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "search"
    # Content should have tool_call block stripped (text content only)
    text_content = get_text_content(message)
    assert "<tool_call>" not in text_content


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [
        ("Qwen/Qwen3-8B", "qwen3"),
    ],
)
def test_qwen3_parse_multiple_tool_calls(model_name: str, renderer_name: str):
    """Test parsing multiple tool calls from Qwen3 response.

    When a model response contains multiple <tool_call> blocks, all should be parsed.
    """
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # Simulate model response with multiple tool calls
    response_text = """I'll get the weather for both cities.
<tool_call>
{"name": "get_weather", "arguments": {"location": "NYC"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"location": "LA"}}
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

    Kimi K2 uses tool_id format "functions.{name}:{idx}", and the function
    name should be extracted correctly.
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


def test_llama3_parse_tool_call():
    """Test parsing tool call from Llama 3 response.

    Llama 3 uses <function=name>{"args"}</function> format.
    """
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("llama3", tokenizer)

    response_text = """I'll get the weather for you.
<function=get_weather>{"location": "San Francisco"}</function><|eot_id|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "get_weather"
    assert "San Francisco" in message["tool_calls"][0].function.arguments
    # Content should have function block stripped (text content only)
    text_content = get_text_content(message)
    assert "<function=" not in text_content


def test_deepseek_parse_tool_call():
    """Test parsing tool call from DeepSeek V3 response.

    DeepSeek V3 HF template format: <｜tool▁call▁begin｜>name<｜tool▁sep｜>args<｜tool▁call▁end｜>
    """
    model_name = "deepseek-ai/DeepSeek-V3.1"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("deepseekv3", tokenizer)

    response_text = """I'll check the weather.
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "NYC"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "get_weather"
    assert "NYC" in message["tool_calls"][0].function.arguments
    # Content should have tool calls section stripped (text content only)
    text_content = get_text_content(message)
    assert "<｜tool▁calls▁begin｜>" not in text_content


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


def test_qwen3_parse_invalid_tool_call_json():
    """Test that invalid JSON in tool call is captured as unparsed_tool_calls."""
    model_name = "Qwen/Qwen3-8B"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("qwen3", tokenizer)

    # Invalid JSON in tool call
    response_text = """<tool_call>
{invalid json here}
</tool_call><|im_end|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    # Parse succeeds, but tool call is captured as unparsed
    assert success is True
    assert "tool_calls" not in message or len(message.get("tool_calls", [])) == 0
    assert "unparsed_tool_calls" in message
    assert len(message["unparsed_tool_calls"]) == 1
    assert "Invalid JSON" in message["unparsed_tool_calls"][0].error
    # Raw text should contain the original tool call
    assert "<tool_call>" in message["unparsed_tool_calls"][0].raw_text


def test_qwen3_mixed_valid_invalid_tool_calls():
    """Test parsing when some tool calls are valid and some are invalid.

    Valid tool calls should be parsed, invalid ones captured in unparsed_tool_calls.
    """
    model_name = "Qwen/Qwen3-8B"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("qwen3", tokenizer)

    # First tool call is valid, second has invalid JSON
    response_text = """I'll try both.
<tool_call>
{"name": "search", "arguments": {"query": "weather"}}
</tool_call>
<tool_call>
{bad json here}
</tool_call><|im_end|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    # Valid tool call should be parsed
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "search"
    # Invalid tool call should be in unparsed_tool_calls
    assert "unparsed_tool_calls" in message
    assert len(message["unparsed_tool_calls"]) == 1
    assert "Invalid JSON" in message["unparsed_tool_calls"][0].error


def test_llama3_parse_invalid_tool_call_json():
    """Test that invalid JSON in Llama 3 tool call is captured as unparsed."""
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("llama3", tokenizer)

    response_text = """I'll get the weather.
<function=get_weather>{invalid json}</function><|eot_id|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    assert "tool_calls" not in message or len(message.get("tool_calls", [])) == 0
    assert "unparsed_tool_calls" in message
    assert len(message["unparsed_tool_calls"]) == 1
    assert "Invalid JSON" in message["unparsed_tool_calls"][0].error


def test_deepseek_parse_invalid_tool_call_json():
    """Test that invalid JSON in DeepSeek tool call is captured as unparsed."""
    model_name = "deepseek-ai/DeepSeek-V3.1"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("deepseekv3", tokenizer)

    response_text = """I'll check.
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{invalid json}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    assert "tool_calls" not in message or len(message.get("tool_calls", [])) == 0
    assert "unparsed_tool_calls" in message
    assert len(message["unparsed_tool_calls"]) == 1
    assert "Invalid JSON" in message["unparsed_tool_calls"][0].error


def test_kimi_k2_parse_invalid_tool_call_json():
    """Test that invalid JSON in Kimi K2 tool call is captured as unparsed."""
    model_name = "moonshotai/Kimi-K2-Thinking"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("kimi_k2", tokenizer)

    response_text = """<think></think>I'll search.
<|tool_calls_section_begin|><|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{invalid}<|tool_call_end|><|tool_calls_section_end|><|im_end|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    assert "tool_calls" not in message or len(message.get("tool_calls", [])) == 0
    assert "unparsed_tool_calls" in message
    assert len(message["unparsed_tool_calls"]) == 1
    assert "Invalid JSON" in message["unparsed_tool_calls"][0].error


# =============================================================================
# GptOss (OpenAI Harmony Format) Tool Calling Tests
# =============================================================================


def test_gptoss_tool_response_rendering():
    """Test that GptOss renders tool responses in Harmony format.

    Tool responses should use functions.{name} to=assistant routing in the
    commentary channel.
    """
    model_name = "openai/gpt-oss-20b"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer)

    tool_message: Message = {
        "role": "tool",
        "content": '{"weather": "sunny", "temp": 22}',
        "tool_call_id": "functions.get_weather:0",
    }

    ctx = RenderContext(idx=0, is_last=False, prev_message=None)
    rendered = renderer.render_message(tool_message, ctx)
    header = rendered.header
    assert header is not None, "Expected header in rendered message"
    output = rendered.output
    assert len(output) > 0, "Expected output in rendered message"

    header_str = tokenizer.decode(list(header.tokens))
    output_chunk = output[0]
    assert isinstance(output_chunk, tinker.EncodedTextChunk), "Expected EncodedTextChunk"
    output_str = tokenizer.decode(list(output_chunk.tokens))

    # Tool responses should use functions.{name} to=assistant format
    assert "functions.get_weather" in header_str
    assert "to=assistant" in header_str
    assert "commentary" in header_str
    # Output should contain the response
    assert '"weather": "sunny"' in output_str


def test_gptoss_tool_call_rendering():
    """Test that GptOss renders assistant messages with tool_calls correctly.

    When an assistant message has tool_calls, they should be rendered in
    the commentary channel with to=functions.{name} routing.
    """
    from tinker_cookbook.renderers import ToolCall

    model_name = "openai/gpt-oss-20b"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer)

    tool_call = ToolCall(
        function=ToolCall.FunctionBody(
            name="get_weather",
            arguments='{"city": "Tokyo"}',
        ),
    )
    assistant_message: Message = {
        "role": "assistant",
        "content": "",  # Tool-only message
        "tool_calls": [tool_call],
    }

    ctx = RenderContext(idx=0, is_last=False, prev_message=None)
    rendered = renderer.render_message(assistant_message, ctx)
    output_chunk = rendered.output[0]
    assert isinstance(output_chunk, tinker.EncodedTextChunk), "Expected EncodedTextChunk"
    output_str = tokenizer.decode(list(output_chunk.tokens))

    # Tool calls should use commentary channel with to=functions.{name}
    assert "<|channel|>commentary to=functions.get_weather" in output_str
    assert "<|constrain|>json" in output_str
    assert '{"city": "Tokyo"}' in output_str
    assert "<|call|>" in output_str


def test_gptoss_tool_call_with_content_rendering():
    """Test that GptOss renders tool calls followed by content correctly."""
    from tinker_cookbook.renderers import ToolCall

    model_name = "openai/gpt-oss-20b"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer)

    tool_call = ToolCall(
        function=ToolCall.FunctionBody(
            name="get_weather",
            arguments='{"city": "Tokyo"}',
        ),
    )
    assistant_message: Message = {
        "role": "assistant",
        "content": "Let me check the weather.",
        "tool_calls": [tool_call],
    }

    ctx = RenderContext(idx=0, is_last=False, prev_message=None)
    rendered = renderer.render_message(assistant_message, ctx)
    output_chunk = rendered.output[0]
    assert isinstance(output_chunk, tinker.EncodedTextChunk), "Expected EncodedTextChunk"
    output_str = tokenizer.decode(list(output_chunk.tokens))

    # Should have tool call followed by final channel
    assert "<|channel|>commentary to=functions.get_weather" in output_str
    assert "<|call|>" in output_str
    assert "<|channel|>final<|message|>Let me check the weather." in output_str


def test_gptoss_parse_single_tool_call():
    """Test parsing a single tool call from GptOss response."""
    model_name = "openai/gpt-oss-20b"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer)

    # Simulate model response with tool call in Harmony format
    response_text = """<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"city": "Tokyo"}<|call|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    assert message["role"] == "assistant"
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "get_weather"
    assert "Tokyo" in message["tool_calls"][0].function.arguments


def test_gptoss_parse_invalid_tool_call_json():
    """Test that invalid JSON in GptOss tool call is captured as unparsed."""
    model_name = "openai/gpt-oss-20b"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer)

    response_text = """<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{invalid json}<|call|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    assert "tool_calls" not in message or len(message.get("tool_calls", [])) == 0
    assert "unparsed_tool_calls" in message
    assert len(message["unparsed_tool_calls"]) == 1
    assert "Invalid JSON" in message["unparsed_tool_calls"][0].error


def test_gptoss_parse_thinking_from_analysis_channel():
    """Test that GptOss extracts CoT/reasoning from analysis channel into ThinkingPart.

    Per the Harmony format, the analysis channel contains chain-of-thought reasoning
    that should be exposed via ThinkingPart in the content list.
    See: https://cookbook.openai.com/articles/gpt-oss/handle-raw-cot
    """
    model_name = "openai/gpt-oss-20b"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer)

    # Simulate model response with analysis (CoT) and final channels
    response_text = """<|channel|>analysis<|message|>Let me think about this step by step.
The sky appears blue due to Rayleigh scattering.
Shorter wavelengths (blue) scatter more than longer ones.<|end|><|start|>assistant<|channel|>final<|message|>The sky is blue because of Rayleigh scattering of sunlight.<|return|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    assert message["role"] == "assistant"
    # Content should be a list with ThinkingPart and TextPart
    content = message["content"]
    assert isinstance(content, list), f"Expected list content, got {type(content)}"
    # Find ThinkingPart and TextPart
    thinking_parts = [p for p in content if p["type"] == "thinking"]
    text_parts = [p for p in content if p["type"] == "text"]
    assert len(thinking_parts) == 1, "Expected one ThinkingPart"
    thinking = thinking_parts[0]
    assert thinking["type"] == "thinking"  # Type narrowing
    assert "Rayleigh scattering" in thinking["thinking"]
    assert "step by step" in thinking["thinking"]
    # Text should be from final channel only
    assert len(text_parts) == 1, "Expected one TextPart"
    text = text_parts[0]
    assert text["type"] == "text"  # Type narrowing
    assert text["text"] == "The sky is blue because of Rayleigh scattering of sunlight."


def test_gptoss_parse_thinking_with_tool_calls():
    """Test that GptOss correctly parses thinking alongside tool calls."""
    model_name = "openai/gpt-oss-20b"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer)

    # Response with analysis channel, tool call, and final channel
    response_text = """<|channel|>analysis<|message|>I need to check the weather for the user.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"city": "Tokyo"}<|call|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    # Content should contain ThinkingPart
    content = message["content"]
    assert isinstance(content, list), f"Expected list content, got {type(content)}"
    thinking_parts = [p for p in content if p["type"] == "thinking"]
    assert len(thinking_parts) == 1
    thinking = thinking_parts[0]
    assert thinking["type"] == "thinking"  # Type narrowing
    assert "check the weather" in thinking["thinking"]
    # Tool call from commentary channel
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "get_weather"


def test_gptoss_create_system_prefix_with_tools():
    """Test that GptOss creates developer message with TypeScript-style tool definitions."""
    from tinker_cookbook.renderers import GptOssRenderer, ToolSpec

    model_name = "openai/gpt-oss-20b"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer)
    assert isinstance(renderer, GptOssRenderer)

    tools: list[ToolSpec] = [
        {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    ]

    messages = renderer.create_system_prefix_with_tools(tools, system_prompt="You are helpful.")
    assert len(messages) == 2

    # First message is developer with tool definitions
    assert messages[0]["role"] == "system"
    # Second message is system prompt
    assert messages[1]["role"] == "system"
    assert messages[1]["content"] == "You are helpful."

    # Developer message should have TypeScript-style definitions
    dev_content = messages[0]["content"]
    assert "namespace functions" in dev_content
    assert "type get_weather" in dev_content
    assert "city: string" in dev_content
    assert 'unit?: "celsius" | "fahrenheit"' in dev_content
    assert "// Get weather for a city" in dev_content

    # Verify create_conversation_prefix_with_tools delegates correctly
    messages2 = renderer.create_conversation_prefix_with_tools(
        tools, system_prompt="You are helpful."
    )
    assert messages == messages2


def test_gptoss_parse_multiple_tool_calls():
    """Test parsing multiple tool calls from a single GptOss response.

    When the model makes multiple tool calls in one response, all should be
    parsed correctly with sequential IDs (functions.{name}:{idx}).
    """
    model_name = "openai/gpt-oss-20b"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer)

    # Response with two tool calls in commentary channels
    response_text = """<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"city": "Tokyo"}<|call|><|start|>assistant<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"city": "London"}<|call|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 2

    # First tool call
    assert message["tool_calls"][0].function.name == "get_weather"
    assert "Tokyo" in message["tool_calls"][0].function.arguments
    assert message["tool_calls"][0].id == "functions.get_weather:0"

    # Second tool call
    assert message["tool_calls"][1].function.name == "get_weather"
    assert "London" in message["tool_calls"][1].function.arguments
    assert message["tool_calls"][1].id == "functions.get_weather:1"


def test_gptoss_tool_response_extracts_function_name():
    """Test that tool response rendering correctly extracts function name from tool_call_id.

    The tool_call_id format is 'functions.{name}:{idx}', and the renderer should
    extract the function name to use in the routing header.
    """
    model_name = "openai/gpt-oss-20b"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer)

    # Test with various tool_call_id formats
    test_cases = [
        ("functions.get_weather:0", "get_weather"),
        ("functions.search:5", "search"),
        ("functions.complex_function_name:123", "complex_function_name"),
    ]

    for tool_call_id, expected_name in test_cases:
        tool_message: Message = {
            "role": "tool",
            "content": '{"result": "success"}',
            "tool_call_id": tool_call_id,
        }

        ctx = RenderContext(idx=0, is_last=False, prev_message=None)
        rendered = renderer.render_message(tool_message, ctx)
        header_str = tokenizer.decode(list(rendered.header.tokens))

        assert f"functions.{expected_name}" in header_str, (
            f"Expected 'functions.{expected_name}' in header for tool_call_id '{tool_call_id}'"
        )
        assert "to=assistant" in header_str


def test_gptoss_parse_thinking_preserved_with_empty_final():
    """Test that thinking is preserved even when final channel is empty.

    Some responses may have analysis (thinking) but no final response content,
    e.g., when the model decides to make tool calls instead of responding.
    """
    model_name = "openai/gpt-oss-20b"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer)

    # Response with thinking in analysis channel but only tool calls (no final)
    response_text = """<|channel|>analysis<|message|>I need to search for this information before responding.<|end|><|start|>assistant<|channel|>commentary to=functions.search<|constrain|>json<|message|>{"query": "latest news"}<|call|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    # Content should be a list with ThinkingPart
    content = message["content"]
    assert isinstance(content, list), f"Expected list content, got {type(content)}"
    thinking_parts = [p for p in content if p["type"] == "thinking"]
    assert len(thinking_parts) == 1
    assert "search for this information" in thinking_parts[0]["thinking"]
    # Tool call should also be parsed
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1


def test_gptoss_parse_response_without_harmony_markers():
    """Test parsing a simple response without Harmony format markers.

    For backward compatibility, responses without channel markers should still
    be parsed as plain text content.
    """
    model_name = "openai/gpt-oss-20b"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer)

    # Simple response without channel markers
    response_text = "Hello, how can I help you today?<|return|>"

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    assert message["role"] == "assistant"
    # Content should be plain string without the stop token
    assert message["content"] == "Hello, how can I help you today?"


def test_gptoss_parse_mixed_valid_and_invalid_tool_calls():
    """Test that valid tool calls are parsed even when some have invalid JSON.

    If one tool call has invalid JSON, it should be captured as unparsed while
    other valid tool calls are still parsed correctly.
    """
    model_name = "openai/gpt-oss-20b"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer)

    # Response with one valid and one invalid tool call
    response_text = """<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"city": "Tokyo"}<|call|><|start|>assistant<|channel|>commentary to=functions.search<|constrain|>json<|message|>{invalid json}<|call|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success is True
    # Valid tool call should be parsed
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "get_weather"
    # Invalid tool call should be in unparsed
    assert "unparsed_tool_calls" in message
    assert len(message["unparsed_tool_calls"]) == 1
    assert "Invalid JSON" in message["unparsed_tool_calls"][0].error


def test_gptoss_tool_call_id_generation():
    """Test that rendered tool calls get proper IDs in functions.{name}:{idx} format.

    When rendering assistant messages with tool_calls, the IDs should follow
    the Harmony format convention.
    """
    from tinker_cookbook.renderers import ToolCall

    model_name = "openai/gpt-oss-20b"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer)

    # Create tool calls without IDs (renderer should not need to generate them for rendering)
    tool_calls = [
        ToolCall(function=ToolCall.FunctionBody(name="func_a", arguments='{"x": 1}')),
        ToolCall(function=ToolCall.FunctionBody(name="func_b", arguments='{"y": 2}')),
    ]
    assistant_message: Message = {
        "role": "assistant",
        "content": "",
        "tool_calls": tool_calls,
    }

    ctx = RenderContext(idx=0, is_last=False, prev_message=None)
    rendered = renderer.render_message(assistant_message, ctx)
    output_str = tokenizer.decode(list(rendered.output[0].tokens))

    # Verify both tool calls are in the output with proper routing
    assert "<|channel|>commentary to=functions.func_a" in output_str
    assert "<|channel|>commentary to=functions.func_b" in output_str
    assert '{"x": 1}' in output_str
    assert '{"y": 2}' in output_str
