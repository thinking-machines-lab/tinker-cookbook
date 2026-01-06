from tinker_cookbook.renderers import parse_content_blocks, ThinkingPart, TextPart
from tinker_cookbook.tests.test_renderers import get_tokenizer
from tinker_cookbook.renderers import Qwen3Renderer, DeepSeekV3ThinkingRenderer, GptOssRenderer


# =============================================================================
# parse_content_blocks Tests
# =============================================================================


def test_parse_content_blocks_no_special_tags():
    """Test parse_content_blocks returns None when no special tags."""
    parts = parse_content_blocks("Just plain text")
    assert parts is None


def test_parse_content_blocks_single_think_block():
    """Test parse_content_blocks with a single think block."""
    parts = parse_content_blocks("<think>reasoning</think>visible answer")
    assert parts is not None

    assert len(parts) == 2
    assert parts[0]["type"] == "thinking"
    assert parts[0]["thinking"] == "reasoning"  # type: ignore[typeddict-item]
    assert parts[1]["type"] == "text"
    assert parts[1]["text"] == "visible answer"  # type: ignore[typeddict-item]


def test_parse_content_blocks_multiple_think_blocks():
    """Test parse_content_blocks with multiple interleaved think blocks."""
    content = "<think>step 1</think>partial<think>step 2</think>final"
    parts = parse_content_blocks(content)
    assert parts is not None

    assert len(parts) == 4
    assert parts[0] == ThinkingPart(type="thinking", thinking="step 1")
    assert parts[1] == TextPart(type="text", text="partial")
    assert parts[2] == ThinkingPart(type="thinking", thinking="step 2")
    assert parts[3] == TextPart(type="text", text="final")


def test_parse_content_blocks_empty_blocks_omitted():
    """Test parse_content_blocks omits empty think blocks."""
    parts = parse_content_blocks("<think></think>visible")
    assert parts is not None

    assert len(parts) == 1
    assert parts[0]["type"] == "text"
    assert parts[0]["text"] == "visible"  # type: ignore[typeddict-item]


def test_parse_content_blocks_whitespace_handling():
    """Test parse_content_blocks preserves whitespace for identity roundtrip."""
    parts = parse_content_blocks("<think>  thinking  </think>  answer  ")
    assert parts is not None

    assert len(parts) == 2
    # Whitespace is preserved exactly for identity roundtrip
    assert parts[0]["type"] == "thinking" and parts[0]["thinking"] == "  thinking  "  # type: ignore[typeddict-item]
    assert parts[1]["type"] == "text" and parts[1]["text"] == "  answer  "  # type: ignore[typeddict-item]


def test_parse_content_blocks_tool_call_only():
    """Test parse_content_blocks parses tool calls."""
    content = '<tool_call>{"name": "search", "arguments": {"query": "test"}}</tool_call>'
    parts = parse_content_blocks(content)
    assert parts is not None

    assert len(parts) == 1
    assert parts[0]["type"] == "tool_call"
    tool_call = parts[0]["tool_call"]  # type: ignore[typeddict-item]
    assert tool_call.function.name == "search"
    assert tool_call.function.arguments == '{"query": "test"}'


def test_parse_content_blocks_interleaved():
    """Test parse_content_blocks handles interleaved think and tool_call blocks."""
    content = '<think>Let me search</think>Searching...<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>Done'
    parts = parse_content_blocks(content)
    assert parts is not None

    assert len(parts) == 4
    assert parts[0] == ThinkingPart(type="thinking", thinking="Let me search")
    assert parts[1] == TextPart(type="text", text="Searching...")
    assert parts[2]["type"] == "tool_call"
    assert parts[2]["tool_call"].function.name == "search"  # type: ignore[typeddict-item]
    assert parts[3] == TextPart(type="text", text="Done")


def test_parse_content_blocks_invalid_tool_call():
    """Test parse_content_blocks handles invalid tool call JSON as UnparsedToolCallPart."""
    content = "<tool_call>not valid json</tool_call>text after"
    parts = parse_content_blocks(content)
    assert parts is not None

    # Invalid tool call is included as UnparsedToolCallPart, text is still captured
    assert len(parts) == 2
    assert parts[0]["type"] == "unparsed_tool_call"
    assert "Invalid JSON" in parts[0]["error"]  # type: ignore[typeddict-item]
    assert parts[1] == TextPart(type="text", text="text after")


# =============================================================================
# Qwen3 parse_response Tests
# =============================================================================


def test_qwen3_parse_response_extracts_thinking():
    """Test Qwen3Renderer.parse_response extracts thinking to ThinkingPart."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    # Simulate a response with thinking
    response_str = "<think>Let me reason about this.</think>The answer is 42.<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    assert message["role"] == "assistant"

    # Content should be a list with ThinkingPart and TextPart
    content = message["content"]
    assert isinstance(content, list)

    thinking_parts = [p for p in content if p["type"] == "thinking"]
    text_parts = [p for p in content if p["type"] == "text"]

    assert len(thinking_parts) == 1
    assert thinking_parts[0]["thinking"] == "Let me reason about this."

    assert len(text_parts) == 1
    assert text_parts[0]["text"] == "The answer is 42."


def test_qwen3_parse_response_multiple_think_blocks():
    """Test Qwen3Renderer.parse_response handles multiple interleaved think blocks."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = "<think>step 1</think>partial answer<think>step 2</think>final answer<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    content = message["content"]
    assert isinstance(content, list)
    assert len(content) == 4

    assert content[0] == ThinkingPart(type="thinking", thinking="step 1")
    assert content[1] == TextPart(type="text", text="partial answer")
    assert content[2] == ThinkingPart(type="thinking", thinking="step 2")
    assert content[3] == TextPart(type="text", text="final answer")


def test_qwen3_parse_response_no_thinking_returns_string():
    """Test Qwen3Renderer.parse_response returns string when no thinking."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = "Just a plain response without thinking.<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    # Content should remain a string for backward compatibility
    assert isinstance(message["content"], str)
    assert message["content"] == "Just a plain response without thinking."


def test_qwen3_parse_response_with_tool_calls():
    """Test Qwen3Renderer.parse_response parses tool calls into ToolCallPart."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = '<think>Let me search</think>I will search for that.<tool_call>{"name": "web_search", "arguments": {"query": "weather"}}</tool_call><|im_end|>'
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    content = message["content"]
    assert isinstance(content, list)

    # Should have ThinkingPart, TextPart, ToolCallPart in order
    assert len(content) == 3
    assert content[0]["type"] == "thinking"
    assert content[0]["thinking"] == "Let me search"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "I will search for that."
    assert content[2]["type"] == "tool_call"
    assert content[2]["tool_call"].function.name == "web_search"

    # Also check backward-compatible tool_calls field
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "web_search"


def test_qwen3_parse_response_tool_call_only():
    """Test Qwen3Renderer.parse_response with only a tool call."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = (
        '<tool_call>{"name": "calculator", "arguments": {"expr": "2+2"}}</tool_call><|im_end|>'
    )
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    content = message["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]["type"] == "tool_call"

    # Backward-compatible field
    assert "tool_calls" in message and len(message["tool_calls"]) == 1


# =============================================================================
# DeepSeek parse_response Tests
# =============================================================================


def test_deepseek_parse_response_extracts_thinking():
    """Test DeepSeekV3ThinkingRenderer.parse_response extracts thinking."""
    tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3.1")
    renderer = DeepSeekV3ThinkingRenderer(tokenizer)

    # Note: DeepSeek uses full-width pipes in special tokens
    response_str = "<think>Let me think about this.</think>The answer is 42.<｜end▁of▁sentence｜>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    content = message["content"]
    assert isinstance(content, list)

    thinking_parts = [p for p in content if p["type"] == "thinking"]
    text_parts = [p for p in content if p["type"] == "text"]

    assert len(thinking_parts) == 1
    assert thinking_parts[0]["thinking"] == "Let me think about this."
    assert len(text_parts) == 1
    assert text_parts[0]["text"] == "The answer is 42."


def test_deepseek_parse_response_no_thinking_returns_string():
    """Test DeepSeekV3ThinkingRenderer.parse_response returns string when no thinking."""
    tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3.1")
    renderer = DeepSeekV3ThinkingRenderer(tokenizer)

    response_str = "Just a plain response.<｜end▁of▁sentence｜>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    assert isinstance(message["content"], str)
    assert message["content"] == "Just a plain response."


def test_deepseek_parse_response_multiple_think_blocks():
    """Test DeepSeekV3ThinkingRenderer.parse_response handles multiple think blocks."""
    tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3.1")
    renderer = DeepSeekV3ThinkingRenderer(tokenizer)

    response_str = "<think>step 1</think>partial<think>step 2</think>final<｜end▁of▁sentence｜>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    content = message["content"]
    assert isinstance(content, list)
    assert len(content) == 4

    assert content[0] == ThinkingPart(type="thinking", thinking="step 1")
    assert content[1] == TextPart(type="text", text="partial")
    assert content[2] == ThinkingPart(type="thinking", thinking="step 2")
    assert content[3] == TextPart(type="text", text="final")


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

    assert success
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

    # Multiple analysis channels (interleaved thinking)
    response_str = "<|channel|>analysis<|message|>First thought.<|end|><|start|>assistant<|channel|>analysis<|message|>Second thought.<|end|><|start|>assistant<|channel|>final<|message|>Done.<|return|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
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

    assert success
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

    assert success
    # No channel markers, so content stays as string
    assert isinstance(message["content"], str)
    assert message["content"] == "Plain response without channels."


# =============================================================================
# GptOss Tool Call Parsing Tests
# =============================================================================


def test_gptoss_parse_response_tool_call():
    """Test GptOssRenderer.parse_response extracts tool calls from commentary channel."""
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")

    # Tool call format: commentary channel with to=functions.name and <|call|> stop token
    response_str = '<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location": "San Francisco"}<|call|>'
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "get_weather"
    assert '"location"' in message["tool_calls"][0].function.arguments


def test_gptoss_parse_response_tool_call_with_analysis():
    """Test GptOssRenderer.parse_response extracts both thinking and tool calls."""
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")

    # Analysis (thinking) followed by tool call
    response_str = '<|channel|>analysis<|message|>I need to check the weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city": "NYC"}<|call|>'
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
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

    # Invalid JSON in tool call
    response_str = "<|channel|>commentary to=functions.broken <|constrain|>json<|message|>not valid json<|call|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    # Should have unparsed_tool_calls
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

    assert success
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

    assert success
    content = message["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0] == TextPart(type="text", text="Checking now.")
    assert "tool_calls" in message and len(message["tool_calls"]) == 1
