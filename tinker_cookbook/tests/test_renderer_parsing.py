from typing import TypeGuard

import pytest

from tinker_cookbook.renderers import (
    ContentPart,
    DeepSeekV3ThinkingRenderer,
    GptOssRenderer,
    Message,
    Qwen3Renderer,
    RenderContext,
    StreamingMessageHeader,
    StreamingTextDelta,
    StreamingThinkingDelta,
    TextPart,
    ThinkingPart,
    format_content_as_string,
    parse_content_blocks,
)
from tinker_cookbook.renderers.base import Utf8TokenDecoder, ensure_list
from tinker_cookbook.renderers.deepseek_v3 import DeepSeekV3DisableThinkingRenderer
from tinker_cookbook.renderers.kimi_k2 import KimiK2Renderer, _longest_matching_suffix_prefix
from tinker_cookbook.tests.test_renderers import get_tokenizer

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


def test_format_content_as_string_roundtrip():
    """Formatted content should be parseable back to original."""
    content = [
        ThinkingPart(type="thinking", thinking="reasoning"),
        TextPart(type="text", text="answer"),
    ]
    # Use empty separator for true roundtrip (default separator adds newlines between parts)
    formatted = format_content_as_string(content, separator="")
    parsed = parse_content_blocks(formatted)
    assert parsed == content


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
    response_str = "Let me think about this.</think>The answer is 42.<｜end▁of▁sentence｜>"
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

    response_str = "step 1</think>partial<think>step 2</think>final<｜end▁of▁sentence｜>"
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


@pytest.mark.parametrize(
    "model_name,renderer_cls,renderer_kwargs",
    [
        ("deepseek-ai/DeepSeek-V3.1", DeepSeekV3ThinkingRenderer, {}),
        ("deepseek-ai/DeepSeek-V3.1", DeepSeekV3DisableThinkingRenderer, {}),
        (
            "openai/gpt-oss-20b",
            GptOssRenderer,
            {"use_system_prompt": True, "reasoning_effort": "medium"},
        ),
        ("Qwen/Qwen3-30B-A3B", Qwen3Renderer, {}),
        ("moonshotai/Kimi-K2-Thinking", KimiK2Renderer, {}),
    ],
)
def test_thinking_generation_parse_correspondence(model_name, renderer_cls, renderer_kwargs):
    """Test that parse_response handles sampled output after thinking prefill.

    Pattern for thinking model tests:
    1. Build generation prompt (may include thinking prefill)
    2. Render expected message to get full response tokens
    3. Strip prefill to simulate what sampling returns
    4. Parse continuation → should recover the expected message
    5. Roundtrip: prompt + continuation = full supervised example
    """
    tokenizer = get_tokenizer(model_name)
    renderer = renderer_cls(tokenizer, **renderer_kwargs)

    # User message
    user_message: Message = {"role": "user", "content": "What is 2+2?"}

    # Expected parsed message (what we want parse_response to produce)
    thinking: list[ContentPart] = []
    if "DisableThinking" not in renderer_cls.__name__:
        thinking = [ThinkingPart(type="thinking", thinking="Let me work through this.")]
    expected_content = thinking + [TextPart(type="text", text="The answer is 42.")]
    expected_message: Message = {"role": "assistant", "content": expected_content}

    # Render expected message to get full response tokens
    rendered = renderer.render_message(
        expected_message, RenderContext(idx=1, is_last=True, prev_message=user_message)
    )
    full_response_tokens = [t for chunk in rendered.output for t in chunk.tokens]

    # Build prompt (may include prefill like <think>)
    prompt = renderer.build_generation_prompt([user_message])
    prompt_tokens = prompt.to_ints()

    # Find prefill by matching end of prompt with start of rendered response
    # This is renderer-agnostic: whatever prefill the renderer adds will be found
    prefill_len = 0
    for i in range(1, min(len(prompt_tokens), len(full_response_tokens)) + 1):
        if prompt_tokens[-i:] == full_response_tokens[:i]:
            prefill_len = i

    # Simulate smpling: strip prefill
    continuation_tokens = full_response_tokens[prefill_len:]

    # Parse the continuation
    parsed_message, _ = renderer.parse_response(continuation_tokens)

    # Should recover the expected message
    assert ensure_list(parsed_message["content"]) == ensure_list(expected_message["content"]), (
        f"Roundtrip failed: parsed_message != expected_message for {model_name} {renderer_cls.__name__}"
    )

    # Roundtrip: full conversation should match prompt + continuation
    full_convo = [user_message, parsed_message]
    supervised, _ = renderer.build_supervised_example(full_convo)
    assert supervised.to_ints() == prompt_tokens + continuation_tokens


# =============================================================================
# Kimi K2 Streaming Parsing Tests
# =============================================================================


def _is_message(obj) -> TypeGuard[Message]:
    """Check if object is a Message dict (TypedDict doesn't support isinstance)."""
    return isinstance(obj, dict) and "role" in obj and "content" in obj


def test_kimi_streaming_simple_text():
    """Test streaming parsing of simple text response."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    renderer = KimiK2Renderer(tokenizer)

    response_str = "Hello, world!<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    # First delta should be header
    assert isinstance(deltas[0], StreamingMessageHeader)
    assert deltas[0].role == "assistant"

    # Last delta should be complete Message
    assert _is_message(deltas[-1])
    assert deltas[-1]["role"] == "assistant"

    # Collect all text deltas
    text_content = "".join(d.text for d in deltas if isinstance(d, StreamingTextDelta))
    assert "Hello, world!" in text_content


def test_kimi_streaming_with_thinking():
    """Test streaming parsing with thinking blocks."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    renderer = KimiK2Renderer(tokenizer)

    response_str = "<think>Let me reason about this.</think>The answer is 42.<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    # First delta should be header
    assert isinstance(deltas[0], StreamingMessageHeader)
    assert deltas[0].role == "assistant"

    # Collect thinking and text deltas
    thinking_content = "".join(d.thinking for d in deltas if isinstance(d, StreamingThinkingDelta))
    text_content = "".join(d.text for d in deltas if isinstance(d, StreamingTextDelta))

    assert "Let me reason about this." in thinking_content
    assert "The answer is 42." in text_content

    # Last delta should be complete Message
    final_message = deltas[-1]
    assert _is_message(final_message)


def test_kimi_streaming_matches_batch():
    """Test that streaming parse produces same final message as batch parse."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    renderer = KimiK2Renderer(tokenizer)

    response_str = "<think>Step 1: Analyze.\nStep 2: Compute.</think>The result is 123.<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    # Batch parse
    batch_message, batch_success = renderer.parse_response(response_tokens)
    assert batch_success

    # Streaming parse
    deltas = list(renderer.parse_response_streaming(response_tokens))
    streaming_message = deltas[-1]

    assert _is_message(streaming_message)
    assert streaming_message["role"] == batch_message["role"]
    assert ensure_list(streaming_message["content"]) == ensure_list(batch_message["content"])


def test_kimi_streaming_content_index_increments():
    """Test that content_index increments when switching content types."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    renderer = KimiK2Renderer(tokenizer)

    response_str = "<think>thinking</think>text<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    # Get content indices from deltas
    thinking_indices = [d.content_index for d in deltas if isinstance(d, StreamingThinkingDelta)]
    text_indices = [d.content_index for d in deltas if isinstance(d, StreamingTextDelta)]

    # Thinking should have content_index > 0 (after header is emitted, we enter thinking)
    # Text should have higher content_index than thinking
    if thinking_indices and text_indices:
        # Text comes after thinking closes, so its index should be higher
        assert max(text_indices) > min(thinking_indices)


def test_kimi_streaming_multiple_think_blocks():
    """Test streaming with multiple interleaved think blocks."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    renderer = KimiK2Renderer(tokenizer)

    response_str = "<think>first thought</think>partial<think>second thought</think>final<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    # Batch parse for reference
    batch_message, _ = renderer.parse_response(response_tokens)

    # Streaming parse
    deltas = list(renderer.parse_response_streaming(response_tokens))

    # Collect all content
    thinking_content = "".join(d.thinking for d in deltas if isinstance(d, StreamingThinkingDelta))
    text_content = "".join(d.text for d in deltas if isinstance(d, StreamingTextDelta))

    assert "first thought" in thinking_content
    assert "second thought" in thinking_content
    assert "partial" in text_content
    assert "final" in text_content

    # Final message should match batch
    streaming_message = deltas[-1]
    assert _is_message(streaming_message)
    assert ensure_list(streaming_message["content"]) == ensure_list(batch_message["content"])


def test_kimi_streaming_empty_response():
    """Test streaming parsing of empty/minimal response."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    renderer = KimiK2Renderer(tokenizer)

    response_str = "<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    # Should still get header and final message
    assert isinstance(deltas[0], StreamingMessageHeader)
    assert _is_message(deltas[-1])


# =============================================================================
# Streaming Helper Function Tests
# =============================================================================


def test_longest_matching_suffix_prefix():
    """Test the suffix-prefix matching helper function."""
    # No match cases
    assert _longest_matching_suffix_prefix("hello", "<think>") == 0
    assert _longest_matching_suffix_prefix("hello world", "<think>") == 0
    assert _longest_matching_suffix_prefix("", "<think>") == 0

    # Partial matches
    assert _longest_matching_suffix_prefix("hello<", "<think>") == 1
    assert _longest_matching_suffix_prefix("hello<t", "<think>") == 2
    assert _longest_matching_suffix_prefix("hello<th", "<think>") == 3
    assert _longest_matching_suffix_prefix("hello<thi", "<think>") == 4
    assert _longest_matching_suffix_prefix("hello<thin", "<think>") == 5
    assert _longest_matching_suffix_prefix("hello<think", "<think>") == 6

    # Non-matching partial (doesn't match prefix)
    assert _longest_matching_suffix_prefix("hello<thx", "<think>") == 0
    assert _longest_matching_suffix_prefix("hello<tx", "<think>") == 0

    # For </think>
    assert _longest_matching_suffix_prefix("thinking</", "</think>") == 2
    assert _longest_matching_suffix_prefix("thinking</t", "</think>") == 3
    assert _longest_matching_suffix_prefix("thinking</think", "</think>") == 7

    # Edge: text shorter than tag
    assert _longest_matching_suffix_prefix("<t", "<think>") == 2
    assert _longest_matching_suffix_prefix("<", "<think>") == 1


def test_kimi_streaming_no_unnecessary_buffering():
    """Test that we don't buffer more than necessary when no tag prefix matches."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    renderer = KimiK2Renderer(tokenizer)

    # "Hello world" has no suffix matching any prefix of "<think>"
    # So all of it should be emitted without buffering
    response_str = "Hello world<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    # Collect all text deltas
    text_content = "".join(d.text for d in deltas if isinstance(d, StreamingTextDelta))

    # Should contain the full text
    assert text_content == "Hello world"


def test_utf8_decoder_non_monotonic_decodability():
    """Test that Utf8TokenDecoder handles non-monotonic decodability.

    This test would FAIL with binary search but PASSES with backwards scan.

    The scenario: tokens [A, B, C, D] where:
    - decode([A]) fails (partial UTF-8)
    - decode([A, B]) fails (still partial)
    - decode([A, B, C]) succeeds (completes the character!)
    - decode([A, B, C, D]) fails (D starts a new partial)

    Binary search would:
    - Try mid=2: decode([A,B]) fails → high=1
    - Try mid=1: decode([A]) fails → high=0
    - Return None (WRONG - missed that [:3] works!)

    Backwards scan:
    - Try removing 1 token: decode([A,B,C]) succeeds → return it ✓
    """

    class MockTokenizer:
        """Mock tokenizer that simulates non-monotonic UTF-8 decodability."""

        def decode(self, tokens: list[int]) -> str:
            # Simulate: tokens 1,2,3 together form valid UTF-8,
            # but subsets [1], [1,2] are invalid, and [1,2,3,4] is invalid
            # (token 4 starts a new incomplete sequence)
            if tokens == [1, 2, 3]:
                return "✓"  # Only this combination decodes
            elif tokens == [1, 2, 3, 4]:
                raise ValueError("Incomplete UTF-8: token 4 is partial")
            elif 4 in tokens:
                raise ValueError("Incomplete UTF-8: token 4 is partial")
            else:
                raise ValueError(f"Incomplete UTF-8: {tokens}")

    decoder = Utf8TokenDecoder(MockTokenizer())  # type: ignore[arg-type]

    # Feed all 4 tokens at once
    result = decoder.decode([1, 2, 3, 4])

    # Should decode [1,2,3] and buffer [4]
    assert result == "✓", f"Expected '✓' but got {result!r}"
    assert decoder._pending_tokens == [4], f"Expected [4] pending but got {decoder._pending_tokens}"


def test_utf8_decoder_with_real_tokenizer_ascii():
    """Test Utf8TokenDecoder with real tokenizer on ASCII text.

    Note: Many tokenizers (including tiktoken-based ones like Kimi) don't throw
    exceptions for incomplete UTF-8 - they return replacement characters (�).
    This means our exception-based buffering doesn't help for those tokenizers.

    However, for ASCII text (single-byte UTF-8), there's no splitting issue,
    so the decoder should work correctly.
    """
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")

    # ASCII-only text won't have UTF-8 splitting issues
    test_str = "Hello World! How are you today?"
    tokens = tokenizer.encode(test_str, add_special_tokens=False)

    # Feed tokens one at a time and collect decoded text
    decoder = Utf8TokenDecoder(tokenizer)
    decoded_parts = []
    for token in tokens:
        result = decoder.decode([token])
        if result is not None:
            decoded_parts.append(result)

    # Flush any remaining
    remaining = decoder.flush()
    if remaining:
        decoded_parts.append(remaining)

    # Concatenated result should match original
    full_decoded = "".join(decoded_parts)
    assert full_decoded == test_str, f"Expected {test_str!r} but got {full_decoded!r}"


def test_utf8_decoder_handles_replacement_chars():
    """Test that Utf8TokenDecoder handles tokenizers that return replacement chars.

    Tiktoken-based tokenizers (like Kimi's) return U+FFFD (replacement character)
    for incomplete UTF-8 instead of raising exceptions. The decoder detects these
    and buffers tokens until the sequence completes.
    """
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")

    # The emoji 🎉 is encoded as multiple tokens
    test_str = "🎉"
    tokens = tokenizer.encode(test_str, add_special_tokens=False)

    # Verify tokens individually decode to replacement chars (confirming tiktoken behavior)
    for tok in tokens:
        decoded = tokenizer.decode([tok])
        assert "�" in decoded, f"Expected replacement char for token {tok}, got {decoded!r}"

    # Now test that our decoder handles this correctly
    decoder = Utf8TokenDecoder(tokenizer)
    decoded_parts = []

    for token in tokens:
        result = decoder.decode([token])
        if result is not None:
            decoded_parts.append(result)

    # Flush any remaining
    remaining = decoder.flush()
    if remaining:
        decoded_parts.append(remaining)

    # The concatenated result should be the original emoji (no replacement chars)
    full_decoded = "".join(decoded_parts)
    assert full_decoded == test_str, f"Expected {test_str!r} but got {full_decoded!r}"
    assert "�" not in full_decoded, "Should not contain replacement characters"


def test_utf8_decoder_mixed_ascii_and_emoji():
    """Test streaming with mixed ASCII and multi-byte Unicode."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")

    # Mix of ASCII and emoji
    test_str = "Hello 🎉 World 🌍!"
    tokens = tokenizer.encode(test_str, add_special_tokens=False)

    decoder = Utf8TokenDecoder(tokenizer)
    decoded_parts = []

    for token in tokens:
        result = decoder.decode([token])
        if result is not None:
            decoded_parts.append(result)

    remaining = decoder.flush()
    if remaining:
        decoded_parts.append(remaining)

    full_decoded = "".join(decoded_parts)
    assert full_decoded == test_str, f"Expected {test_str!r} but got {full_decoded!r}"
    assert "�" not in full_decoded, "Should not contain replacement characters"


def test_kimi_streaming_with_emoji():
    """Test that streaming parser handles emoji correctly."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    renderer = KimiK2Renderer(tokenizer)

    # Response with emoji in both thinking and text
    response_str = "<think>Let me think 🤔</think>Here's a party 🎉!<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    # Collect thinking content
    thinking_content = "".join(d.thinking for d in deltas if isinstance(d, StreamingThinkingDelta))

    # Collect text content
    text_content = "".join(d.text for d in deltas if isinstance(d, StreamingTextDelta))

    # Verify no replacement characters in streamed content
    assert "�" not in thinking_content, f"Thinking has replacement chars: {thinking_content!r}"
    assert "�" not in text_content, f"Text has replacement chars: {text_content!r}"

    # Verify emoji are preserved
    assert "🤔" in thinking_content, f"Missing thinking emoji in: {thinking_content!r}"
    assert "🎉" in text_content, f"Missing party emoji in: {text_content!r}"

    # Verify final message also has correct emoji
    final_messages = [d for d in deltas if isinstance(d, dict) and "role" in d]
    assert len(final_messages) == 1
    final = final_messages[0]

    # Get text from final message
    content = final["content"]
    if isinstance(content, list):
        final_thinking = "".join(p["thinking"] for p in content if p["type"] == "thinking")
        final_text = "".join(p["text"] for p in content if p["type"] == "text")
    else:
        final_thinking = ""
        final_text = content

    assert "🤔" in final_thinking, "Final message missing thinking emoji"
    assert "🎉" in final_text, "Final message missing party emoji"
