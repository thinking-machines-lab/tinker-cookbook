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
from tinker_cookbook.renderers.base import (
    ToolCall,
    UnparsedToolCall,
    Utf8TokenDecoder,
    _longest_matching_suffix_prefix,
    ensure_list,
)
from tinker_cookbook.renderers.deepseek_v3 import DeepSeekV3DisableThinkingRenderer
from tinker_cookbook.renderers.kimi_k2 import KimiK2Renderer
from tinker_cookbook.renderers.qwen3_5 import Qwen3_5DisableThinkingRenderer, Qwen3_5Renderer
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
    result = parse_content_blocks("<think>reasoning</think>visible answer")
    assert result is not None
    parts, tool_calls = result

    assert len(parts) == 2
    assert parts[0]["type"] == "thinking"
    assert parts[0]["thinking"] == "reasoning"
    assert parts[1]["type"] == "text"
    assert parts[1]["text"] == "visible answer"
    assert len(tool_calls) == 0


def test_parse_content_blocks_multiple_think_blocks():
    """Test parse_content_blocks with multiple think blocks preserves order."""
    result = parse_content_blocks(
        "<think>first thought</think>middle<think>second thought</think>end"
    )
    assert result is not None
    parts, tool_calls = result
    assert len(parts) == 4
    assert parts[0] == ThinkingPart(type="thinking", thinking="first thought")
    assert parts[1] == TextPart(type="text", text="middle")
    assert parts[2] == ThinkingPart(type="thinking", thinking="second thought")
    assert parts[3] == TextPart(type="text", text="end")
    assert len(tool_calls) == 0


def test_parse_content_blocks_empty_blocks_omitted():
    """Test that empty think and tool_call blocks are omitted."""
    result = parse_content_blocks("<think></think>text<tool_call></tool_call>")
    assert result is not None
    parts, tool_calls = result
    # Empty <think></think> and <tool_call></tool_call> should be omitted
    assert len(parts) == 1
    assert parts[0] == TextPart(type="text", text="text")


def test_parse_content_blocks_whitespace_handling():
    """Test whitespace-only blocks are omitted or preserved correctly."""
    result = parse_content_blocks("<think>  \n  </think>content")
    assert result is not None
    parts, _ = result
    # Whitespace-only think blocks are kept (not truly empty)
    thinking_parts = [p for p in parts if p["type"] == "thinking"]
    assert len(thinking_parts) == 1
    assert thinking_parts[0]["thinking"] == "  \n  "


def test_parse_content_blocks_tool_call_only():
    """Test parsing response with only tool calls."""
    result = parse_content_blocks(
        '<tool_call>\n{"name": "search", "arguments": {"query": "test"}}\n</tool_call>'
    )
    assert result is not None
    parts, tool_calls = result
    assert len(tool_calls) == 1
    assert isinstance(tool_calls[0], ToolCall)
    assert tool_calls[0].function.name == "search"


def test_parse_content_blocks_interleaved():
    """Test think blocks interleaved with tool calls preserves order."""
    content = (
        "<think>reasoning</think>I'll search for that."
        '\n<tool_call>\n{"name": "search", "arguments": {"query": "test"}}\n</tool_call>'
    )
    result = parse_content_blocks(content)
    assert result is not None
    parts, tool_calls = result

    # Should have thinking, text, then tool call
    assert parts[0]["type"] == "thinking"
    assert parts[1]["type"] == "text"
    assert len(tool_calls) == 1


def test_parse_content_blocks_invalid_tool_call():
    """Test that malformed tool calls produce UnparsedToolCall."""
    result = parse_content_blocks("<tool_call>\nnot valid json\n</tool_call>")
    assert result is not None
    parts, tool_calls = result
    assert len(tool_calls) == 1
    assert isinstance(tool_calls[0], UnparsedToolCall)


def test_format_content_as_string_roundtrip():
    """Test format_content_as_string with various content types."""
    # String content
    assert format_content_as_string("hello") == "hello"

    # Structured content
    parts: list[ContentPart] = [
        ThinkingPart(type="thinking", thinking="reasoning"),
        TextPart(type="text", text="answer"),
    ]
    result = format_content_as_string(parts)
    assert "reasoning" in result
    assert "answer" in result


# =============================================================================
# Qwen3 Response Parsing Tests
# =============================================================================


def test_qwen3_parse_response_extracts_thinking():
    """Test that Qwen3 parse_response correctly extracts thinking blocks."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = "<think>Let me reason about this.</think>The answer is 42.<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)
    assert success
    assert message["role"] == "assistant"

    content = message["content"]
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0]["type"] == "thinking"
    assert content[0]["thinking"] == "Let me reason about this."
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "The answer is 42."


def test_qwen3_parse_response_multiple_think_blocks():
    """Test Qwen3 parse_response with multiple think/text segments."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = "<think>first</think>middle<think>second</think>end<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)
    assert success

    content = message["content"]
    assert isinstance(content, list)
    assert len(content) == 4
    assert content[0] == ThinkingPart(type="thinking", thinking="first")
    assert content[1] == TextPart(type="text", text="middle")
    assert content[2] == ThinkingPart(type="thinking", thinking="second")
    assert content[3] == TextPart(type="text", text="end")


def test_qwen3_parse_response_no_thinking_returns_string():
    """Test Qwen3 parse_response returns string when no thinking blocks."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = "Just plain text response<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)
    assert success
    assert isinstance(message["content"], str)
    assert message["content"] == "Just plain text response"


def test_qwen3_parse_response_with_tool_calls():
    """Test Qwen3 parse_response extracts tool calls correctly."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = (
        "<think>I need to search</think>Let me look that up."
        '\n<tool_call>\n{"name": "search", "arguments": {"query": "test"}}\n</tool_call>'
        "<|im_end|>"
    )
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)
    assert success
    assert "tool_calls" in message
    assert message["tool_calls"][0].function.name == "search"


def test_qwen3_parse_response_tool_call_only():
    """Test Qwen3 parse_response with tool call but no text."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "SF"}}\n</tool_call>'
        "<|im_end|>"
    )
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)
    assert success
    assert "tool_calls" in message
    assert message["tool_calls"][0].function.name == "get_weather"


# =============================================================================
# DeepSeek V3 Response Parsing Tests
# =============================================================================


def test_deepseek_parse_response_extracts_thinking():
    """Test that DeepSeek parse_response correctly extracts thinking blocks."""
    tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3")
    renderer = DeepSeekV3ThinkingRenderer(tokenizer)

    content = "<think>reasoning here</think>The visible answer."
    end_token = tokenizer.encode("<｜end▁of▁sentence｜>", add_special_tokens=False)
    response_tokens = tokenizer.encode(content, add_special_tokens=False) + end_token

    message, success = renderer.parse_response(response_tokens)
    assert success

    content_parts = message["content"]
    assert isinstance(content_parts, list)
    thinking = [p for p in content_parts if p["type"] == "thinking"]
    text = [p for p in content_parts if p["type"] == "text"]
    assert len(thinking) == 1
    assert thinking[0]["thinking"] == "reasoning here"
    assert len(text) == 1
    assert text[0]["text"] == "The visible answer."


def test_deepseek_parse_response_no_thinking_returns_string():
    """Test DeepSeek parse_response returns string when no thinking blocks."""
    tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3")
    renderer = DeepSeekV3ThinkingRenderer(tokenizer)

    content = "No thinking here."
    end_token = tokenizer.encode("<｜end▁of▁sentence｜>", add_special_tokens=False)
    response_tokens = tokenizer.encode(content, add_special_tokens=False) + end_token

    message, success = renderer.parse_response(response_tokens)
    assert success
    assert isinstance(message["content"], str)
    assert message["content"] == "No thinking here."


def test_deepseek_parse_response_multiple_think_blocks():
    """Test DeepSeek parse_response with multiple think/text segments."""
    tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3")
    renderer = DeepSeekV3ThinkingRenderer(tokenizer)

    content = "<think>first</think>middle<think>second</think>end"
    end_token = tokenizer.encode("<｜end▁of▁sentence｜>", add_special_tokens=False)
    response_tokens = tokenizer.encode(content, add_special_tokens=False) + end_token

    message, success = renderer.parse_response(response_tokens)
    assert success

    parts = message["content"]
    assert isinstance(parts, list)
    assert len(parts) == 4
    assert parts[0] == ThinkingPart(type="thinking", thinking="first")
    assert parts[1] == TextPart(type="text", text="middle")
    assert parts[2] == ThinkingPart(type="thinking", thinking="second")
    assert parts[3] == TextPart(type="text", text="end")


# =============================================================================
# GptOss Response Parsing Tests
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


# =============================================================================
# Thinking-Generation-Parse Roundtrip Tests
# =============================================================================


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
        ("Qwen/Qwen3.5-35B-A3B", Qwen3_5Renderer, {}),
        ("Qwen/Qwen3.5-35B-A3B", Qwen3_5DisableThinkingRenderer, {}),
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

    # Simulate sampling: strip prefill
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
# Utf8TokenDecoder Tests
# =============================================================================


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
            if tokens == [1, 2, 3]:
                return "✓"
            elif tokens == [1, 2, 3, 4]:
                raise ValueError("Incomplete UTF-8: token 4 is partial")
            elif 4 in tokens:
                raise ValueError("Incomplete UTF-8: token 4 is partial")
            else:
                raise ValueError(f"Incomplete UTF-8: {tokens}")

    decoder = Utf8TokenDecoder(MockTokenizer())  # type: ignore[arg-type]
    result = decoder.decode([1, 2, 3, 4])

    assert result == "✓", f"Expected '✓' but got {result!r}"
    assert decoder._pending_tokens == [4], f"Expected [4] pending but got {decoder._pending_tokens}"


def test_utf8_decoder_with_real_tokenizer_ascii():
    """Test Utf8TokenDecoder with real tokenizer on ASCII text."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    test_str = "Hello World! How are you today?"
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


def test_utf8_decoder_handles_replacement_chars():
    """Test Utf8TokenDecoder buffers tokens that decode to replacement chars.

    Tiktoken-based tokenizers return U+FFFD for incomplete UTF-8 instead
    of raising exceptions. The decoder detects these and buffers.
    """
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    test_str = "🎉"
    tokens = tokenizer.encode(test_str, add_special_tokens=False)

    # Verify individual tokens produce replacement chars (tiktoken behavior)
    for tok in tokens:
        decoded = tokenizer.decode([tok])
        assert "�" in decoded, f"Expected replacement char for token {tok}, got {decoded!r}"

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


def test_utf8_decoder_mixed_ascii_and_emoji():
    """Test Utf8TokenDecoder with mixed ASCII and multi-byte Unicode."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
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


# =============================================================================
# _longest_matching_suffix_prefix Tests
# =============================================================================


def test_longest_matching_suffix_prefix():
    """Test the suffix-prefix matching helper used for tag boundary detection."""
    # No match
    assert _longest_matching_suffix_prefix("hello", "<think>") == 0
    assert _longest_matching_suffix_prefix("hello world", "<think>") == 0
    assert _longest_matching_suffix_prefix("", "<think>") == 0

    # Partial matches for <think>
    assert _longest_matching_suffix_prefix("hello<", "<think>") == 1
    assert _longest_matching_suffix_prefix("hello<t", "<think>") == 2
    assert _longest_matching_suffix_prefix("hello<th", "<think>") == 3
    assert _longest_matching_suffix_prefix("hello<thi", "<think>") == 4
    assert _longest_matching_suffix_prefix("hello<thin", "<think>") == 5
    assert _longest_matching_suffix_prefix("hello<think", "<think>") == 6

    # Non-matching partial
    assert _longest_matching_suffix_prefix("hello<thx", "<think>") == 0
    assert _longest_matching_suffix_prefix("hello<tx", "<think>") == 0

    # Partial matches for </think>
    assert _longest_matching_suffix_prefix("thinking</", "</think>") == 2
    assert _longest_matching_suffix_prefix("thinking</t", "</think>") == 3
    assert _longest_matching_suffix_prefix("thinking</think", "</think>") == 7

    # Edge: text shorter than tag
    assert _longest_matching_suffix_prefix("<t", "<think>") == 2
    assert _longest_matching_suffix_prefix("<", "<think>") == 1


# =============================================================================
# Kimi K2 Streaming Tests
#
# Tests for parse_response_streaming on KimiK2Renderer, organized as:
# 1. TestKimiK2StreamingBatchEquivalence - comprehensive streaming vs batch
#    comparison across all response patterns (text, thinking, tool calls, etc.)
# 2. TestKimiK2StreamingBehavior - streaming-specific behavior (delta structure,
#    content indexing, buffering, emoji/unicode handling)
# =============================================================================


def _is_message(obj) -> TypeGuard[Message]:
    """Check if object is a Message dict (TypedDict doesn't support isinstance)."""
    return isinstance(obj, dict) and "role" in obj and "content" in obj


def _assert_streaming_matches_batch(renderer, response_str: str):
    """Helper: verify streaming and batch parsing produce identical results.

    Checks that:
    - Streaming yields header → deltas → final Message
    - Final Message matches batch parse_response exactly (content, tool_calls)
    - Concatenated deltas reconstruct the expected content
    """
    tokenizer = renderer.tokenizer
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    # Batch parse
    batch_message, batch_success = renderer.parse_response(response_tokens)

    # Streaming parse
    deltas = list(renderer.parse_response_streaming(response_tokens))

    # Structure checks
    assert len(deltas) >= 2, "Should have at least header + final message"
    assert isinstance(deltas[0], StreamingMessageHeader), "First delta should be header"
    assert deltas[0].role == "assistant"
    assert _is_message(deltas[-1]), "Last delta should be complete Message"

    streaming_message = deltas[-1]

    # Content equivalence
    assert streaming_message["role"] == batch_message["role"]
    assert ensure_list(streaming_message["content"]) == ensure_list(batch_message["content"])

    # Tool calls equivalence
    assert streaming_message.get("tool_calls") == batch_message.get("tool_calls")
    assert streaming_message.get("unparsed_tool_calls") == batch_message.get("unparsed_tool_calls")

    # Verify streamed deltas reconstruct the content
    thinking_from_deltas = "".join(
        d.thinking for d in deltas if isinstance(d, StreamingThinkingDelta)
    )
    text_from_deltas = "".join(d.text for d in deltas if isinstance(d, StreamingTextDelta))

    # Extract expected content from batch message
    batch_content = batch_message["content"]
    if isinstance(batch_content, list):
        expected_thinking = "".join(
            p["thinking"] for p in batch_content if p["type"] == "thinking"
        )
        expected_text = "".join(p["text"] for p in batch_content if p["type"] == "text")
    else:
        expected_thinking = ""
        expected_text = batch_content

    assert thinking_from_deltas == expected_thinking, (
        f"Thinking mismatch:\n  deltas: {thinking_from_deltas!r}\n  batch:  {expected_thinking!r}"
    )
    # Text may include tool call markup in the streamed version (since deltas
    # emit raw text before final parsing strips it). Only compare when no tool calls.
    if not batch_message.get("tool_calls") and not batch_message.get("unparsed_tool_calls"):
        assert text_from_deltas == expected_text, (
            f"Text mismatch:\n  deltas: {text_from_deltas!r}\n  batch:  {expected_text!r}"
        )

    return deltas, batch_message


class TestKimiK2StreamingBatchEquivalence:
    """Verify parse_response_streaming matches parse_response for all patterns.

    Each test feeds the same tokens to both batch and streaming parsers and
    asserts the final Message is identical. This is the primary correctness
    guarantee for the streaming implementation.
    """

    @pytest.fixture
    def renderer(self):
        tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
        return KimiK2Renderer(tokenizer)

    # --- Text patterns ---

    def test_simple_text(self, renderer):
        """Plain text without thinking."""
        _assert_streaming_matches_batch(renderer, "Hello, world!<|im_end|>")

    def test_whitespace_only(self, renderer):
        """Whitespace-only response."""
        _assert_streaming_matches_batch(renderer, "   \n\t  <|im_end|>")

    def test_empty_response(self, renderer):
        """Only end token."""
        _assert_streaming_matches_batch(renderer, "<|im_end|>")

    def test_special_characters(self, renderer):
        """Special chars, newlines, unicode math."""
        _assert_streaming_matches_batch(
            renderer,
            "<think>Analysis of x² + y² = r²\nwhere r > 0</think>"
            "The equation x² + y² = r² defines a circle.\n"
            "Special chars: <>&\"'`~!@#$%^&*()<|im_end|>",
        )

    def test_emoji_in_thinking_and_text(self, renderer):
        """Emoji in both thinking and text."""
        _assert_streaming_matches_batch(
            renderer,
            "<think>🤔 Let me think about this carefully 💭</think>"
            "Here's the answer 🎉✨!<|im_end|>",
        )

    def test_code_blocks(self, renderer):
        """Response containing code blocks."""
        _assert_streaming_matches_batch(
            renderer,
            "<think>The user needs a Python function.</think>"
            "Here's the code:\n```python\ndef hello():\n    print('world')\n```<|im_end|>",
        )

    def test_html_like_content(self, renderer):
        """HTML-like tags that aren't think tags."""
        _assert_streaming_matches_batch(
            renderer,
            "<think>Generating HTML example</think>"
            "Use <div class=\"container\"><p>Hello</p></div><|im_end|>",
        )

    # --- Thinking patterns ---

    def test_thinking_then_text(self, renderer):
        """Standard thinking + answer."""
        _assert_streaming_matches_batch(
            renderer,
            "<think>Let me reason step by step.\n1. First...\n2. Then...</think>"
            "The answer is 42.<|im_end|>",
        )

    def test_empty_thinking(self, renderer):
        """Empty think block (non-thinking mode)."""
        _assert_streaming_matches_batch(
            renderer, "<think></think>Direct answer.<|im_end|>"
        )

    def test_long_thinking(self, renderer):
        """Extended reasoning with multiple paragraphs."""
        thinking = (
            "First, let me understand the problem. The user is asking about "
            "quantum entanglement.\n\n"
            "Key concepts:\n"
            "1. Superposition - particles exist in multiple states\n"
            "2. Measurement - observing collapses the state\n"
            "3. Non-locality - entangled particles correlate instantly\n\n"
            "I should explain this clearly without jargon."
        )
        _assert_streaming_matches_batch(
            renderer,
            f"<think>{thinking}</think>"
            "Quantum entanglement is a phenomenon where two particles become linked.<|im_end|>",
        )

    def test_multiple_think_blocks(self, renderer):
        """Multiple interleaved think/text blocks."""
        _assert_streaming_matches_batch(
            renderer,
            "<think>first thought</think>partial answer"
            "<think>second thought</think>final answer<|im_end|>",
        )

    def test_multiline_thinking_with_newlines(self, renderer):
        """Thinking with varied newline formatting."""
        _assert_streaming_matches_batch(
            renderer,
            "<think>\nStep 1: Parse the input\n\nStep 2: Process\n\n\nStep 3: Output\n</think>"
            "\nHere is the result.\n<|im_end|>",
        )

    # --- Tool call patterns ---

    def test_tool_call_with_thinking(self, renderer):
        """Thinking followed by a tool call."""
        _assert_streaming_matches_batch(
            renderer,
            "<think>I need to search for this.</think>"
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>"
            '{"query": "quantum physics"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|><|im_end|>",
        )

    def test_tool_call_without_thinking(self, renderer):
        """Direct tool call with no thinking."""
        _assert_streaming_matches_batch(
            renderer,
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>"
            '{"city": "San Francisco"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|><|im_end|>",
        )

    def test_text_then_tool_call(self, renderer):
        """Text content followed by a tool call."""
        _assert_streaming_matches_batch(
            renderer,
            "Let me look that up for you."
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>"
            '{"query": "weather today"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|><|im_end|>",
        )

    def test_multiple_tool_calls(self, renderer):
        """Multiple tool calls in one response."""
        _assert_streaming_matches_batch(
            renderer,
            "<think>I need to call two functions.</think>"
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>"
            '{"query": "python"}'
            "<|tool_call_end|>"
            "<|tool_call_begin|>functions.calculate:1<|tool_call_argument_begin|>"
            '{"expression": "2+2"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|><|im_end|>",
        )

    # --- Edge cases ---

    def test_no_end_token(self, renderer):
        """Truncated response (no end token). Both should return success=False."""
        tokenizer = renderer.tokenizer
        response_str = "<think>reasoning</think>partial answer"
        response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

        batch_message, batch_success = renderer.parse_response(response_tokens)
        assert not batch_success

        deltas = list(renderer.parse_response_streaming(response_tokens))
        streaming_message = deltas[-1]
        assert _is_message(streaming_message)
        assert streaming_message["role"] == batch_message["role"]


class TestKimiK2StreamingBehavior:
    """Test streaming-specific behavior: delta structure, buffering, unicode."""

    @pytest.fixture
    def renderer(self):
        tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
        return KimiK2Renderer(tokenizer)

    def test_content_index_ordering(self, renderer):
        """Content index strictly increases across type transitions."""
        tokenizer = renderer.tokenizer
        response_str = (
            "<think>thought 1</think>text 1<think>thought 2</think>text 2<|im_end|>"
        )
        response_tokens = tokenizer.encode(response_str, add_special_tokens=False)
        deltas = list(renderer.parse_response_streaming(response_tokens))

        indexed = []
        for d in deltas:
            if isinstance(d, StreamingThinkingDelta):
                indexed.append(("thinking", d.content_index))
            elif isinstance(d, StreamingTextDelta):
                indexed.append(("text", d.content_index))

        # Indices should be monotonically non-decreasing
        indices = [idx for _, idx in indexed]
        assert indices == sorted(indices), f"Content indices not monotonic: {indexed}"

        # Each type transition should increment
        for i in range(1, len(indexed)):
            if indexed[i][0] != indexed[i - 1][0]:
                assert indexed[i][1] > indexed[i - 1][1], (
                    f"Index didn't increment on type change: {indexed[i-1]} -> {indexed[i]}"
                )

    def test_no_unnecessary_buffering(self, renderer):
        """Text without any tag-like suffixes is emitted immediately."""
        tokenizer = renderer.tokenizer
        response_str = "Hello world<|im_end|>"
        response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

        deltas = list(renderer.parse_response_streaming(response_tokens))
        text_content = "".join(d.text for d in deltas if isinstance(d, StreamingTextDelta))
        assert text_content == "Hello world"

    def test_emoji_no_replacement_chars(self, renderer):
        """Emoji are preserved without replacement characters in deltas."""
        tokenizer = renderer.tokenizer
        response_str = "<think>Let me think 🤔</think>Here's a party 🎉!<|im_end|>"
        response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

        deltas = list(renderer.parse_response_streaming(response_tokens))

        thinking = "".join(d.thinking for d in deltas if isinstance(d, StreamingThinkingDelta))
        text = "".join(d.text for d in deltas if isinstance(d, StreamingTextDelta))

        assert "�" not in thinking, f"Replacement chars in thinking: {thinking!r}"
        assert "�" not in text, f"Replacement chars in text: {text!r}"
        assert "🤔" in thinking
        assert "🎉" in text

        # Final message also preserves emoji
        final = [d for d in deltas if _is_message(d)]
        assert len(final) == 1
        content = final[0]["content"]
        if isinstance(content, list):
            final_thinking = "".join(p["thinking"] for p in content if p["type"] == "thinking")
            final_text = "".join(p["text"] for p in content if p["type"] == "text")
        else:
            final_thinking, final_text = "", content
        assert "🤔" in final_thinking
        assert "🎉" in final_text
