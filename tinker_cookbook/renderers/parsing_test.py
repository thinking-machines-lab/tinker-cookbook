"""Tests for shared parsing utilities and cross-renderer roundtrip behavior."""

import pytest

from tinker_cookbook.renderers import (
    ContentPart,
    DeepSeekV3ThinkingRenderer,
    GptOssRenderer,
    Message,
    Qwen3Renderer,
    RenderContext,
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
from tinker_cookbook.tokenizer_utils import get_tokenizer

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
    assert parts[0]["thinking"] == "reasoning"  # type: ignore[typeddict-item]
    assert parts[1]["type"] == "text"
    assert parts[1]["text"] == "visible answer"  # type: ignore[typeddict-item]
    assert tool_calls == []


def test_parse_content_blocks_multiple_think_blocks():
    """Test parse_content_blocks with multiple interleaved think blocks."""
    content = "<think>step 1</think>partial<think>step 2</think>final"
    result = parse_content_blocks(content)
    assert result is not None
    parts, tool_calls = result

    assert len(parts) == 4
    assert parts[0] == ThinkingPart(type="thinking", thinking="step 1")
    assert parts[1] == TextPart(type="text", text="partial")
    assert parts[2] == ThinkingPart(type="thinking", thinking="step 2")
    assert parts[3] == TextPart(type="text", text="final")
    assert tool_calls == []


def test_parse_content_blocks_empty_blocks_omitted():
    """Test parse_content_blocks omits empty think blocks."""
    result = parse_content_blocks("<think></think>visible")
    assert result is not None
    parts, tool_calls = result

    assert len(parts) == 1
    assert parts[0]["type"] == "text"
    assert parts[0]["text"] == "visible"  # type: ignore[typeddict-item]
    assert tool_calls == []


def test_parse_content_blocks_whitespace_handling():
    """Test parse_content_blocks preserves whitespace for identity roundtrip."""
    result = parse_content_blocks("<think>  thinking  </think>  answer  ")
    assert result is not None
    parts, tool_calls = result

    assert len(parts) == 2
    # Whitespace is preserved exactly for identity roundtrip
    assert parts[0]["type"] == "thinking" and parts[0]["thinking"] == "  thinking  "  # type: ignore[typeddict-item]
    assert parts[1]["type"] == "text" and parts[1]["text"] == "  answer  "  # type: ignore[typeddict-item]
    assert tool_calls == []


def test_parse_content_blocks_tool_call_only():
    """Test parse_content_blocks parses tool calls into separate list."""
    content = '<tool_call>{"name": "search", "arguments": {"query": "test"}}</tool_call>'
    result = parse_content_blocks(content)
    assert result is not None
    parts, tool_calls = result

    assert len(parts) == 0
    assert len(tool_calls) == 1
    assert isinstance(tool_calls[0], ToolCall)
    assert tool_calls[0].function.name == "search"
    assert tool_calls[0].function.arguments == '{"query": "test"}'


def test_parse_content_blocks_interleaved():
    """Test parse_content_blocks handles interleaved think and tool_call blocks."""
    content = '<think>Let me search</think>Searching...<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>Done'
    result = parse_content_blocks(content)
    assert result is not None
    parts, tool_calls = result

    # Content parts: think, text before tool_call, text after tool_call
    assert len(parts) == 3
    assert parts[0] == ThinkingPart(type="thinking", thinking="Let me search")
    assert parts[1] == TextPart(type="text", text="Searching...")
    assert parts[2] == TextPart(type="text", text="Done")

    # Tool call extracted separately
    assert len(tool_calls) == 1
    assert isinstance(tool_calls[0], ToolCall)
    assert tool_calls[0].function.name == "search"


def test_parse_content_blocks_invalid_tool_call():
    """Test parse_content_blocks handles invalid tool call JSON as UnparsedToolCall."""
    content = "<tool_call>not valid json</tool_call>text after"
    result = parse_content_blocks(content)
    assert result is not None
    parts, tool_calls = result

    # Text after tool call is captured in content parts
    assert len(parts) == 1
    assert parts[0] == TextPart(type="text", text="text after")

    # Invalid tool call is in tool_calls list as UnparsedToolCall
    assert len(tool_calls) == 1
    assert isinstance(tool_calls[0], UnparsedToolCall)
    assert "Invalid JSON" in tool_calls[0].error


def test_format_content_as_string_roundtrip():
    """Formatted content should be parseable back to original."""
    content = [
        ThinkingPart(type="thinking", thinking="reasoning"),
        TextPart(type="text", text="answer"),
    ]
    # Use empty separator for true roundtrip (default separator adds newlines between parts)
    formatted = format_content_as_string(content, separator="")
    result = parse_content_blocks(formatted)
    assert result is not None
    parts, tool_calls = result
    assert parts == content
    assert tool_calls == []


# =============================================================================
# _longest_matching_suffix_prefix Tests
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
            # Simulate: tokens 1,2,3 together form valid UTF-8,
            # but subsets [1], [1,2] are invalid, and [1,2,3,4] is invalid
            # (token 4 starts a new incomplete sequence)
            if tokens == [1, 2, 3]:
                return "✓"  # Only this combination decodes
            elif tokens == [1, 2, 3, 4] or 4 in tokens:
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
    exceptions for incomplete UTF-8 - they return replacement characters (â).
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

    # Verify tokens individually decode to replacement/garbled chars (confirming tiktoken behavior)
    for tok in tokens:
        decoded = tokenizer.decode([tok])
        assert decoded != test_str, (
            f"Expected garbled output for partial token {tok}, got {decoded!r}"
        )

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
    assert "â" not in full_decoded, "Should not contain replacement characters"


# =============================================================================
# Cross-Renderer Parse Correspondence Tests
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
