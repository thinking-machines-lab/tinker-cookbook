"""Tests specific to DeepSeek V3 renderers (parse_response, tool call behavior, streaming)."""

import pytest
import tinker

from tinker_cookbook.renderers import (
    Message,
    RenderContext,
    StreamingMessageHeader,
    TextPart,
    ThinkingPart,
    ToolCall,
)
from tinker_cookbook.renderers.base import ensure_list
from tinker_cookbook.renderers.deepseek_v3 import (
    DeepSeekV3DisableThinkingRenderer,
    DeepSeekV3ThinkingRenderer,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

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
# DeepSeek Tool Call / Formatting Tests
# =============================================================================


def test_deepseek_thinking_preserved_with_tool_calls():
    """
    Test that thinking is preserved in messages that have tool_calls.
    The thinking represents the model's reasoning about WHY it's making the tool call.
    """
    model_name = "deepseek-ai/DeepSeek-V3.1"
    tokenizer = get_tokenizer(model_name)
    renderer = DeepSeekV3ThinkingRenderer(tokenizer)  # Default strip_thinking_from_history=True

    messages: list[Message] = [
        {"role": "user", "content": "What's the weather in NYC?"},
        {
            "role": "assistant",
            "content": "<think>I need to check the weather.</think>Let me look that up.",
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="get_weather",
                        arguments='{"location": "NYC"}',
                    ),
                    id="call_1",
                )
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 72}',
            "tool_call_id": "call_1",
        },
        {"role": "assistant", "content": "The temperature in NYC is 72°F."},
    ]

    model_input, _ = renderer.build_supervised_example(messages)
    decoded = tokenizer.decode(model_input.to_ints())

    # Thinking in message with tool_calls should be preserved
    assert "I need to check the weather" in decoded, (
        f"Thinking in tool_call message should be preserved: {decoded}"
    )


def test_deepseek_post_tool_formatting():
    """
    Test that assistant messages following tool responses have correct formatting.
    Post-tool assistant messages should not have the role token or </think> prefix.
    """
    model_name = "deepseek-ai/DeepSeek-V3.1"
    tokenizer = get_tokenizer(model_name)
    renderer = DeepSeekV3ThinkingRenderer(tokenizer)

    messages: list[Message] = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="get_weather",
                        arguments='{"location": "NYC"}',
                    ),
                    id="call_1",
                )
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 72}',
            "tool_call_id": "call_1",
        },
        {"role": "assistant", "content": "The temperature is 72°F."},
    ]

    for idx, message in enumerate(messages):
        ctx = RenderContext(
            idx=idx,
            is_last=idx == len(messages) - 1,
            prev_message=messages[idx - 1] if idx > 0 else None,
        )
        follows_tool = ctx.prev_message is not None and ctx.prev_message["role"] == "tool"
        rendered = renderer.render_message(message, ctx)

        if message["role"] == "assistant" and follows_tool:
            # Post-tool assistant should have no header (no role token)
            header = rendered.header
            assert header is None or len(header.tokens) == 0, (
                f"Post-tool assistant should have no header, got: {header}"
            )

            # Output should not start with </think>
            output_chunk = rendered.output[0]
            assert isinstance(output_chunk, tinker.EncodedTextChunk), "Expected EncodedTextChunk"
            output_str = str(tokenizer.decode(list(output_chunk.tokens)))
            assert not output_str.startswith("</think>"), (
                f"Post-tool assistant should not have </think> prefix: {output_str}"
            )


# =============================================================================
# DeepSeek Streaming Tests
# =============================================================================


def _is_message(obj) -> bool:
    return isinstance(obj, dict) and "role" in obj and "content" in obj


def _assert_deepseek_streaming_matches_batch(renderer, response_str: str):
    """Helper: verify streaming and batch parsing produce identical results."""
    tokenizer = renderer.tokenizer
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    batch_message, batch_success = renderer.parse_response(response_tokens)
    deltas = list(renderer.parse_response_streaming(response_tokens))

    assert len(deltas) >= 2, "Should have at least header + final message"
    assert isinstance(deltas[0], StreamingMessageHeader)
    assert _is_message(deltas[-1])

    streaming_message = deltas[-1]
    assert streaming_message["role"] == batch_message["role"]
    assert ensure_list(streaming_message["content"]) == ensure_list(batch_message["content"])

    return deltas, batch_message


class TestDeepSeekStreamingBatchEquivalence:
    """Verify parse_response_streaming matches parse_response for DeepSeek patterns."""

    @pytest.fixture
    def thinking_renderer(self):
        tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3.1")
        return DeepSeekV3ThinkingRenderer(tokenizer)

    @pytest.fixture
    def non_thinking_renderer(self):
        tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3.1")
        return DeepSeekV3DisableThinkingRenderer(tokenizer)

    def test_simple_text(self, thinking_renderer):
        _assert_deepseek_streaming_matches_batch(
            thinking_renderer, "Hello, world!<｜end▁of▁sentence｜>"
        )

    def test_thinking_then_text(self, thinking_renderer):
        _assert_deepseek_streaming_matches_batch(
            thinking_renderer,
            "Let me think about this.</think>The answer is 42.<｜end▁of▁sentence｜>",
        )

    def test_multiple_think_blocks(self, thinking_renderer):
        _assert_deepseek_streaming_matches_batch(
            thinking_renderer,
            "step 1</think>partial<think>step 2</think>final<｜end▁of▁sentence｜>",
        )

    def test_empty_response(self, thinking_renderer):
        _assert_deepseek_streaming_matches_batch(thinking_renderer, "<｜end▁of▁sentence｜>")

    def test_non_thinking_renderer(self, non_thinking_renderer):
        _assert_deepseek_streaming_matches_batch(
            non_thinking_renderer, "Direct answer.<｜end▁of▁sentence｜>"
        )

    def test_no_end_token(self, thinking_renderer):
        """Truncated response — streaming should still parse think blocks."""
        tokenizer = thinking_renderer.tokenizer
        response_tokens = tokenizer.encode("reasoning</think>partial", add_special_tokens=False)

        deltas = list(thinking_renderer.parse_response_streaming(response_tokens))
        final = deltas[-1]
        assert _is_message(final)
        content = final["content"]
        assert isinstance(content, list), "Truncated response should still parse think blocks"
        thinking = [p for p in content if p["type"] == "thinking"]
        text = [p for p in content if p["type"] == "text"]
        assert len(thinking) == 1 and thinking[0]["thinking"] == "reasoning"
        assert len(text) == 1 and text[0]["text"] == "partial"
