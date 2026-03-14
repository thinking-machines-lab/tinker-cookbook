"""Tests specific to DeepSeek V3 renderers (parse_response, tool call behavior)."""

import tinker

from tinker_cookbook.renderers import (
    Message,
    RenderContext,
    TextPart,
    ThinkingPart,
    ToolCall,
)
from tinker_cookbook.renderers.deepseek_v3 import (
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
            output_str = tokenizer.decode(list(output_chunk.tokens))
            assert not output_str.startswith("</think>"), (
                f"Post-tool assistant should not have </think> prefix: {output_str}"
            )
