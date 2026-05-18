"""Tests specific to KimiK2Renderer (streaming parsing, thinking stripping, supervised examples)."""

from typing import TypeGuard

import pytest

from tinker_cookbook.renderers import (
    Message,
    StreamingMessageHeader,
    StreamingTextDelta,
    StreamingThinkingDelta,
    TextPart,
    ThinkingPart,
    ToolCall,
    TrainOnWhat,
    get_renderer,
)
from tinker_cookbook.renderers.base import ensure_list
from tinker_cookbook.renderers.kimi_k2 import KimiK2Renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer


def _is_message(obj) -> TypeGuard[Message]:
    """Check if object is a Message dict (TypedDict doesn't support isinstance)."""
    return isinstance(obj, dict) and "role" in obj and "content" in obj


# =============================================================================
# Conversation helpers
# =============================================================================


def _get_basic_4turn() -> list[Message]:
    return [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "user", "content": "And what is 3+3?"},
        {"role": "assistant", "content": "The answer is 6."},
    ]


def _get_tool_call_conversation() -> list[Message]:
    return [
        {"role": "user", "content": "What's the weather in San Francisco?"},
        {
            "role": "assistant",
            "content": "I'll check the weather for you.",
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="get_weather",
                        arguments='{"location": "San Francisco"}',
                    ),
                    id="call_123",
                )
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 72, "condition": "sunny"}',
            "tool_call_id": "call_123",
            "name": "get_weather",
        },
        {"role": "assistant", "content": "The weather in San Francisco is sunny with 72°F."},
    ]


# =============================================================================
# KimiK2 Streaming Parsing Tests
# =============================================================================


def test_kimi_streaming_simple_text():
    """Test streaming parsing of simple text response."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2.6")
    renderer = KimiK2Renderer(tokenizer)

    response_str = "Hello, world!<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    assert isinstance(deltas[0], StreamingMessageHeader)
    assert deltas[0].role == "assistant"

    assert _is_message(deltas[-1])
    assert deltas[-1]["role"] == "assistant"

    text_content = "".join(d.text for d in deltas if isinstance(d, StreamingTextDelta))
    assert "Hello, world!" in text_content


def test_kimi_streaming_with_thinking():
    """Test streaming parsing with thinking blocks."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2.6")
    renderer = KimiK2Renderer(tokenizer)

    response_str = "<think>Let me reason about this.</think>The answer is 42.<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    assert isinstance(deltas[0], StreamingMessageHeader)
    assert deltas[0].role == "assistant"

    thinking_content = "".join(d.thinking for d in deltas if isinstance(d, StreamingThinkingDelta))
    text_content = "".join(d.text for d in deltas if isinstance(d, StreamingTextDelta))

    assert "Let me reason about this." in thinking_content
    assert "The answer is 42." in text_content

    final_message = deltas[-1]
    assert _is_message(final_message)


def test_kimi_streaming_matches_batch():
    """Test that streaming parse produces same final message as batch parse."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2.6")
    renderer = KimiK2Renderer(tokenizer)

    response_str = "<think>Step 1: Analyze.\nStep 2: Compute.</think>The result is 123.<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    batch_message, batch_success = renderer.parse_response(response_tokens)
    assert batch_success.is_clean

    deltas = list(renderer.parse_response_streaming(response_tokens))
    streaming_message = deltas[-1]

    assert _is_message(streaming_message)
    assert streaming_message["role"] == batch_message["role"]
    assert ensure_list(streaming_message["content"]) == ensure_list(batch_message["content"])


def test_kimi_streaming_content_index_increments():
    """Test that content_index increments when switching content types."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2.6")
    renderer = KimiK2Renderer(tokenizer)

    response_str = "<think>thinking</think>text<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    thinking_indices = [d.content_index for d in deltas if isinstance(d, StreamingThinkingDelta)]
    text_indices = [d.content_index for d in deltas if isinstance(d, StreamingTextDelta)]

    if thinking_indices and text_indices:
        assert max(text_indices) > min(thinking_indices)


def test_kimi_streaming_multiple_think_blocks():
    """Test streaming with multiple interleaved think blocks."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2.6")
    renderer = KimiK2Renderer(tokenizer)

    response_str = "<think>first thought</think>partial<think>second thought</think>final<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    batch_message, _ = renderer.parse_response(response_tokens)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    thinking_content = "".join(d.thinking for d in deltas if isinstance(d, StreamingThinkingDelta))
    text_content = "".join(d.text for d in deltas if isinstance(d, StreamingTextDelta))

    assert "first thought" in thinking_content
    assert "second thought" in thinking_content
    assert "partial" in text_content
    assert "final" in text_content

    streaming_message = deltas[-1]
    assert _is_message(streaming_message)
    assert ensure_list(streaming_message["content"]) == ensure_list(batch_message["content"])


def test_kimi_streaming_empty_response():
    """Test streaming parsing of empty/minimal response."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2.6")
    renderer = KimiK2Renderer(tokenizer)

    response_str = "<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    assert isinstance(deltas[0], StreamingMessageHeader)
    assert _is_message(deltas[-1])


def test_kimi_streaming_no_unnecessary_buffering():
    """Test that we don't buffer more than necessary when no tag prefix matches."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2.6")
    renderer = KimiK2Renderer(tokenizer)

    response_str = "Hello world<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    text_content = "".join(d.text for d in deltas if isinstance(d, StreamingTextDelta))
    assert text_content == "Hello world"


def test_kimi_streaming_with_emoji():
    """Test that streaming parser handles emoji correctly."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2.6")
    renderer = KimiK2Renderer(tokenizer)

    response_str = "<think>Let me think 🤔</think>Here's a party 🎉!<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    thinking_content = "".join(d.thinking for d in deltas if isinstance(d, StreamingThinkingDelta))
    text_content = "".join(d.text for d in deltas if isinstance(d, StreamingTextDelta))

    assert "�" not in thinking_content, f"Thinking has replacement chars: {thinking_content!r}"
    assert "�" not in text_content, f"Text has replacement chars: {text_content!r}"

    assert "🤔" in thinking_content, f"Missing thinking emoji in: {thinking_content!r}"
    assert "🎉" in text_content, f"Missing party emoji in: {text_content!r}"

    final_messages = [d for d in deltas if isinstance(d, dict) and "role" in d]
    assert len(final_messages) == 1
    final = final_messages[0]

    content = final["content"]
    if isinstance(content, list):
        final_thinking = "".join(p["thinking"] for p in content if p["type"] == "thinking")
        final_text = "".join(p["text"] for p in content if p["type"] == "text")
    else:
        final_thinking = ""
        final_text = content

    assert "🤔" in final_thinking, "Final message missing thinking emoji"
    assert "🎉" in final_text, "Final message missing party emoji"


# =============================================================================
# Streaming vs Batch Equivalence Tests
# =============================================================================


def _assert_streaming_matches_batch(renderer, response_str: str):
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
    assert streaming_message.get("tool_calls") == batch_message.get("tool_calls")
    assert streaming_message.get("unparsed_tool_calls") == batch_message.get("unparsed_tool_calls")

    # Verify streamed deltas reconstruct the content
    thinking_from_deltas = "".join(
        d.thinking for d in deltas if isinstance(d, StreamingThinkingDelta)
    )
    text_from_deltas = "".join(d.text for d in deltas if isinstance(d, StreamingTextDelta))

    batch_content = batch_message["content"]
    if isinstance(batch_content, list):
        expected_thinking = "".join(p["thinking"] for p in batch_content if p["type"] == "thinking")
        expected_text = "".join(p["text"] for p in batch_content if p["type"] == "text")
    else:
        expected_thinking = ""
        expected_text = batch_content

    assert thinking_from_deltas == expected_thinking
    # Text deltas may include tool call markup before final parsing strips it
    if not batch_message.get("tool_calls") and not batch_message.get("unparsed_tool_calls"):
        assert text_from_deltas == expected_text

    return deltas, batch_message


class TestKimiK2StreamingBatchEquivalence:
    """Verify parse_response_streaming matches parse_response for all patterns."""

    @pytest.fixture
    def renderer(self):
        tokenizer = get_tokenizer("moonshotai/Kimi-K2.6")
        return KimiK2Renderer(tokenizer)

    def test_simple_text(self, renderer):
        _assert_streaming_matches_batch(renderer, "Hello, world!<|im_end|>")

    def test_thinking_then_text(self, renderer):
        _assert_streaming_matches_batch(
            renderer,
            "<think>Let me reason step by step.\n1. First...\n2. Then...</think>"
            "The answer is 42.<|im_end|>",
        )

    def test_empty_thinking(self, renderer):
        _assert_streaming_matches_batch(renderer, "<think></think>Direct answer.<|im_end|>")

    def test_long_thinking(self, renderer):
        thinking = (
            "First, let me understand the problem.\n\n"
            "Key concepts:\n1. Superposition\n2. Measurement\n3. Non-locality\n\n"
            "I should explain this clearly."
        )
        _assert_streaming_matches_batch(
            renderer, f"<think>{thinking}</think>Quantum entanglement links particles.<|im_end|>"
        )

    def test_multiple_think_blocks(self, renderer):
        _assert_streaming_matches_batch(
            renderer,
            "<think>first thought</think>partial<think>second thought</think>final<|im_end|>",
        )

    def test_empty_response(self, renderer):
        _assert_streaming_matches_batch(renderer, "<|im_end|>")

    def test_whitespace_only(self, renderer):
        _assert_streaming_matches_batch(renderer, "   \n\t  <|im_end|>")

    def test_special_characters(self, renderer):
        _assert_streaming_matches_batch(
            renderer,
            "<think>x² + y² = r²</think>Special chars: <>&\"'`~!@#$%^&*()<|im_end|>",
        )

    def test_emoji(self, renderer):
        _assert_streaming_matches_batch(
            renderer, "<think>🤔 thinking 💭</think>Answer 🎉✨!<|im_end|>"
        )

    def test_code_blocks(self, renderer):
        _assert_streaming_matches_batch(
            renderer,
            "<think>Need a function.</think>"
            "```python\ndef hello():\n    print('world')\n```<|im_end|>",
        )

    def test_html_like_content(self, renderer):
        _assert_streaming_matches_batch(
            renderer,
            "<think>HTML example</think><div><p>Hello</p></div><|im_end|>",
        )

    def test_tool_call_with_thinking(self, renderer):
        _assert_streaming_matches_batch(
            renderer,
            "<think>I need to search.</think>"
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>"
            '{"query": "quantum physics"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|><|im_end|>",
        )

    def test_tool_call_without_thinking(self, renderer):
        _assert_streaming_matches_batch(
            renderer,
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>"
            '{"city": "San Francisco"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|><|im_end|>",
        )

    def test_text_then_tool_call(self, renderer):
        _assert_streaming_matches_batch(
            renderer,
            "Let me look that up."
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>"
            '{"query": "weather"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|><|im_end|>",
        )

    def test_multiple_tool_calls(self, renderer):
        _assert_streaming_matches_batch(
            renderer,
            "<think>Two calls needed.</think>"
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>"
            '{"query": "python"}'
            "<|tool_call_end|>"
            "<|tool_call_begin|>functions.calculate:1<|tool_call_argument_begin|>"
            '{"expression": "2+2"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|><|im_end|>",
        )

    def test_multiline_thinking(self, renderer):
        _assert_streaming_matches_batch(
            renderer,
            "<think>\nStep 1\n\nStep 2\n\nStep 3\n</think>\nResult.\n<|im_end|>",
        )

    def test_no_end_token(self, renderer):
        """Truncated response — streaming should still parse think blocks."""
        tokenizer = renderer.tokenizer
        response_tokens = tokenizer.encode(
            "<think>reasoning</think>partial", add_special_tokens=False
        )

        deltas = list(renderer.parse_response_streaming(response_tokens))
        final = deltas[-1]
        assert _is_message(final)
        # Even without end token, streaming should parse think blocks
        content = final["content"]
        assert isinstance(content, list), "Truncated response should still parse think blocks"
        thinking = [p for p in content if p["type"] == "thinking"]
        text = [p for p in content if p["type"] == "text"]
        assert len(thinking) == 1 and thinking[0]["thinking"] == "reasoning"
        assert len(text) == 1 and text[0]["text"] == "partial"

    def test_content_index_ordering(self, renderer):
        """Content index strictly increases across type transitions."""
        response_tokens = renderer.tokenizer.encode(
            "<think>t1</think>x1<think>t2</think>x2<|im_end|>", add_special_tokens=False
        )
        deltas = list(renderer.parse_response_streaming(response_tokens))

        indexed = []
        for d in deltas:
            if isinstance(d, StreamingThinkingDelta):
                indexed.append(("thinking", d.content_index))
            elif isinstance(d, StreamingTextDelta):
                indexed.append(("text", d.content_index))

        indices = [idx for _, idx in indexed]
        assert indices == sorted(indices), f"Not monotonic: {indexed}"
        for i in range(1, len(indexed)):
            if indexed[i][0] != indexed[i - 1][0]:
                assert indexed[i][1] > indexed[i - 1][1]


# =============================================================================
# KimiK2 Thinking Stripping / Preservation Tests
# =============================================================================


def test_kimi_k2_thinking_stripped_when_no_suffix_messages():
    """
    Kimi K2 should preserve thinking only after the last non-tool-call assistant.
    This test checks that the history thinking is stripped with the presence of a non-tool-call assistant.
    """
    model_name = "moonshotai/Kimi-K2.6"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("kimi_k2", tokenizer)

    messages: list[Message] = [
        {"role": "user", "content": "Q"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="think A"),
            ],
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="get_weather",
                        arguments='{"location":"NYC"}',
                    ),
                    id="call_1",
                )
            ],
        },
        {"role": "tool", "content": '{"temperature": 72}', "tool_call_id": "call_1"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="think B"),
                TextPart(type="text", text="A"),
            ],
        },
    ]

    model_input, _ = renderer.build_supervised_example(messages)
    decoded = tokenizer.decode(model_input.to_ints())

    assert "think A" in decoded, f"Non-suffix thinking should be preserved: {decoded}"
    assert "think B" in decoded, f"Non-suffix thinking should be preserved: {decoded}"
    assert "A" in decoded, f"Non-suffix text should be preserved: {decoded}"

    model_input = renderer.build_generation_prompt(messages)
    decoded = tokenizer.decode(model_input.to_ints())

    assert "think A" not in decoded, f"History thinking should be stripped: {decoded}"
    assert "think B" not in decoded, f"History thinking should be stripped: {decoded}"
    assert "A" in decoded, f"History text should be preserved: {decoded}"


def test_kimi_k2_thinking_preserved_in_suffix_after_last_non_tool_call():
    """
    Kimi K2 should preserve thinking only after the last non-tool-call assistant.
    Suffix thinking is preserved but history thinking is stripped relative to the
    position of the last non-tool-call assistant.
    """
    model_name = "moonshotai/Kimi-K2.6"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("kimi_k2", tokenizer)

    messages: list[Message] = [
        {"role": "user", "content": "Q1"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="think A"),
                TextPart(type="text", text="A1"),
            ],
            "tool_calls": [],
        },
        {"role": "user", "content": "Q2"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="think B"),
            ],
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="get_weather",
                        arguments='{"location":"NYC"}',
                    ),
                    id="call_1",
                )
            ],
        },
        {"role": "tool", "content": '{"temperature": 72}', "tool_call_id": "call_1"},
    ]

    model_input, _ = renderer.build_supervised_example(messages)
    decoded = tokenizer.decode(model_input.to_ints())

    assert "think A" not in decoded, f"History thinking should be stripped: {decoded}"
    assert "A1" in decoded, f"History text should be preserved: {decoded}"
    assert "think B" in decoded, f"Suffix thinking should be preserved: {decoded}"

    model_input = renderer.build_generation_prompt(messages)
    decoded = tokenizer.decode(model_input.to_ints())

    assert "think A" not in decoded, f"History thinking should be stripped: {decoded}"
    assert "A1" in decoded, f"History text should be preserved: {decoded}"
    assert "think B" in decoded, f"Suffix thinking should be preserved: {decoded}"


def test_kimi_k2_thinking_preserved_when_no_non_tool_call_assistant():
    """
    When no non-tool-call assistant exists, all thinking should be preserved.
    """
    model_name = "moonshotai/Kimi-K2.6"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("kimi_k2", tokenizer)

    messages: list[Message] = [
        {"role": "user", "content": "Q"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="think A"),
            ],
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="get_weather",
                        arguments='{"location":"NYC"}',
                    ),
                    id="call_1",
                )
            ],
        },
        {"role": "tool", "content": '{"temperature": 72}', "tool_call_id": "call_1"},
    ]

    model_input, _ = renderer.build_supervised_example(messages)
    decoded = tokenizer.decode(model_input.to_ints())

    assert "think A" in decoded, f"Suffix thinking should be preserved: {decoded}"


# =============================================================================
# KimiK2 build_supervised_examples Tests
# =============================================================================


def test_kimi_k2_build_supervised_examples_last_assistant_matches():
    model_name = "moonshotai/Kimi-K2.6"
    tokenizer = get_tokenizer(model_name)
    renderer: KimiK2Renderer = get_renderer("kimi_k2", tokenizer)  # type: ignore

    messages = _get_basic_4turn()

    single_input, single_weights = renderer.build_supervised_example(messages)
    examples = renderer.build_supervised_examples(
        messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
    )

    assert len(examples) == 1, "Expected a single supervised example"
    list_input, list_weights = examples[0]
    assert list_input.to_ints() == single_input.to_ints()
    assert list_weights.tolist() == single_weights.tolist()


def test_kimi_k2_build_supervised_examples_all_assistant_matches():
    model_name = "moonshotai/Kimi-K2.6"
    tokenizer = get_tokenizer(model_name)
    renderer: KimiK2Renderer = get_renderer("kimi_k2", tokenizer)  # type: ignore

    messages: list[Message] = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "Q3"},
        {"role": "assistant", "content": "A3"},
    ]

    examples = renderer.build_supervised_examples(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )

    assert len(examples) == 3, (
        "Expected one example per user turn after the first and one for the full conversation"
    )

    ex0_tokens = examples[0][0].to_ints()
    ex1_tokens = examples[1][0].to_ints()
    ex2_tokens = examples[2][0].to_ints()
    ex0_decoded = tokenizer.decode(ex0_tokens)
    ex1_decoded = tokenizer.decode(ex1_tokens)
    ex2_decoded = tokenizer.decode(ex2_tokens)

    assert "A1" in ex0_decoded
    assert "A2" not in ex0_decoded
    assert "A3" not in ex0_decoded

    assert "A1" in ex1_decoded
    assert "A2" in ex1_decoded
    assert "A3" not in ex1_decoded

    assert "A1" in ex2_decoded
    assert "A2" in ex2_decoded
    assert "A3" in ex2_decoded


def test_kimi_k2_build_supervised_examples_warns_on_non_assistant_mode():
    model_name = "moonshotai/Kimi-K2.6"
    tokenizer = get_tokenizer(model_name)
    renderer: KimiK2Renderer = get_renderer("kimi_k2", tokenizer)  # type: ignore

    messages = _get_basic_4turn()

    with pytest.warns(UserWarning, match="does not satisfy the extension property"):
        examples = renderer.build_supervised_examples(
            messages, train_on_what=TrainOnWhat.ALL_MESSAGES
        )

    assert len(examples) == 2, (
        "Expected one example for the full conversation and one for the last user turn"
    )
    ex0_tokens = examples[0][0].to_ints()
    ex1_tokens = examples[1][0].to_ints()
    ex0_decoded = tokenizer.decode(ex0_tokens)
    ex1_decoded = tokenizer.decode(ex1_tokens)

    assert "2+2" in ex0_decoded
    assert "4" in ex0_decoded
    assert "3+3" not in ex0_decoded
    assert "6" not in ex0_decoded

    assert "2+2" in ex1_decoded
    assert "4" in ex1_decoded
    assert "3+3" in ex1_decoded
    assert "6" in ex1_decoded


def test_kimi_k2_build_supervised_examples_all_assistant_matches_with_tool_calls():
    model_name = "moonshotai/Kimi-K2.6"
    tokenizer = get_tokenizer(model_name)
    renderer: KimiK2Renderer = get_renderer("kimi_k2", tokenizer)  # type: ignore

    messages: list[Message] = [
        {"role": "user", "content": "Q"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="think A"),
            ],
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="get_weather",
                        arguments='{"location":"NYC"}',
                    ),
                    id="call_1",
                )
            ],
        },
        {"role": "tool", "content": '{"temperature": 72}', "tool_call_id": "call_1"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="think B"),
                TextPart(type="text", text="A"),
            ],
        },
        {"role": "user", "content": "Q2"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="think C"),
            ],
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="get_weather",
                        arguments='{"location":"NYC"}',
                    ),
                    id="call_1",
                )
            ],
        },
        {"role": "tool", "content": '{"temperature": 72}', "tool_call_id": "call_1"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="think D"),
                TextPart(type="text", text="A2"),
            ],
        },
    ]

    examples = renderer.build_supervised_examples(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )

    assert len(examples) == 2
    example0_input, example0_weights = examples[0]
    example1_input, example1_weights = examples[1]

    expected_input, expected_weights = renderer.build_supervised_example(
        messages[:4], train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN
    )
    all_assist_input, all_assist_weights = renderer.build_supervised_example(
        messages[:4], train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )

    assert example0_input.to_ints() == expected_input.to_ints()
    assert example0_weights.tolist() == expected_weights.tolist()
    # since we only have one turn in `messages[:4]`, the weights should be the same
    assert example0_weights.tolist() == all_assist_weights.tolist()

    expected_input, expected_weights = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN
    )
    all_assist_input, all_assist_weights = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )

    assert example1_input.to_ints() == expected_input.to_ints()
    assert example1_weights.tolist() == expected_weights.tolist()
    assert example1_weights.tolist() != all_assist_weights.tolist()
