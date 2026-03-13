"""Tests specific to Qwen3 renderers (parse_response, streaming, disable-thinking behavior)."""

from typing import TypeGuard, cast

import pytest
from transformers.models.auto.tokenization_auto import AutoTokenizer

from tinker_cookbook.renderers import (
    Message,
    StreamingMessageHeader,
    StreamingTextDelta,
    StreamingThinkingDelta,
    TextPart,
    ThinkingPart,
    get_renderer,
)
from tinker_cookbook.renderers.base import ensure_list
from tinker_cookbook.renderers.qwen3 import Qwen3Renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer


def _is_message(obj) -> TypeGuard[Message]:
    """Check if object is a Message dict (TypedDict doesn't support isinstance)."""
    return isinstance(obj, dict) and "role" in obj and "content" in obj


# =============================================================================
# Qwen3 parse_response Tests
# =============================================================================


def test_qwen3_parse_response_extracts_thinking():
    """Test Qwen3Renderer.parse_response extracts thinking to ThinkingPart."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = "<think>Let me reason about this.</think>The answer is 42.<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    assert message["role"] == "assistant"

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
    """Test Qwen3Renderer.parse_response puts tool calls in message['tool_calls'], not content."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = '<think>Let me search</think>I will search for that.<tool_call>{"name": "web_search", "arguments": {"query": "weather"}}</tool_call><|im_end|>'
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    content = message["content"]
    assert isinstance(content, list)

    # Content should only have ThinkingPart and TextPart — no tool calls
    assert len(content) == 2
    assert content[0]["type"] == "thinking"
    assert content[0]["thinking"] == "Let me search"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "I will search for that."

    # Tool calls live exclusively in message["tool_calls"]
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
    # Content should be empty — only a tool call, no text or thinking
    assert len(content) == 0

    # Tool call lives in message["tool_calls"]
    assert "tool_calls" in message and len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "calculator"


# =============================================================================
# Qwen3 Disable-Thinking Tests
# =============================================================================


def _get_basic_2turn() -> list[Message]:
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
    ]


def _get_basic_3turn() -> list[Message]:
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
    ]


def _get_basic_4turn() -> list[Message]:
    return [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "user", "content": "And what is 3+3?"},
        {"role": "assistant", "content": "The answer is 6."},
    ]


def test_qwen3_disable_thinking_supervised():
    """
    Test that Qwen3DisableThinkingRenderer adds the correct empty thinking block
    to assistant messages for SFT, matching HF tokenizer with thinking=False.
    """
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    renderer = get_renderer("qwen3_disable_thinking", tokenizer)

    messages = _get_basic_2turn()

    model_input, _ = renderer.build_supervised_example(messages)
    tinker_tokens = model_input.to_ints()
    tinker_decoded = tokenizer.decode(tinker_tokens)

    # Get expected format from official Qwen3 tokenizer with thinking=False
    hf_decoded = tokenizer.apply_chat_template(
        cast(list[dict[str, str]], messages), tokenize=False, thinking=False
    )

    # Verify the complete empty thinking block is present
    assert "<think>\n\n</think>\n\n" in tinker_decoded, (
        f"Renderer must add '<think>\\n\\n</think>\\n\\n' but got: {tinker_decoded}"
    )

    # Verify matches HF
    assert tinker_decoded == hf_decoded.rstrip("\n"), (
        f"Tinker and HuggingFace outputs differ:\n"
        f"TINKER:\n{tinker_decoded!r}\n\n"
        f"HUGGINGFACE:\n{hf_decoded!r}"
    )


def test_qwen3_disable_thinking_generation():
    """Test Qwen3DisableThinkingRenderer generation matches HF with enable_thinking=False."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    cookbook_renderer = get_renderer("qwen3_disable_thinking", tokenizer)

    convo = _get_basic_3turn()

    cookbook_tokens = cookbook_renderer.build_generation_prompt(convo).to_ints()
    hf_tokens = tokenizer.apply_chat_template(
        cast(list[dict[str, str]], convo),
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )

    hf_tokens_list = cast(list[int], hf_tokens)

    assert cookbook_tokens == hf_tokens_list, (
        f"Cookbook tokens: {cookbook_tokens}\n"
        f"Cookbook string: {tokenizer.decode(cookbook_tokens)}\n"
        f"HF tokens: {hf_tokens_list}\n"
        f"HF string: {tokenizer.decode(hf_tokens_list)}"
    )


def test_qwen3_disable_thinking_4turn():
    """
    Test Qwen3DisableThinkingRenderer with 4-turn conversation.
    Only the last assistant message should have the empty thinking block
    (historical thinking is stripped, matching HF behavior).
    """
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    renderer = get_renderer("qwen3_disable_thinking", tokenizer)

    messages = _get_basic_4turn()

    model_input, _ = renderer.build_supervised_example(messages)
    tinker_tokens = model_input.to_ints()
    tinker_decoded = tokenizer.decode(tinker_tokens)

    # Get expected format from HF
    hf_decoded = tokenizer.apply_chat_template(
        cast(list[dict[str, str]], messages), tokenize=False, thinking=False
    )

    assert tinker_decoded == hf_decoded.rstrip("\n"), (
        f"Tinker and HuggingFace outputs differ:\n"
        f"TINKER:\n{tinker_decoded!r}\n\n"
        f"HUGGINGFACE:\n{hf_decoded!r}"
    )


# =============================================================================
# Qwen3 Streaming Parsing Tests
# =============================================================================


def test_qwen3_streaming_simple_text():
    """Test streaming parsing of simple text response without thinking."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = "Hello, world!<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    assert isinstance(deltas[0], StreamingMessageHeader)
    assert deltas[0].role == "assistant"

    assert _is_message(deltas[-1])
    assert deltas[-1]["role"] == "assistant"

    text_content = "".join(d.text for d in deltas if isinstance(d, StreamingTextDelta))
    assert "Hello, world!" in text_content


def test_qwen3_streaming_with_thinking():
    """Test streaming parsing with thinking blocks."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

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


def test_qwen3_streaming_matches_batch():
    """Test that streaming parse produces same final message as batch parse."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = "<think>Step 1: Analyze.\nStep 2: Compute.</think>The result is 123.<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    batch_message, batch_success = renderer.parse_response(response_tokens)
    assert batch_success

    deltas = list(renderer.parse_response_streaming(response_tokens))
    streaming_message = deltas[-1]

    assert _is_message(streaming_message)
    assert streaming_message["role"] == batch_message["role"]
    assert ensure_list(streaming_message["content"]) == ensure_list(batch_message["content"])


def test_qwen3_streaming_content_index_increments():
    """Test that content_index increments when switching content types."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = "<think>thinking</think>text<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    thinking_indices = [d.content_index for d in deltas if isinstance(d, StreamingThinkingDelta)]
    text_indices = [d.content_index for d in deltas if isinstance(d, StreamingTextDelta)]

    if thinking_indices and text_indices:
        assert max(text_indices) > min(thinking_indices)


def test_qwen3_streaming_empty_response():
    """Test streaming parsing of empty/minimal response."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = "<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    assert isinstance(deltas[0], StreamingMessageHeader)
    assert _is_message(deltas[-1])


def test_qwen3_streaming_multiple_think_blocks():
    """Test streaming with multiple interleaved think blocks."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

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


def test_qwen3_streaming_no_unnecessary_buffering():
    """Test that we don't buffer more than necessary when no tag prefix matches."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = "Hello world<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    text_content = "".join(d.text for d in deltas if isinstance(d, StreamingTextDelta))
    assert text_content == "Hello world"


def test_qwen3_streaming_with_emoji():
    """Test that streaming parser handles multi-byte UTF-8 (emoji) correctly."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = "<think>Let me think 🤔</think>Here's a party 🎉!<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    thinking_content = "".join(d.thinking for d in deltas if isinstance(d, StreamingThinkingDelta))
    text_content = "".join(d.text for d in deltas if isinstance(d, StreamingTextDelta))

    assert "�" not in thinking_content, f"Thinking has replacement chars: {thinking_content!r}"
    assert "�" not in text_content, f"Text has replacement chars: {text_content!r}"

    assert "🤔" in thinking_content
    assert "🎉" in text_content


@pytest.mark.parametrize(
    "renderer_name",
    ["qwen3", "qwen3_disable_thinking", "qwen3_instruct"],
)
def test_qwen3_streaming_supported_by_text_variants(renderer_name):
    """All text-only Qwen3 renderer variants support streaming."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = get_renderer(renderer_name, tokenizer)

    response_str = "<think>reasoning</think>answer<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    deltas = list(renderer.parse_response_streaming(response_tokens))

    assert isinstance(deltas[0], StreamingMessageHeader)
    assert _is_message(deltas[-1])


# =============================================================================
# Qwen3 Streaming vs Batch Equivalence Tests
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


class TestQwen3StreamingBatchEquivalence:
    """Verify parse_response_streaming matches parse_response for all patterns."""

    @pytest.fixture
    def renderer(self):
        tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
        return Qwen3Renderer(tokenizer)

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
            "<think>I need to search.</think>I will search."
            '<tool_call>\n{"name": "web_search", "arguments": {"query": "weather"}}\n</tool_call>'
            "<|im_end|>",
        )

    def test_tool_call_without_thinking(self, renderer):
        _assert_streaming_matches_batch(
            renderer,
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "SF"}}\n</tool_call>'
            "<|im_end|>",
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
