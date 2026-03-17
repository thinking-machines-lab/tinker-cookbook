"""Tests specific to Qwen3 renderers (parse_response, disable-thinking behavior)."""

from typing import cast

from transformers.models.auto.tokenization_auto import AutoTokenizer

from tinker_cookbook.renderers import (
    Message,
    TextPart,
    ThinkingPart,
    get_renderer,
)
from tinker_cookbook.renderers.qwen3 import Qwen3Renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

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
