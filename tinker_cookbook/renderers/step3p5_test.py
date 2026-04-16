"""Tests specific to Step3p5FlashRenderer (stepfun-ai/Step-3.5-Flash)."""

import pytest
from transformers.models.auto.tokenization_auto import AutoTokenizer

from tinker_cookbook.renderers import Message, get_renderer
from tinker_cookbook.renderers.step3p5 import Step3p5FlashRenderer
from tinker_cookbook.tokenizer_utils import get_tokenizer


def test_step3p5_basic_render():
    """Test basic message rendering with Step3p5FlashRenderer."""
    tokenizer = get_tokenizer("Qwen/Qwen3-0.6B")
    renderer = Step3p5FlashRenderer(tokenizer)

    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello!"),
        Message(role="assistant", content="Hi there!"),
    ]

    model_input, weights = renderer.build_supervised_example(messages)
    tokens = model_input.to_ints()
    assert len(tokens) > 0
    assert len(weights) == len(tokens)


def test_step3p5_thinking_content():
    """Test that thinking content is rendered with <think>... tags."""
    tokenizer = get_tokenizer("Qwen/Qwen3-0.6B")
    renderer = Step3p5FlashRenderer(tokenizer)

    messages = [
        Message(role="user", content="What is 2+2?"),
        Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "Let me compute 2+2."},
                {"type": "text", "text": "The answer is 4."},
            ],
        ),
    ]

    model_input, weights = renderer.build_supervised_example(messages)
    rendered_str = tokenizer.decode(model_input.to_ints())
    assert "<think>" in rendered_str
    assert "" in rendered_str
    assert "Let me compute" in rendered_str


def test_step3p5_parse_response():
    """Test parse_response extracts thinking and text parts correctly."""
    tokenizer = get_tokenizer("Qwen/Qwen3-0.6B")
    renderer = Step3p5FlashRenderer(tokenizer)

    response_str = "<think>Thinking step here.Final answer here.<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    assert message["role"] == "assistant"
    content = message["content"]
    assert isinstance(content, list)
    thinking_parts = [p for p in content if p["type"] == "thinking"]
    text_parts = [p for p in content if p["type"] == "text"]
    assert len(thinking_parts) == 1
    assert thinking_parts[0]["thinking"] == "Thinking step here."
    assert any("Final answer" in p["text"] for p in text_parts)


def test_step3p5_get_renderer_factory():
    """Test that get_renderer('step3p5', ...) returns a Step3p5FlashRenderer."""
    tokenizer = get_tokenizer("Qwen/Qwen3-0.6B")
    renderer = get_renderer("step3p5", tokenizer)
    assert isinstance(renderer, Step3p5FlashRenderer)


def test_step3p5_tool_call():
    """Test tool call rendering."""
    tokenizer = get_tokenizer("Qwen/Qwen3-0.6B")
    renderer = Step3p5FlashRenderer(tokenizer)

    messages = [
        Message(
            role="assistant",
            content=[
                {"type": "text", "text": "Let me search for that."},
            ],
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query": "weather today"}',
                    },
                }
            ],
        ),
    ]

    model_input, weights = renderer.build_supervised_example(messages)
    rendered_str = tokenizer.decode(model_input.to_ints())
    assert "<tool_call>" in rendered_str
    assert "web_search" in rendered_str


def test_step3p5_stop_token():
    """Test that stop token is correctly identified."""
    tokenizer = get_tokenizer("Qwen/Qwen3-0.6B")
    renderer = Step3p5FlashRenderer(tokenizer)

    stop_seqs = renderer.get_stop_sequences()
    assert len(stop_seqs) == 1
    im_end_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    assert len(im_end_token) == 1
    assert stop_seqs[0] == im_end_token[0]