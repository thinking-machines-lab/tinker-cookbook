"""Tests for renderer content-byte reporting (bits-per-byte denominator).

``build_supervised_example_with_metadata`` reports the UTF-8 byte count of the
semantic content of loss-weighted messages. The count must exclude everything
the renderer injects (think tags, role headers, tool framing, end-of-turn
markers) and must be identical across renderers for the same messages -- that
is what makes the BPB metric comparable across models.
"""

import functools

import pytest
import tinker
import torch

from tinker_cookbook.renderers import (
    Message,
    Renderer,
    TextPart,
    ThinkingPart,
    TrainOnWhat,
    get_renderer,
)
from tinker_cookbook.renderers.base import message_content_byte_count
from tinker_cookbook.renderers.qwen3 import Qwen3Renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer


@functools.cache
def _qwen3_tokenizer():
    return get_tokenizer("Qwen/Qwen3-8B")


@functools.cache
def _kimi_tokenizer():
    return get_tokenizer("moonshotai/Kimi-K2-Instruct")


USER_TEXT = "What is 2+2?"
RESPONSE_TEXT = "2+2 equals 4."
RESPONSE_BYTES = len(RESPONSE_TEXT.encode("utf-8"))


def _simple_convo() -> list[Message]:
    return [
        Message(role="user", content=USER_TEXT),
        Message(role="assistant", content=RESPONSE_TEXT),
    ]


def test_message_content_byte_count_text_and_thinking():
    message = Message(
        role="assistant",
        content=[
            ThinkingPart(type="thinking", thinking="chain"),
            TextPart(type="text", text="répond"),
        ],
    )
    assert message_content_byte_count(message) == len(b"chain") + len("répond".encode())
    assert message_content_byte_count(message, include_thinking=False) == len("répond".encode())


def test_qwen3_simple_conversation_counts_response_bytes():
    renderer = get_renderer("qwen3", _qwen3_tokenizer())
    example = renderer.build_supervised_example_with_metadata(
        _simple_convo(), train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN
    )
    assert example.trained_content_bytes == RESPONSE_BYTES


def test_kimi_k2_excludes_think_scaffolding_but_trains_on_it():
    """Kimi injects <think></think> into the trained assistant body. Those
    tokens carry loss weight, yet they must not count as content bytes."""
    tokenizer = _kimi_tokenizer()
    renderer = get_renderer("kimi_k2", tokenizer)
    example = renderer.build_supervised_example_with_metadata(
        _simple_convo(), train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN
    )
    tokens = example.model_input.to_ints()
    trained = [t for t, w in zip(tokens, example.weights.tolist()) if w > 0]
    trained_text = tokenizer.decode(trained, skip_special_tokens=False)
    assert "<think></think>" in trained_text  # scaffolding IS trained...
    assert example.trained_content_bytes == RESPONSE_BYTES  # ...but not counted


def test_kimi_k2_and_qwen3_agree_on_content_bytes():
    """The whole point: the denominator is renderer-independent."""
    kimi = get_renderer("kimi_k2", _kimi_tokenizer())
    qwen = get_renderer("qwen3", _qwen3_tokenizer())
    convo = _simple_convo()
    kimi_bytes = kimi.build_supervised_example_with_metadata(convo).trained_content_bytes
    qwen_bytes = qwen.build_supervised_example_with_metadata(convo).trained_content_bytes
    assert kimi_bytes == qwen_bytes == RESPONSE_BYTES


@pytest.mark.parametrize("renderer_name", ["qwen3", "kimi_k2"])
def test_thinking_counted_only_where_rendered(renderer_name: str):
    """Thinking bytes count for the last assistant message (where thinking is
    preserved) but not for historical ones (where it is stripped)."""
    tokenizer = _kimi_tokenizer() if renderer_name == "kimi_k2" else _qwen3_tokenizer()
    renderer = get_renderer(renderer_name, tokenizer)
    thinking = "Let me think about this carefully."
    convo_with_thinking = [
        Message(role="user", content=USER_TEXT),
        Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking=thinking),
                TextPart(type="text", text=RESPONSE_TEXT),
            ],
        ),
    ]
    # Last assistant message: thinking preserved and counted.
    example = renderer.build_supervised_example_with_metadata(
        convo_with_thinking, train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN
    )
    assert example.trained_content_bytes == len(thinking.encode()) + RESPONSE_BYTES

    # Same assistant message in history: thinking stripped, only text counted.
    # (The final assistant message contributes its own RESPONSE_BYTES.)
    convo_multi = convo_with_thinking + [
        Message(role="user", content="And 3+3?"),
        Message(role="assistant", content=RESPONSE_TEXT),
    ]
    example_multi = renderer.build_supervised_example_with_metadata(
        convo_multi, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    assert example_multi.trained_content_bytes == 2 * RESPONSE_BYTES


def test_untrained_messages_contribute_no_content_bytes():
    renderer = get_renderer("qwen3", _qwen3_tokenizer())
    convo = [
        Message(role="user", content="A" * 1000),
        Message(role="assistant", content=RESPONSE_TEXT),
    ]
    example = renderer.build_supervised_example_with_metadata(
        convo, train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN
    )
    assert example.trained_content_bytes == RESPONSE_BYTES


def test_role_colon_counts_content_without_separators():
    renderer = get_renderer("role_colon", _qwen3_tokenizer())
    example = renderer.build_supervised_example_with_metadata(
        _simple_convo(), train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN
    )
    # The " " prefix, "\n\n" suffix, and "User:" stop-overlap are scaffolding.
    assert example.trained_content_bytes == RESPONSE_BYTES


class _LegacyOnlyRenderer(Qwen3Renderer):
    """Simulates a pre-existing custom renderer overriding only the legacy API."""

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        return super().build_supervised_example(messages, train_on_what=train_on_what)


def test_legacy_only_override_falls_back_to_none():
    """A renderer overriding only build_supervised_example keeps working; the
    metadata entry point delegates to it and reports no content bytes."""
    renderer = _LegacyOnlyRenderer(_qwen3_tokenizer())
    baseline = get_renderer("qwen3", _qwen3_tokenizer())
    convo = _simple_convo()

    model_input, weights = renderer.build_supervised_example(convo)
    base_model_input, base_weights = baseline.build_supervised_example(convo)
    assert model_input.to_ints() == base_model_input.to_ints()
    assert torch.equal(weights, base_weights)

    example = renderer.build_supervised_example_with_metadata(convo)
    assert example.model_input.to_ints() == base_model_input.to_ints()
    assert example.trained_content_bytes is None


def test_both_entry_points_agree_for_builtin_renderers():
    """The legacy tuple API and the metadata API must return identical
    tokens/weights for renderers that override the metadata method."""
    for name, tokenizer in [("qwen3", _qwen3_tokenizer()), ("kimi_k2", _kimi_tokenizer())]:
        renderer: Renderer = get_renderer(name, tokenizer)
        convo = _simple_convo()
        model_input, weights = renderer.build_supervised_example(convo)
        example = renderer.build_supervised_example_with_metadata(convo)
        assert model_input.to_ints() == example.model_input.to_ints(), name
        assert torch.equal(weights, example.weights), name
