"""
Tests for Kimi K2.6 renderers.

Verifies that each K2.6 renderer variant matches the HuggingFace chat
template for the corresponding flag combination:

1. ``kimi_k26`` — ``thinking=true, preserve_thinking=false`` (HF default)
2. ``kimi_k26_disable_thinking`` — ``thinking=false, preserve_thinking=false``
3. ``kimi_k26_preserve_thinking`` — ``thinking=true, preserve_thinking=true``
"""

import pytest

from tinker_cookbook.renderers import Message, get_renderer
from tinker_cookbook.renderers.testing_utils import extract_token_ids
from tinker_cookbook.tokenizer_utils import get_tokenizer

KIMI_K26_MODEL = "moonshotai/Kimi-K2.6"


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def kimi_tokenizer():
    """Get the Kimi K2.6 tokenizer (cached per module)."""
    try:
        return get_tokenizer(KIMI_K26_MODEL)
    except ModuleNotFoundError as e:
        if "Kimi-K2" in str(e):
            pytest.skip(f"K2.6 tokenizer has HF module import bug: {e}")
        raise


@pytest.fixture(scope="module")
def kimi_renderer(kimi_tokenizer):
    return get_renderer("kimi_k26", kimi_tokenizer)


@pytest.fixture(scope="module")
def kimi_renderer_disable_thinking(kimi_tokenizer):
    return get_renderer("kimi_k26_disable_thinking", kimi_tokenizer)


@pytest.fixture(scope="module")
def kimi_renderer_preserve_thinking(kimi_tokenizer):
    return get_renderer("kimi_k26_preserve_thinking", kimi_tokenizer)


# =============================================================================
# Multi-turn conversation with historical thinking
# =============================================================================


def _multi_turn_conversation_with_thinking() -> list[Message]:
    """Two full assistant turns each with thinking content, then a user follow-up.

    The historical assistant turn's thinking ("HIST_THINK_A") is what we use
    to distinguish strip-history vs. preserve-history behavior: the default
    renderer must collapse it to <think></think>, while the preserve-thinking
    variant must keep it intact.
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "HIST_THINK_A"},
                {"type": "text", "text": "4"},
            ],
        },
        {"role": "user", "content": "Now what is 3+3?"},
    ]


# =============================================================================
# Prefill tests
# =============================================================================


def test_kimi_k26_default_generation_prompt_has_think_prefill(kimi_tokenizer, kimi_renderer):
    gen_prompt = kimi_renderer.build_generation_prompt(_multi_turn_conversation_with_thinking())
    decoded = kimi_tokenizer.decode(gen_prompt.to_ints())
    assert decoded.endswith("<|im_assistant|>assistant<|im_middle|><think>")


def test_kimi_k26_disable_thinking_generation_prompt(
    kimi_tokenizer, kimi_renderer_disable_thinking
):
    gen_prompt = kimi_renderer_disable_thinking.build_generation_prompt(
        _multi_turn_conversation_with_thinking()
    )
    decoded = kimi_tokenizer.decode(gen_prompt.to_ints())
    assert decoded.endswith("<|im_assistant|>assistant<|im_middle|><think></think>")


def test_kimi_k26_preserve_thinking_generation_prompt(
    kimi_tokenizer, kimi_renderer_preserve_thinking
):
    gen_prompt = kimi_renderer_preserve_thinking.build_generation_prompt(
        _multi_turn_conversation_with_thinking()
    )
    decoded = kimi_tokenizer.decode(gen_prompt.to_ints())
    assert decoded.endswith("<|im_assistant|>assistant<|im_middle|><think>")


# =============================================================================
# preserve_thinking vs default behavior
# =============================================================================


def test_kimi_k26_default_strips_history_thinking(kimi_tokenizer, kimi_renderer):
    """Default renderer must collapse historical thinking to <think></think>."""
    gen_prompt = kimi_renderer.build_generation_prompt(_multi_turn_conversation_with_thinking())
    decoded = kimi_tokenizer.decode(gen_prompt.to_ints())

    assert "HIST_THINK_A" not in decoded
    assert "<think></think>4" in decoded


def test_kimi_k26_preserve_thinking_keeps_history_thinking(
    kimi_tokenizer, kimi_renderer_preserve_thinking
):
    """preserve_thinking variant must keep the historical <think>...</think> block."""
    gen_prompt = kimi_renderer_preserve_thinking.build_generation_prompt(
        _multi_turn_conversation_with_thinking()
    )
    decoded = kimi_tokenizer.decode(gen_prompt.to_ints())

    assert "<think>HIST_THINK_A</think>4" in decoded


# =============================================================================
# HF template equivalence
# =============================================================================


def _hf_tokens(tokenizer, hf_messages, *, thinking: bool, preserve_thinking: bool) -> list[int]:
    return extract_token_ids(
        tokenizer.apply_chat_template(
            hf_messages,
            add_generation_prompt=True,
            tokenize=True,
            thinking=thinking,
            preserve_thinking=preserve_thinking,
        )
    )


def test_kimi_k26_default_matches_hf(kimi_tokenizer, kimi_renderer):
    """Default renderer == HF template with thinking=true, preserve_thinking=false."""
    messages = _multi_turn_conversation_with_thinking()
    cookbook_tokens = kimi_renderer.build_generation_prompt(messages).to_ints()

    hf_messages = [kimi_renderer.to_openai_message(m) for m in messages]
    hf_tokens = _hf_tokens(kimi_tokenizer, hf_messages, thinking=True, preserve_thinking=False)

    assert cookbook_tokens == hf_tokens, (
        f"Cookbook: {kimi_tokenizer.decode(cookbook_tokens)}\n"
        f"HF:       {kimi_tokenizer.decode(hf_tokens)}"
    )


def test_kimi_k26_disable_thinking_matches_hf(kimi_tokenizer, kimi_renderer_disable_thinking):
    """Disable-thinking renderer == HF template with thinking=false, preserve_thinking=false."""
    messages = _multi_turn_conversation_with_thinking()
    cookbook_tokens = kimi_renderer_disable_thinking.build_generation_prompt(messages).to_ints()

    hf_messages = [kimi_renderer_disable_thinking.to_openai_message(m) for m in messages]
    hf_tokens = _hf_tokens(kimi_tokenizer, hf_messages, thinking=False, preserve_thinking=False)

    assert cookbook_tokens == hf_tokens, (
        f"Cookbook: {kimi_tokenizer.decode(cookbook_tokens)}\n"
        f"HF:       {kimi_tokenizer.decode(hf_tokens)}"
    )


def test_kimi_k26_preserve_thinking_matches_hf(kimi_tokenizer, kimi_renderer_preserve_thinking):
    """Preserve-thinking renderer == HF template with thinking=true, preserve_thinking=true."""
    messages = _multi_turn_conversation_with_thinking()
    cookbook_tokens = kimi_renderer_preserve_thinking.build_generation_prompt(messages).to_ints()

    hf_messages = [kimi_renderer_preserve_thinking.to_openai_message(m) for m in messages]
    hf_tokens = _hf_tokens(kimi_tokenizer, hf_messages, thinking=True, preserve_thinking=True)

    assert cookbook_tokens == hf_tokens, (
        f"Cookbook: {kimi_tokenizer.decode(cookbook_tokens)}\n"
        f"HF:       {kimi_tokenizer.decode(hf_tokens)}"
    )


# =============================================================================
# model_info registration
# =============================================================================


def test_kimi_k26_registered_in_model_info():
    from tinker_cookbook import model_info

    attrs = model_info.get_model_attributes("moonshotai/Kimi-K2.6")
    assert attrs.organization == "moonshotai"
    assert attrs.is_vl is True
    assert attrs.recommended_renderers[0] == "kimi_k26"
    assert "kimi_k26_disable_thinking" in attrs.recommended_renderers
    assert "kimi_k26_preserve_thinking" in attrs.recommended_renderers

    assert model_info.get_recommended_renderer_name("moonshotai/Kimi-K2.6") == "kimi_k26"
