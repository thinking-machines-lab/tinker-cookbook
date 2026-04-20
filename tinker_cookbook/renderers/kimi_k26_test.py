"""
Tests for Kimi K2.6 renderers.

Verifies that each K2.6 renderer variant matches the HuggingFace chat
template for the corresponding flag combination:

1. ``kimi_k26`` — ``thinking=true, preserve_thinking=false`` (HF default)
2. ``kimi_k26_disable_thinking`` — ``thinking=false, preserve_thinking=false``
3. ``kimi_k26_preserve_thinking`` — ``thinking=true, preserve_thinking=true``
"""

import pytest

from tinker_cookbook.renderers import Message, ToolCall, ToolSpec, get_renderer
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


def _hf_tokens(
    tokenizer,
    hf_messages,
    *,
    thinking: bool,
    preserve_thinking: bool,
    tools: list | None = None,
    add_generation_prompt: bool = True,
) -> list[int]:
    return extract_token_ids(
        tokenizer.apply_chat_template(
            hf_messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
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
# preserve_thinking guardrails: supervised + tool-calling
# =============================================================================


def _preserve_thinking_multi_assistant_conversation() -> list[Message]:
    """Two assistant turns, each with distinct thinking. preserve_thinking=true
    must retain both; default K2.5 behavior strips the first."""
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
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "CURRENT_THINK"},
                {"type": "text", "text": "6"},
            ],
        },
    ]


def test_kimi_k26_preserve_thinking_supervised_matches_hf(
    kimi_tokenizer, kimi_renderer_preserve_thinking
):
    """build_supervised_example output with preserve_thinking=true keeps every
    historical <think>...</think> block and matches HF apply_chat_template."""
    messages = _preserve_thinking_multi_assistant_conversation()
    model_input, _ = kimi_renderer_preserve_thinking.build_supervised_example(messages)
    cookbook_tokens = model_input.to_ints()

    hf_messages = [kimi_renderer_preserve_thinking.to_openai_message(m) for m in messages]
    hf_tokens = _hf_tokens(
        kimi_tokenizer,
        hf_messages,
        thinking=True,
        preserve_thinking=True,
        add_generation_prompt=False,
    )

    assert cookbook_tokens == hf_tokens, (
        f"Cookbook: {kimi_tokenizer.decode(cookbook_tokens)}\n"
        f"HF:       {kimi_tokenizer.decode(hf_tokens)}"
    )
    decoded = kimi_tokenizer.decode(cookbook_tokens)
    assert "<think>HIST_THINK_A</think>4" in decoded
    assert "<think>CURRENT_THINK</think>6" in decoded


def _get_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state"},
            },
            "required": ["location"],
        },
    )


def test_kimi_k26_preserve_thinking_tool_call_matches_hf(
    kimi_tokenizer, kimi_renderer_preserve_thinking
):
    """Multi-turn tool-calling with preserve_thinking=true: a completed non-tool
    assistant turn exists before the tool-calling turn, so the historical
    thinking block must be preserved. Asserts HF token equivalence."""
    tools = [_get_tool_spec()]
    tool_call = ToolCall(
        id="functions.get_weather:0",
        function=ToolCall.FunctionBody(
            name="get_weather",
            arguments='{"location": "New York, NY"}',
        ),
    )
    messages: list[Message] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How are you today?"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "HIST_REASONING"},
                {"type": "text", "text": "I'm doing great!"},
            ],
        },
        {"role": "user", "content": "What's the weather in NYC?"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "CURRENT_REASONING"},
                {"type": "text", "text": ""},
            ],
            "tool_calls": [tool_call],
        },
        {
            "role": "tool",
            "name": "get_weather",
            "tool_call_id": "functions.get_weather:0",
            "content": '{"temperature": 72, "condition": "sunny"}',
        },
    ]

    openai_tools = [{"type": "function", "function": tool} for tool in tools]
    prefix_messages = kimi_renderer_preserve_thinking.create_conversation_prefix_with_tools(
        tools, system_prompt="You are a helpful assistant."
    )
    prefix_messages = [m for m in prefix_messages if m["role"] == "tool_declare"]
    full_messages = prefix_messages + messages
    cookbook_tokens = kimi_renderer_preserve_thinking.build_generation_prompt(
        full_messages
    ).to_ints()

    hf_messages = [kimi_renderer_preserve_thinking.to_openai_message(m) for m in messages]
    hf_tokens = _hf_tokens(
        kimi_tokenizer,
        hf_messages,
        thinking=True,
        preserve_thinking=True,
        tools=openai_tools,
    )

    assert cookbook_tokens == hf_tokens, (
        f"Cookbook: {kimi_tokenizer.decode(cookbook_tokens)}\n"
        f"HF:       {kimi_tokenizer.decode(hf_tokens)}"
    )
    decoded = kimi_tokenizer.decode(cookbook_tokens)
    assert "<think>HIST_REASONING</think>I'm doing great!" in decoded
    assert "<think>CURRENT_REASONING</think>" in decoded


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
