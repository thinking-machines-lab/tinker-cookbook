"""
Tests for tinker_cookbook renderers against HuggingFace chat templates.

These tests verify that tinker-cookbook renderers produce identical token sequences
to HuggingFace's chat templates. This is important because:

1. The OpenAI-compatible inference endpoint (/chat/completions) uses HuggingFace
   chat templates to render conversations to tokens.
2. Users who train with tinker-cookbook and want to use the OpenAI endpoint for
   inference need their training to use HF-compatible rendering.

For models with thinking capabilities (Qwen3, DeepSeek), we test both the default
renderer (thinking enabled) and the disable_thinking variant.

See docs/rendering.mdx for more details on the rendering system.
See docs/compatible-apis/openai.mdx for the OpenAI-compatible endpoint documentation.
"""

from datetime import date
from typing import Any, cast

import pytest
from transformers.models.auto.tokenization_auto import AutoTokenizer

from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.model_info import get_model_attributes, get_recommended_renderer_name
from tinker_cookbook.renderers import Message, Qwen3Renderer, get_renderer


def _load_tokenizer(model_name: str) -> Any:
    """Load tokenizer with special handling for models that need trust_remote_code."""
    kwargs: dict[str, Any] = {}
    if model_name == "moonshotai/Kimi-K2-Thinking":
        kwargs["trust_remote_code"] = True
        kwargs["revision"] = "612681931a8c906ddb349f8ad0f582cb552189cd"

    return AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)


# =============================================================================
# Basic HF Compatibility Tests (3-turn conversations)
# =============================================================================


@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen3-30B-A3B",
        "deepseek-ai/DeepSeek-V3.1",
        "openai/gpt-oss-20b",
        "moonshotai/Kimi-K2-Thinking",
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
    ],
)
def test_generation_against_hf_chat_templates(model_name: str):
    """Test generation prompt against HF chat templates (3-turn conversation)."""
    tokenizer = _load_tokenizer(model_name)
    attributes = get_model_attributes(model_name)
    image_processor = get_image_processor(model_name) if attributes.is_vl else None
    # not using get_tokenizer(model_name)
    # because we want to test against the original tokenizer from HF, not the mirror
    # gpt_oss HF matches gpt_oss_medium_reasoning and not the default gpt_oss
    render_name = (
        get_recommended_renderer_name(model_name)
        if not model_name.startswith("openai")
        else "gpt_oss_medium_reasoning"
    )
    cookbook_renderer = get_renderer(render_name, tokenizer, image_processor)
    convo: list[Message] = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    if model_name.startswith("meta"):
        today = date.today().strftime("%d %b %Y")
        system_msg: Message = {
            "role": "system",
            "content": f"Cutting Knowledge Date: December 2023\nToday Date: {today}\n\n",
        }
        aug_convo = [system_msg] + convo
    elif model_name.startswith("Qwen"):
        aug_convo = convo
    elif model_name.startswith("deepseek-ai"):
        aug_convo = convo
    elif model_name.startswith("openai"):
        # Thinking field should not be rendered in this case as it is not the last message.
        convo[1]["thinking"] = "The user is sharing a greeting. We should respond politely."
        aug_convo = convo
    elif model_name.startswith("moonshotai"):
        aug_convo = convo
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    cookbook_tokens = cookbook_renderer.build_generation_prompt(aug_convo).to_ints()
    hf_convo = cast(list[dict[str, str]], convo)
    hf_tokens = tokenizer.apply_chat_template(hf_convo, add_generation_prompt=True, tokenize=True)

    assert cookbook_tokens == hf_tokens, (
        f"Cookbook tokens: {cookbook_tokens}\n"
        f"Cookbook string: {tokenizer.decode(cookbook_tokens)}\n"
        f"HF tokens: {hf_tokens}\n"
        f"HF string: {tokenizer.decode(hf_tokens)}"
    )


@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen3-30B-A3B",
        "deepseek-ai/DeepSeek-V3.1",
        "openai/gpt-oss-20b",
        "moonshotai/Kimi-K2-Thinking",
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
    ],
)
def test_supervised_example_against_hf_chat_templates(model_name: str):
    """Test supervised example against HF chat templates (2-turn conversation)."""
    tokenizer = _load_tokenizer(model_name)
    attributes = get_model_attributes(model_name)
    image_processor = get_image_processor(model_name) if attributes.is_vl else None
    # not using get_tokenizer(model_name)
    # because we want to test against the original tokenizer from HF, not the mirror
    render_name = (
        get_recommended_renderer_name(model_name)
        if not model_name.startswith("openai")
        else "gpt_oss_medium_reasoning"
    )
    cookbook_renderer = get_renderer(render_name, tokenizer, image_processor)
    convo: list[Message] = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
    ]

    if model_name.startswith("meta"):
        today = date.today().strftime("%d %b %Y")
        system_msg: Message = {
            "role": "system",
            "content": f"Cutting Knowledge Date: December 2023\nToday Date: {today}\n\n",
        }
        aug_convo = [system_msg] + convo
    elif model_name.startswith("Qwen"):
        # HF includes thinking tags in assistant content for supervised examples.
        aug_convo = convo.copy()
        aug_convo[1]["content"] = "<think>\n\n</think>\n\n I'm fine, thank you!"
    elif model_name.startswith("deepseek-ai"):
        aug_convo = convo
    elif model_name.startswith("openai"):
        # Test thinking field for GPT-OSS is rendered.
        convo[1]["thinking"] = "The user is sharing a greeting. We should respond politely."
        aug_convo = convo
    elif model_name.startswith("moonshotai"):
        # Kimi K2 adds empty <think></think> blocks for assistant messages
        aug_convo = convo.copy()
        aug_convo[1]["content"] = "<think></think>I'm fine, thank you!"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    cookbook_model_input, _ = cookbook_renderer.build_supervised_example(aug_convo)
    cookbook_tokens = cookbook_model_input.to_ints()
    hf_convo = cast(list[dict[str, str]], convo)
    hf_output = tokenizer.apply_chat_template(hf_convo, tokenize=False, add_generation_prompt=False)
    hf_tokens = tokenizer.encode(hf_output.rstrip("\n"), add_special_tokens=False)

    assert cookbook_tokens == hf_tokens, (
        f"Cookbook tokens: {cookbook_tokens}\n"
        f"Cookbook string: {tokenizer.decode(cookbook_tokens)}\n"
        f"HF tokens: {hf_tokens}\n"
        f"HF string: {tokenizer.decode(hf_tokens)}"
    )


# =============================================================================
# Qwen3 Thinking Tests (multi-turn with thinking content)
# =============================================================================


def test_qwen3_2turn_preserves_thinking():
    """
    For 2-turn conversations (user + assistant), thinking should be fully preserved
    since the assistant message is the last message.
    """
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    renderer = Qwen3Renderer(tokenizer)

    messages: list[Message] = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "<think>\nLet me calculate this.\n</think>\n\nThe answer is 4.",
        },
    ]

    model_input, _ = renderer.build_supervised_example(messages)
    tinker_tokens = model_input.to_ints()
    tinker_decoded = tokenizer.decode(tinker_tokens)

    # Get expected format from HuggingFace tokenizer
    hf_decoded = tokenizer.apply_chat_template(cast(list[dict[str, str]], messages), tokenize=False)

    # Tinker and HuggingFace should produce the same output (strip trailing newline from HF)
    assert tinker_decoded == hf_decoded.rstrip("\n"), (
        f"Tinker and HuggingFace outputs differ:\n"
        f"TINKER:\n{tinker_decoded!r}\n\n"
        f"HUGGINGFACE:\n{hf_decoded!r}"
    )


def test_qwen3_4turn_only_last_thinking_preserved():
    """
    For 4-turn conversations, only the last assistant message's thinking should be preserved.
    Earlier assistant thinking blocks are stripped (matching HF behavior with strip_thinking_from_history=True).
    """
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    renderer = Qwen3Renderer(tokenizer)

    messages: list[Message] = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "<think>\nFirst turn reasoning here.\n</think>\n\nThe answer is 4.",
        },
        {"role": "user", "content": "And what is 3+3?"},
        {
            "role": "assistant",
            "content": "<think>\nSecond turn reasoning here.\n</think>\n\nThe answer is 6.",
        },
    ]

    model_input, _ = renderer.build_supervised_example(messages)
    tinker_tokens = model_input.to_ints()
    tinker_decoded = tokenizer.decode(tinker_tokens)

    # Get expected format from HuggingFace tokenizer
    hf_decoded = tokenizer.apply_chat_template(cast(list[dict[str, str]], messages), tokenize=False)

    # Tinker and HuggingFace should produce the same output (strip trailing newline from HF)
    assert tinker_decoded == hf_decoded.rstrip("\n"), (
        f"Tinker and HuggingFace outputs differ:\n"
        f"TINKER:\n{tinker_decoded!r}\n\n"
        f"HUGGINGFACE:\n{hf_decoded!r}"
    )


def test_qwen3_generation_matches_hf():
    """Test Qwen3Renderer generation prompt matches HF with enable_thinking=True (default)."""
    tokenizer = _load_tokenizer("Qwen/Qwen3-8B")
    cookbook_renderer = get_renderer("qwen3", tokenizer)

    convo: list[Message] = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    cookbook_tokens = cookbook_renderer.build_generation_prompt(convo).to_ints()
    hf_tokens = tokenizer.apply_chat_template(
        cast(list[dict[str, str]], convo),
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=True,  # Explicit, though this is the default
    )

    assert cookbook_tokens == hf_tokens, (
        f"Cookbook tokens: {cookbook_tokens}\n"
        f"Cookbook string: {tokenizer.decode(cookbook_tokens)}\n"
        f"HF tokens: {hf_tokens}\n"
        f"HF string: {tokenizer.decode(hf_tokens)}"
    )


# =============================================================================
# Qwen3 Disable Thinking Tests
# =============================================================================


def test_qwen3_disable_thinking_supervised():
    """
    Test that Qwen3DisableThinkingRenderer adds the correct empty thinking block
    to assistant messages for SFT, matching HF tokenizer with thinking=False.
    """
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    renderer = get_renderer("qwen3_disable_thinking", tokenizer)

    messages: list[Message] = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
    ]

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
    tokenizer = _load_tokenizer("Qwen/Qwen3-8B")
    cookbook_renderer = get_renderer("qwen3_disable_thinking", tokenizer)

    convo: list[Message] = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    cookbook_tokens = cookbook_renderer.build_generation_prompt(convo).to_ints()
    hf_tokens = tokenizer.apply_chat_template(
        cast(list[dict[str, str]], convo),
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )

    assert cookbook_tokens == hf_tokens, (
        f"Cookbook tokens: {cookbook_tokens}\n"
        f"Cookbook string: {tokenizer.decode(cookbook_tokens)}\n"
        f"HF tokens: {hf_tokens}\n"
        f"HF string: {tokenizer.decode(hf_tokens)}"
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

    messages: list[Message] = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "user", "content": "And what is 3+3?"},
        {"role": "assistant", "content": "The answer is 6."},
    ]

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
# EOT Parsing Tests
# =============================================================================


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [
        ("Qwen/Qwen3-30B-A3B", "qwen3"),
        ("Qwen/Qwen3-8B", "qwen3_disable_thinking"),
        ("meta-llama/Llama-3.2-1B-Instruct", "llama3"),
        ("deepseek-ai/DeepSeek-V3.1", "deepseekv3"),
        ("deepseek-ai/DeepSeek-V3.1", "deepseekv3_disable_thinking"),
        ("openai/gpt-oss-20b", "gpt_oss_medium_reasoning"),
        ("moonshotai/Kimi-K2-Thinking", "kimi_k2"),
    ],
)
def test_eot_parsing(model_name: str, renderer_name: str):
    """Test EOT token parsing behavior for different renderers using real tokenizers."""
    tokenizer = _load_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # Get the appropriate EOT token for each renderer
    # Note: DeepSeek uses full-width pipes (｜) not ASCII pipes (|)
    eot_tokens = {
        "llama3": "<|eot_id|>",
        "qwen3": "<|im_end|>",
        "qwen3_disable_thinking": "<|im_end|>",
        "deepseekv3": "<｜end▁of▁sentence｜>",  # Full-width pipes
        "deepseekv3_disable_thinking": "<｜end▁of▁sentence｜>",  # Full-width pipes
        "gpt_oss_medium_reasoning": "<|return|>",
        "kimi_k2": "<|im_end|>",
    }
    eot_token = eot_tokens.get(renderer_name)
    if eot_token is None:
        raise ValueError(f"Unknown renderer: {renderer_name}")

    # Test case 1: Normal case with single EOT - should parse correctly
    test_response_with_eot = f"53 + 18 = 71{eot_token}"
    response_tokens = tokenizer.encode(test_response_with_eot, add_special_tokens=False)

    message, format_correct = renderer.parse_response(response_tokens)
    assert message["role"] == "assistant"
    assert message["content"] == "53 + 18 = 71"
    assert format_correct is True

    # Test case 2: No EOT token - should have format=False
    test_response_no_eot = "53 + 18 = 71"
    response_tokens_no_eot = tokenizer.encode(test_response_no_eot, add_special_tokens=False)

    message, format_correct = renderer.parse_response(response_tokens_no_eot)
    assert message["role"] == "assistant"
    assert message["content"] == "53 + 18 = 71"
    assert format_correct is False

    # Test case 3: Double EOT token - should raise ValueError
    test_response_double_eot = f"53 + 18 = 71{eot_token}{eot_token}"
    response_tokens_double_eot = tokenizer.encode(
        test_response_double_eot, add_special_tokens=False
    )

    with pytest.raises(ValueError, match="expected to split into 1 or 2 pieces"):
        _ = renderer.parse_response(response_tokens_double_eot)


# =============================================================================
# strip_thinking_from_history=False Tests (Extension Property)
# =============================================================================


def test_qwen3_strip_thinking_false_preserves_all():
    """
    Test that strip_thinking_from_history=False preserves thinking in ALL messages.
    This mode is used for multi-turn RL where the extension property is needed.
    Note: This mode does NOT match HF behavior - it's a special mode for efficiency.
    """
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    renderer = Qwen3Renderer(tokenizer, strip_thinking_from_history=False)

    messages: list[Message] = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "<think>\nFirst calculation.\n</think>\n\nThe answer is 4.",
        },
        {"role": "user", "content": "And what is 3+3?"},
        {
            "role": "assistant",
            "content": "<think>\nSecond calculation.\n</think>\n\nThe answer is 6.",
        },
    ]

    model_input, _ = renderer.build_supervised_example(messages)
    decoded = tokenizer.decode(model_input.to_ints())

    # Both thinking blocks should be present
    assert decoded.count("<think>") == 2, (
        f"Expected 2 thinking blocks with strip_thinking_from_history=False, got: {decoded}"
    )
    assert "First calculation" in decoded
    assert "Second calculation" in decoded
