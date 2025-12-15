"""
Test that Qwen3Renderer correctly strips thinking from history while preserving it in the last message.
"""

from typing import cast
from transformers import AutoTokenizer

from tinker_cookbook.renderers import Qwen3Renderer, Message, get_renderer
from tinker_cookbook.tests.test_renderers import _load_tokenizer


def test_2_turn_preserves_thinking():
    """
    For 2-turn conversations (user + assistant), the <think> content should be fully preserved
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

    model_input, weights = renderer.build_supervised_example(messages)
    tokens = []
    for chunk in model_input.chunks:
        tokens.extend(chunk.tokens)  # type: ignore[attr-defined]
    tinker_decoded = tokenizer.decode(tokens)

    # Get expected format from HuggingFace tokenizer
    hf_decoded = tokenizer.apply_chat_template(messages, tokenize=False)

    print("\n" + "=" * 80)
    print("2-TURN CONVERSATION - THINKING SHOULD BE PRESERVED:")
    print("=" * 80)
    print("TINKER:")
    print(tinker_decoded)
    print("\nHUGGINGFACE:")
    print(hf_decoded)

    # Tinker and HuggingFace should produce the same output (strip trailing newline from HF)
    assert tinker_decoded == hf_decoded.rstrip("\n"), (
        f"Tinker and HuggingFace outputs differ:\n"
        f"TINKER:\n{tinker_decoded!r}\n\n"
        f"HUGGINGFACE:\n{hf_decoded!r}"
    )


def test_4_turn_only_last_thinking_preserved():
    """
    For 4-turn conversations, only the last assistant message's thinking should be preserved.
    Earlier assistant thinking blocks are stripped.
    """
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    renderer = Qwen3Renderer(tokenizer)

    first_thinking = "First turn reasoning here."
    last_thinking = "Second turn reasoning here."

    messages: list[Message] = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": f"<think>\n{first_thinking}\n</think>\n\nThe answer is 4.",
        },
        {"role": "user", "content": "And what is 3+3?"},
        {"role": "assistant", "content": f"<think>\n{last_thinking}\n</think>\n\nThe answer is 6."},
    ]

    model_input, weights = renderer.build_supervised_example(messages)
    tokens = []
    for chunk in model_input.chunks:
        tokens.extend(chunk.tokens)  # type: ignore[attr-defined]
    tinker_decoded = tokenizer.decode(tokens)

    # Get expected format from HuggingFace tokenizer
    hf_decoded = tokenizer.apply_chat_template(messages, tokenize=False)

    print("\n" + "=" * 80)
    print("4-TURN CONVERSATION - ONLY LAST THINKING SHOULD BE PRESERVED:")
    print("=" * 80)
    print("TINKER:")
    print(tinker_decoded)
    print("\nHUGGINGFACE:")
    print(hf_decoded)

    # Tinker and HuggingFace should produce the same output (strip trailing newline from HF)
    assert tinker_decoded == hf_decoded.rstrip("\n"), (
        f"Tinker and HuggingFace outputs differ:\n"
        f"TINKER:\n{tinker_decoded!r}\n\n"
        f"HUGGINGFACE:\n{hf_decoded!r}"
    )


def test_qwen_enable_thinking_generation():
    """Test Qwen3Renderer to ensure it matches official Qwen3-8B tokenizer format."""
    tokenizer = _load_tokenizer("Qwen/Qwen3-8B")
    cookbook_renderer = get_renderer("qwen3", tokenizer)
    convo: list[Message] = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    cookbook_tokens = cookbook_renderer.build_generation_prompt(convo).to_ints()
    hf_convo = cast(list[dict[str, str]], convo)
    hf_tokens = tokenizer.apply_chat_template(
        hf_convo,
        add_generation_prompt=True,
        tokenize=True,
        # HF will add <think>\n only
        enable_thinking=True,
    )
    print(f"Cookbook tokens: {cookbook_tokens}")
    print(f"Cookbook string: {tokenizer.decode(cookbook_tokens)}")
    print(f"HF tokens: {hf_tokens}")
    print(f"HF string: {tokenizer.decode(hf_tokens)}")

    assert cookbook_tokens == hf_tokens, (
        f"Cookbook tokens: {cookbook_tokens}\n"
        f"Cookbook string: {tokenizer.decode(cookbook_tokens)}\n"
        f"HF tokens: {hf_tokens}\n"
        f"HF string: {tokenizer.decode(hf_tokens)}"
    )


if __name__ == "__main__":
    test_2_turn_preserves_thinking()
    test_4_turn_only_last_thinking_preserved()
    test_qwen_enable_thinking_generation()
