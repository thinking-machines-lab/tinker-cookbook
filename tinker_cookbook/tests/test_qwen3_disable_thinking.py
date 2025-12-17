"""
Test case for Qwen3DisableThinkingRenderer to ensure it matches official Qwen3-8B tokenizer format.
Related to: https://github.com/thinking-machines-lab/tinker-cookbook/issues/176
"""

from typing import cast
from transformers import AutoTokenizer

from tinker_cookbook.renderers import Qwen3DisableThinkingRenderer, Message, get_renderer
from tinker_cookbook.tests.test_renderers import _load_tokenizer


def test_qwen3_disable_thinking_format():
    """
    Test that Qwen3DisableThinkingRenderer adds the correct empty thinking block
    to assistant messages, matching the official Qwen3-8B tokenizer format.
    """
    # Load tokenizer
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Create renderer
    renderer = Qwen3DisableThinkingRenderer(tokenizer)

    # Test messages
    messages: list[Message] = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
    ]

    # Build supervised example
    model_input, weights = renderer.build_supervised_example(messages)
    tokens = []
    for chunk in model_input.chunks:
        tokens.extend(chunk.tokens)  # type: ignore[attr-defined]
    decoded = tokenizer.decode(tokens)

    # Get expected format from official Qwen3-8B tokenizer
    official_format = tokenizer.apply_chat_template(messages, tokenize=False, thinking=False)

    print("\n" + "=" * 80)
    print("ACTUAL (tinker-cookbook renderer):")
    print("=" * 80)
    print(decoded)

    print("\n" + "=" * 80)
    print("EXPECTED (Qwen3-8B tokenizer):")
    print("=" * 80)
    print(official_format)

    # Must have the COMPLETE empty thinking block
    # Not just <think>\n but the full <think>\n\n</think>\n\n
    assert "<think>\n\n</think>\n\n" in decoded, (
        f"Renderer must add '<think>\\n\\n</think>\\n\\n' but got: {decoded}"
    )


def test_qwen_disable_thinking_generation():
    """Test Qwen3DisableThinkingRenderer to ensure it matches official Qwen3-8B tokenizer format."""
    tokenizer = _load_tokenizer("Qwen/Qwen3-8B")
    cookbook_renderer = get_renderer("qwen3_disable_thinking", tokenizer)
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
        # HF will add <think>\n\n</think>\n\n non-thinking tokens.
        enable_thinking=False,
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
    test_qwen3_disable_thinking_format()
    test_qwen_disable_thinking_generation()
