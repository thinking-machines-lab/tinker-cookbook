"""
Test case for Qwen3DisableThinkingRenderer to ensure it matches official Qwen3-8B tokenizer format.
Related to: https://github.com/thinking-machines-lab/tinker-cookbook/issues/176
"""

from transformers import AutoTokenizer

from tinker_cookbook.renderers import Qwen3DisableThinkingRenderer, Message


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


if __name__ == "__main__":
    test_qwen3_disable_thinking_format()
