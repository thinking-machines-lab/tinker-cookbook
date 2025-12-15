"""
Test that Qwen3Renderer correctly strips thinking from history while preserving it in the last message.
"""

from transformers import AutoTokenizer

from tinker_cookbook.renderers import Qwen3Renderer, Message


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
        tokens.extend(chunk.tokens)
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
        tokens.extend(chunk.tokens)
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


if __name__ == "__main__":
    test_2_turn_preserves_thinking()
    test_4_turn_only_last_thinking_preserved()
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
