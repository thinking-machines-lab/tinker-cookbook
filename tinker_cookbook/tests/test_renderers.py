from datetime import date

import pytest
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import Message, get_renderer
from transformers.models.auto.tokenization_auto import AutoTokenizer


@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen3-30B-A3B",
        "deepseek-ai/DeepSeek-V3.1",
        "openai/gpt-oss-20b",
    ],
)
def test_against_hf_chat_templates(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # not using get_tokenizer(model_name)
    # because we want to test against the original tokenizer from HF, not the mirror
    # gpt_oss HF matches gpt_oss_medium_reasoning and not the default gpt_oss
    render_name = (
        get_recommended_renderer_name(model_name)
        if not model_name.startswith("openai")
        else "gpt_oss_medium_reasoning"
    )
    cookbook_renderer = get_renderer(render_name, tokenizer)
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
        aug_convo = convo
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    cookbook_tokens = cookbook_renderer.build_generation_prompt(aug_convo).to_ints()
    hf_tokens = tokenizer.apply_chat_template(convo, add_generation_prompt=True)
    assert cookbook_tokens == hf_tokens, (
        f"Cookbook tokens: {cookbook_tokens}\n"
        f"Cookbook string: {tokenizer.decode(cookbook_tokens)}\n"
        f"HF tokens: {hf_tokens}\n"
        f"HF string: {tokenizer.decode(hf_tokens)}"
    )


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [
        ("Qwen/Qwen3-30B-A3B", "qwen3"),
        ("meta-llama/Llama-3.2-1B-Instruct", "llama3"),
        ("openai/gpt-oss-20b", "gpt_oss_medium_reasoning"),
    ],
)
def test_eot_parsing(model_name: str, renderer_name: str):
    """Test EOT token parsing behavior for different renderers using real tokenizers."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    renderer = get_renderer(renderer_name, tokenizer)

    # Get the appropriate EOT token for each renderer
    if renderer_name == "llama3":
        eot_token = "<|eot_id|>"
    elif renderer_name == "qwen3":
        eot_token = "<|im_end|>"
    elif renderer_name.startswith("gpt_oss"):
        eot_token = "<|return|>"
    else:
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


def test_gptoss_channels():
    """Test GPT-OSS multi-channel rendering for assistant messages with thinking/content fields."""
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", use_fast=True)
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer)

    # Test case 1: Training format with both thinking and content
    messages_training: list[Message] = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "thinking": "The user wants me to calculate 2+2. Let me compute: 2+2 = 4.",
            "content": "The answer is 4.",
        },
    ]

    tokens, weights = renderer.build_supervised_example(messages_training)
    decoded = tokenizer.decode(tokens.tolist())

    assert "<|channel|>analysis<|message|>The user wants me to calculate 2+2" in decoded, "Should have analysis channel"
    assert "<|channel|>final<|message|>The answer is 4" in decoded, "Should have final channel"
    analysis_pos = decoded.find("<|channel|>analysis")
    final_pos = decoded.find("<|channel|>final")
    assert analysis_pos < final_pos, "Analysis channel should come before final channel"

    # Test case 2: Inference format excludes thinking from non-last messages
    messages_inference: list[Message] = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "thinking": "Previous reasoning",
            "content": "Previous answer.",
        },
        {"role": "user", "content": "What about 3+3?"},
    ]

    prompt_tokens = renderer.build_generation_prompt(messages_inference)
    decoded_inference = tokenizer.decode(prompt_tokens.to_ints())

    assert "Previous reasoning" not in decoded_inference
    assert "Previous answer." in decoded_inference

    # Test case 3: Content-only messages without reasoning still work
    messages_content_only: list[Message] = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
    ]

    tokens_compat, weights_compat = renderer.build_supervised_example(messages_content_only)
    decoded_compat = tokenizer.decode(tokens_compat.tolist())

    assert "<|channel|>analysis" not in decoded_compat
    assert "<|channel|>final<|message|>The answer is 4." in decoded_compat


if __name__ == "__main__":
    # test_against_hf_chat_templates("meta-llama/Llama-3.2-1B-Instruct")
    # test_against_hf_chat_templates("Qwen/Qwen2.5-VL-3B-Instruct")
    test_eot_parsing("Qwen/Qwen3-30B-A3B", "qwen3")
    test_gptoss_channels()
