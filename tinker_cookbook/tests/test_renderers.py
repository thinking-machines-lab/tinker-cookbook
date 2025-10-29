from datetime import date

import pytest
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import Message, get_renderer
from transformers.models.auto.tokenization_auto import AutoTokenizer


@pytest.mark.parametrize("model_name", ["meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen3-30B-A3B"])
def test_against_hf_chat_templates(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # not using get_tokenizer(model_name)
    # because we want to test against the original tokenizer from HF, not the mirror
    cookbook_renderer = get_renderer(get_recommended_renderer_name(model_name), tokenizer)
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


if __name__ == "__main__":
    # test_against_hf_chat_templates("meta-llama/Llama-3.2-1B-Instruct")
    # test_against_hf_chat_templates("Qwen/Qwen2.5-VL-3B-Instruct")
    test_eot_parsing("Qwen/Qwen3-30B-A3B", "qwen3")
