from datetime import date
from typing import Any, cast

import pytest
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import Message, get_renderer
from transformers.models.auto.tokenization_auto import AutoTokenizer


def _load_tokenizer(model_name: str) -> Any:
    """Load tokenizer with special handling for models that need trust_remote_code."""
    kwargs: dict[str, Any] = {}
    if model_name == "moonshotai/Kimi-K2-Thinking":
        kwargs["trust_remote_code"] = True
        kwargs["revision"] = "612681931a8c906ddb349f8ad0f582cb552189cd"

    return AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)


@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/Llama-3.2-1B-Instruct",
        # "Qwen/Qwen3-30B-A3B", TODO: This was broken, will address in another PR.
        "deepseek-ai/DeepSeek-V3.1",
        "openai/gpt-oss-20b",
        "moonshotai/Kimi-K2-Thinking",
    ],
)
def test_generation_against_hf_chat_templates(model_name: str):
    """Test generation prompt against HF chat templates"""
    tokenizer = _load_tokenizer(model_name)
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
    ],
)
def test_supervised_example_against_hf_chat_templates(model_name: str):
    """Test supervised example against HF chat templates"""
    tokenizer = _load_tokenizer(model_name)
    # not using get_tokenizer(model_name)
    # because we want to test against the original tokenizer from HF, not the mirror
    render_name = (
        get_recommended_renderer_name(model_name)
        if not model_name.startswith("openai")
        else "gpt_oss_medium_reasoning"
    )
    cookbook_renderer = get_renderer(render_name, tokenizer)
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


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [
        ("Qwen/Qwen3-30B-A3B", "qwen3"),
        ("meta-llama/Llama-3.2-1B-Instruct", "llama3"),
        ("openai/gpt-oss-20b", "gpt_oss_medium_reasoning"),
        ("moonshotai/Kimi-K2-Thinking", "kimi_k2"),
    ],
)
def test_eot_parsing(model_name: str, renderer_name: str):
    """Test EOT token parsing behavior for different renderers using real tokenizers."""
    tokenizer = _load_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # Get the appropriate EOT token for each renderer
    if renderer_name == "llama3":
        eot_token = "<|eot_id|>"
    elif renderer_name == "qwen3":
        eot_token = "<|im_end|>"
    elif renderer_name.startswith("gpt_oss"):
        eot_token = "<|return|>"
    elif renderer_name == "kimi_k2":
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
    test_generation_against_hf_chat_templates("openai/gpt-oss-20b")
    test_supervised_example_against_hf_chat_templates("openai/gpt-oss-20b")
    test_eot_parsing("Qwen/Qwen3-30B-A3B", "qwen3")
