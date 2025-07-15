from datetime import date

import pytest
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import Message, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer


@pytest.mark.parametrize(
    "model_name", ["meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"]
)
def test_against_hf_chat_templates(model_name: str):
    tokenizer = get_tokenizer(model_name)
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
        system_message = Message(role="system", content="You are a helpful assistant.")
        aug_convo = [system_message] + convo
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    cookbook_tokens = cookbook_renderer.build_generation_prompt(aug_convo).to_ints()
    hf_tokens = tokenizer.apply_chat_template(convo, add_generation_prompt=True)  # type: ignore
    assert cookbook_tokens == hf_tokens, (
        f"Cookbook tokens: {cookbook_tokens}\n"
        f"Cookbook string: {tokenizer.decode(cookbook_tokens)}\n"
        f"HF tokens: {hf_tokens}\n"
        f"HF string: {tokenizer.decode(hf_tokens)}"
    )


if __name__ == "__main__":
    # test_against_hf_chat_templates("meta-llama/Llama-3.2-1B-Instruct")
    test_against_hf_chat_templates("Qwen/Qwen2.5-VL-3B-Instruct")
