from tinker_cookbook.renderer_templates import LLAMA_32_CHAT_TEMPLATE, QWEN_25_VL_CHAT_TEMPLATE
from tinker_cookbook.tokenizer_utils import get_tokenizer

dialog = [
    {
        "role": "system",
        "content": "Cutting Knowledge Date: December 2023\nToday Date: 02 Jun\n\n",
        "train": False,
    },
    {"role": "user", "content": "Hello, how are you?", "train": False},
    {"role": "assistant", "content": "I'm good, thank you!", "train": False},
    {"role": "user", "content": "What is the capital of France?", "train": False},
    {"role": "assistant", "content": "Paris", "train": True},
]


def main():
    tokenizer = get_tokenizer("Qwen/Qwen2.5-VL-7B-Instruct")
    output = tokenizer.apply_chat_template(
        dialog,
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    output_with_chat_template = tokenizer.apply_chat_template(
        dialog,
        chat_template=QWEN_25_VL_CHAT_TEMPLATE,
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    assert output_with_chat_template["input_ids"] == output["input_ids"]  # type: ignore

    tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B-Instruct")

    output = tokenizer.apply_chat_template(
        dialog,
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    output_with_chat_template = tokenizer.apply_chat_template(
        dialog,
        chat_template=LLAMA_32_CHAT_TEMPLATE,
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    assert output_with_chat_template["input_ids"] == output["input_ids"]  # type: ignore


if __name__ == "__main__":
    main()
