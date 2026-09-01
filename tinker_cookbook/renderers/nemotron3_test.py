import pytest

from tinker_cookbook.renderers import Message, get_renderer
from tinker_cookbook.renderers.testing_utils import extract_token_ids
from tinker_cookbook.renderers.tml import TmlRendererAdapter
from tinker_cookbook.tokenizer_utils import get_tokenizer

NANO_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
SUPER_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
ULTRA_MODEL = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"


@pytest.mark.parametrize(
    "name",
    [
        "nemotron3",
        "nemotron3_low_thinking",
        "nemotron3_disable_thinking",
        "nemotron3_preserve_thinking",
        "nemotron3_ultra",
        "nemotron3_ultra_disable_thinking",
        "nemotron3_ultra_medium_thinking",
        "nemotron3_ultra_preserve_thinking",
    ],
)
def test_nemotron_variants_use_public_renderer(name: str) -> None:
    renderer = get_renderer(name, get_tokenizer(NANO_MODEL))
    assert isinstance(renderer, TmlRendererAdapter)


@pytest.mark.parametrize(
    "model,name,template_args",
    [
        (NANO_MODEL, "nemotron3", {"enable_thinking": True}),
        (
            SUPER_MODEL,
            "nemotron3_low_thinking",
            {"enable_thinking": True, "low_effort": True},
        ),
        (NANO_MODEL, "nemotron3_disable_thinking", {"enable_thinking": False}),
        (ULTRA_MODEL, "nemotron3_ultra", {"enable_thinking": True}),
        (
            ULTRA_MODEL,
            "nemotron3_ultra_medium_thinking",
            {"enable_thinking": True, "medium_effort": True},
        ),
        (ULTRA_MODEL, "nemotron3_ultra_disable_thinking", {"enable_thinking": False}),
    ],
)
def test_nemotron_generation_matches_hf(
    model: str, name: str, template_args: dict[str, bool]
) -> None:
    tokenizer = get_tokenizer(model)
    messages: list[Message] = [Message(role="user", content="What is 2 + 2?")]
    rendered = get_renderer(name, tokenizer).build_generation_prompt(messages).to_ints()
    expected = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **template_args,
    )
    assert rendered == extract_token_ids(expected)


@pytest.mark.parametrize(
    "model,name,truncate_history_thinking",
    [
        (NANO_MODEL, "nemotron3", True),
        (NANO_MODEL, "nemotron3_preserve_thinking", False),
        (ULTRA_MODEL, "nemotron3_ultra", True),
        (ULTRA_MODEL, "nemotron3_ultra_preserve_thinking", False),
    ],
)
def test_nemotron_history_matches_hf(
    model: str, name: str, truncate_history_thinking: bool
) -> None:
    tokenizer = get_tokenizer(model)
    messages: list[Message] = [
        Message(role="user", content="First"),
        Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "reason"},
                {"type": "text", "text": "answer"},
            ],
        ),
        Message(role="user", content="Second"),
    ]
    rendered = get_renderer(name, tokenizer).build_generation_prompt(messages).to_ints()
    expected = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "<think>\nreason\n</think>\nanswer"},
            {"role": "user", "content": "Second"},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        truncate_history_thinking=truncate_history_thinking,
    )
    assert rendered == extract_token_ids(expected)
