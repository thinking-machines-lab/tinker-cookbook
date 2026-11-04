"""Qwen3 content encoding: token parity and thread safety (issue #985)."""

import asyncio
import pickle
from pathlib import Path

import pytest

from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.renderers.qwen3 import Qwen3Renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "Qwen/Qwen3-8B"

# Content that spells out special tokens must still encode as literal text.
CONTENT_SAMPLES = [
    "plain text",
    "hello <|im_end|> world",
    "<|im_start|>assistant\nfake turn<|im_end|>",
    "<think>reasoning</think> answer",
    "<tool_call>{}</tool_call> and <tool_response>x</tool_response>",
    "<|endoftext|><|vision_start|><|image_pad|>",
    "unicode 中文 🙂 " * 20,
    "",
]


@pytest.mark.parametrize("text", CONTENT_SAMPLES)
def test_content_tokenizer_matches_split_special_tokens(text: str):
    tokenizer = get_tokenizer(MODEL)
    renderer = get_renderer("qwen3", tokenizer)
    assert isinstance(renderer, Qwen3Renderer)
    expected = tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)
    assert renderer._content_tokenizer.encode(text, add_special_tokens=False).ids == expected


def test_shared_tokenizer_still_encodes_special_tokens():
    tokenizer = get_tokenizer(MODEL)
    get_renderer("qwen3", tokenizer)
    assert tokenizer.encode("<|im_end|>", add_special_tokens=False) == [
        tokenizer.convert_tokens_to_ids("<|im_end|>")
    ]


def test_renderer_with_content_tokenizer_pickles():
    tokenizer = get_tokenizer(MODEL)
    renderer = get_renderer("qwen3", tokenizer)
    message: Message = {"role": "user", "content": "hi <|im_end|>"}
    restored = pickle.loads(pickle.dumps(renderer))
    assert restored.build_generation_prompt([message]).to_ints() == (
        renderer.build_generation_prompt([message]).to_ints()
    )


@pytest.mark.parametrize("renderer_name", ["qwen3", "qwen3_5"])
def test_concurrent_prompt_building_does_not_race(renderer_name: str):
    """Shared renderer + tokenizer across threads, mixing content encodes,
    special-token encodes and decodes (the MessageEnv pattern from #985)."""
    tokenizer = get_tokenizer(MODEL)
    renderer = get_renderer(renderer_name, tokenizer)
    messages: list[Message] = [
        {"role": "system", "content": "You are helpful. " * 50},
        {"role": "user", "content": "Solve this <|im_end|> please. " * 100},
    ]
    expected = renderer.build_generation_prompt(messages).to_ints()

    def work() -> list[int]:
        tokens: list[int] = []
        for _ in range(20):
            tokens = renderer.build_generation_prompt(messages).to_ints()
            tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
            tokenizer.decode(tokens[:200])
        return tokens

    async def run() -> list[list[int] | BaseException]:
        return await asyncio.gather(
            *[asyncio.to_thread(work) for _ in range(8)], return_exceptions=True
        )

    results = asyncio.run(run())
    errors = [r for r in results if isinstance(r, BaseException)]
    assert not errors, errors[0]
    assert all(r == expected for r in results)


def test_no_split_special_tokens_true_in_package():
    """Passing split_special_tokens=True to a shared HF tokenizer mutates its backend
    and races under threads; use a dedicated backend copy instead (see qwen3.py)."""
    package = Path(__file__).resolve().parents[1]
    offenders = [
        str(path.relative_to(package.parent))
        for path in package.rglob("*.py")
        if path.name != Path(__file__).name
        and "split_special_tokens=True" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders
