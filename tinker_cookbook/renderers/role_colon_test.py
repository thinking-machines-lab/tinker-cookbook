"""Tests for RoleColonRenderer.parse_response.

Regression tests for issue #685: base models that terminate single-turn
responses with EOS (no "\\n\\nUser:" delimiter) must report ``ParseTermination.EOS``
(``is_clean=True``), otherwise EnvFromMessageEnv short-circuits with
failed_parse_reward=0 and never grades the answer.
"""

import asyncio
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer as RustTokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from tinker_cookbook.renderers.base import ParseTermination
from tinker_cookbook.renderers.role_colon import RoleColonRenderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Qwen3.5 base models recommend role_colon and have a stable EOS token.
_BASE_MODEL = "Qwen/Qwen3.5-9B-Base"


@pytest.fixture(scope="module")
def renderer() -> RoleColonRenderer:
    return RoleColonRenderer(get_tokenizer(_BASE_MODEL))


def test_parse_response_eos_only_is_eos(renderer: RoleColonRenderer):
    """Base model produces a clean answer and terminates with EOS — the common
    single-turn eval case. Must return EOS so eval grading runs but strict
    R1-Zero format reward can still distinguish it."""
    answer = "The answer is \\boxed{42}."
    tokens = renderer.tokenizer.encode(answer, add_special_tokens=False)
    assert isinstance(tokens, list)
    eos_token_id = renderer.tokenizer.eos_token_id
    assert isinstance(eos_token_id, int)
    tokens.append(eos_token_id)

    message, termination = renderer.parse_response(tokens)

    assert termination == ParseTermination.EOS
    assert termination.is_clean
    assert not termination.is_stop_sequence
    assert message["role"] == "assistant"
    assert message["content"] == answer


def test_parse_response_user_delimiter_is_stop_sequence(renderer: RoleColonRenderer):
    """Model produced the expected stop sequence — STOP_SEQUENCE."""
    text = "Some answer.\n\nUser:"
    tokens = renderer.tokenizer.encode(text, add_special_tokens=False)

    message, termination = renderer.parse_response(tokens)

    assert termination == ParseTermination.STOP_SEQUENCE
    assert termination.is_clean
    assert termination.is_stop_sequence
    assert message["content"] == "Some answer."


def test_parse_response_no_terminator_is_malformed(renderer: RoleColonRenderer):
    """No EOS and no User: delimiter — likely truncated, MALFORMED."""
    text = "Some incomplete answer"
    tokens = renderer.tokenizer.encode(text, add_special_tokens=False)

    message, termination = renderer.parse_response(tokens)

    assert termination == ParseTermination.MALFORMED
    assert not termination.is_clean
    assert message["content"] == "Some incomplete answer"


def test_parse_response_user_delimiter_with_eos_is_malformed(renderer: RoleColonRenderer):
    """If both the User: delimiter AND EOS appear, the response is malformed
    (sampling should have stopped at User:)."""
    text = "Some answer.\n\nUser:"
    tokens = renderer.tokenizer.encode(text, add_special_tokens=False)
    assert isinstance(tokens, list)
    eos_token_id = renderer.tokenizer.eos_token_id
    assert isinstance(eos_token_id, int)
    tokens.append(eos_token_id)

    message, termination = renderer.parse_response(tokens)

    assert termination == ParseTermination.MALFORMED
    assert message["content"] == "Some answer."


def test_parse_response_multiple_user_delimiters_is_malformed(renderer: RoleColonRenderer):
    """Multiple User: delimiters indicate the model role-played the user turn."""
    text = "Answer.\n\nUser: question?\n\nUser:"
    tokens = renderer.tokenizer.encode(text, add_special_tokens=False)

    message, termination = renderer.parse_response(tokens)

    assert termination == ParseTermination.MALFORMED
    assert message["content"] == "Answer."


@pytest.mark.parametrize(
    "content",
    [" leading", "trailing ", "  both  ", " "],
    ids=["leading", "trailing", "both", "only-a-space"],
)
def test_parse_response_keeps_the_content_whitespace_it_rendered(
    renderer: RoleColonRenderer, content: str
):
    """Content whose own whitespace is load-bearing: `" leading"` and `"leading"` render
    differently, so they must parse back differently."""
    rendered = " " + content + "\n\n"
    tokens = renderer.tokenizer.encode(rendered + "User:", add_special_tokens=False)

    message, termination = renderer.parse_response(tokens)

    assert termination == ParseTermination.STOP_SEQUENCE
    assert message["content"] == content


@pytest.fixture
def multiple_eos_renderer(tmp_path: Path) -> RoleColonRenderer:
    # Model generation can stop on tokens absent from tokenizer.eos_token_id.
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=RustTokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
        eos_token="<|im_end|>",
        additional_special_tokens=["<|endoftext|>", "\\boxed{42}", "\n\nUser:"],
    )
    tokenizer.save_pretrained(tmp_path)
    (tmp_path / "generation_config.json").write_text(
        json.dumps(
            {"eos_token_id": tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])}
        )
    )
    return RoleColonRenderer(get_tokenizer(str(tmp_path)))


@pytest.mark.parametrize("ending", ["<|im_end|>", "<|endoftext|>"])
def test_parse_response_accepts_model_eos(
    multiple_eos_renderer: RoleColonRenderer, ending: str
) -> None:
    tokens = multiple_eos_renderer.tokenizer.encode("\\boxed{42}" + ending)
    original_tokens = tokens.copy()

    message, termination = multiple_eos_renderer.parse_response(tokens)

    assert termination == ParseTermination.EOS
    assert message["content"] == "\\boxed{42}"
    assert tokens == original_tokens


@pytest.mark.parametrize(
    "suffix",
    [
        "",
        "<|endoftext|><|endoftext|>",
        "<|endoftext|><|im_end|>",
        "<|endoftext|>\\boxed{42}",
        "<|endoftext|>\n\nUser:",
        "\n\nUser:<|endoftext|>",
    ],
)
def test_parse_response_rejects_invalid_model_eos(
    multiple_eos_renderer: RoleColonRenderer, suffix: str
) -> None:
    tokens = multiple_eos_renderer.tokenizer.encode("\\boxed{42}" + suffix)

    _, termination = multiple_eos_renderer.parse_response(tokens)

    assert termination == ParseTermination.MALFORMED


@pytest.mark.parametrize(
    "ending,expected_format",
    [("<|endoftext|>", 1), ("<|im_end|>", 1), ("\n\nUser:", 1), ("", 0)],
)
def test_deepmath_rewards_clean_termination(
    multiple_eos_renderer: RoleColonRenderer,
    monkeypatch: pytest.MonkeyPatch,
    ending: str,
    expected_format: int,
) -> None:
    from datasets import Dataset

    from tinker_cookbook.recipes.math_rl import math_env

    data = Dataset.from_list([{"question": "6 * 7?", "final_answer": "42"}])
    monkeypatch.setattr(math_env, "load_dataset", lambda *args, **kwargs: data)
    dataset = math_env.DeepMathDataset(1, 1, multiple_eos_renderer)
    builder = dataset.get_batch(0)[0]
    env = asyncio.run(builder.make_envs())[0]
    tokens = multiple_eos_renderer.tokenizer.encode("\\boxed{42}" + ending)

    result = asyncio.run(env.step(tokens))

    assert result.metrics == {"correct": 1.0, "format": expected_format}
    assert result.reward == pytest.approx(1.0 if expected_format else 0.9)
