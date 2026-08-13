"""Tests for RoleColon rendering and parsing.

Regression tests for issue #685: base models that terminate single-turn
responses with EOS (no "\\n\\nUser:" delimiter) must report ``ParseTermination.EOS``
(``is_clean=True``), otherwise EnvFromMessageEnv short-circuits with
failed_parse_reward=0 and never grades the answer.
"""

import pytest

from tinker_cookbook.renderers.base import Message, ParseTermination, TrainOnWhat
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


def test_supervised_trains_the_full_stop_on_intermediate_turns(renderer: RoleColonRenderer):
    messages: list[Message] = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]

    model_input, weights = renderer.build_supervised_example(
        messages, TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    trained = [
        token
        for token, weight in zip(model_input.to_ints(), weights.tolist(), strict=True)
        if weight > 0
    ]

    assert renderer.tokenizer.decode(trained) == " a1\n\nUser: a2\n\nUser:"

    last_input, last_weights = renderer.build_supervised_example(
        messages, TrainOnWhat.LAST_ASSISTANT_MESSAGE
    )
    last_trained = [
        token
        for token, weight in zip(last_input.to_ints(), last_weights.tolist(), strict=True)
        if weight > 0
    ]
    assert renderer.tokenizer.decode(last_trained) == " a2\n\nUser:"


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
