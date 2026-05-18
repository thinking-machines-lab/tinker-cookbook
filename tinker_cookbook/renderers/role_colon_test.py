"""Tests for RoleColonRenderer.parse_response.

Regression tests for issue #685: base models that terminate single-turn
responses with EOS (no "\\n\\nUser:" delimiter) must report ``ParseTermination.EOS``
(``is_clean=True``), otherwise EnvFromMessageEnv short-circuits with
failed_parse_reward=0 and never grades the answer.
"""

import pytest

from tinker_cookbook.renderers.base import ParseTermination
from tinker_cookbook.renderers.role_colon import RoleColonRenderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Llama base models recommend role_colon and have a stable EOS token.
_BASE_MODEL = "meta-llama/Llama-3.1-8B"


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
