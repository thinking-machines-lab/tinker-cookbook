"""Tests for RoleColonRenderer.parse_response.

Regression tests for issue #685: base models that terminate single-turn
responses with EOS (no "\\n\\nUser:" delimiter) must still be reported as a
successful parse, otherwise EnvFromMessageEnv short-circuits with
failed_parse_reward=0 and never grades the answer.
"""

import pytest

from tinker_cookbook.renderers.role_colon import RoleColonRenderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Llama base models recommend role_colon and have a stable EOS token.
_BASE_MODEL = "meta-llama/Llama-3.1-8B"


@pytest.fixture(scope="module")
def renderer() -> RoleColonRenderer:
    return RoleColonRenderer(get_tokenizer(_BASE_MODEL))


def test_parse_response_eos_only_is_success(renderer: RoleColonRenderer):
    """Base model produces a clean answer and terminates with EOS — the common
    single-turn eval case. Must report parse_success=True so the grader runs."""
    answer = "The answer is \\boxed{42}."
    tokens = renderer.tokenizer.encode(answer, add_special_tokens=False)
    tokens.append(renderer.tokenizer.eos_token_id)

    message, parse_success = renderer.parse_response(tokens)

    assert parse_success is True
    assert message["role"] == "assistant"
    assert message["content"] == answer


def test_parse_response_user_delimiter_is_success(renderer: RoleColonRenderer):
    """Model produced the expected stop sequence — parse_success=True."""
    text = "Some answer.\n\nUser:"
    tokens = renderer.tokenizer.encode(text, add_special_tokens=False)

    message, parse_success = renderer.parse_response(tokens)

    assert parse_success is True
    assert message["content"] == "Some answer."


def test_parse_response_no_terminator_is_failure(renderer: RoleColonRenderer):
    """No EOS and no User: delimiter — likely truncated, parse_success=False."""
    text = "Some incomplete answer"
    tokens = renderer.tokenizer.encode(text, add_special_tokens=False)

    message, parse_success = renderer.parse_response(tokens)

    assert parse_success is False
    assert message["content"] == "Some incomplete answer"


def test_parse_response_user_delimiter_with_eos_is_failure(renderer: RoleColonRenderer):
    """If both the User: delimiter AND EOS appear, the response is malformed
    (sampling should have stopped at User:). Keep parse_success=False."""
    text = "Some answer.\n\nUser:"
    tokens = renderer.tokenizer.encode(text, add_special_tokens=False)
    tokens.append(renderer.tokenizer.eos_token_id)

    message, parse_success = renderer.parse_response(tokens)

    assert parse_success is False
    assert message["content"] == "Some answer."


def test_parse_response_multiple_user_delimiters_is_failure(renderer: RoleColonRenderer):
    """Multiple User: delimiters indicate the model role-played the user turn."""
    text = "Answer.\n\nUser: question?\n\nUser:"
    tokens = renderer.tokenizer.encode(text, add_special_tokens=False)

    message, parse_success = renderer.parse_response(tokens)

    assert parse_success is False
    assert message["content"] == "Answer."
