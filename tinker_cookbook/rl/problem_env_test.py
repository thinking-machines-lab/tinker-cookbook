"""Tests for ProblemEnv format-reward semantics.

Covers the interaction between ``RoleColonRenderer.parse_response`` (lenient:
EOS counts as a clean parse) and the strict R1-Zero format reward (where
stop-sequence termination is the only "well-formed" outcome). The default
``require_stop_sequence_for_format=True`` preserves the pre-PR behavior
introduced in #339.
"""

from __future__ import annotations

import asyncio

import pytest

from tinker_cookbook.renderers.role_colon import RoleColonRenderer
from tinker_cookbook.rl.problem_env import ProblemEnv
from tinker_cookbook.tokenizer_utils import get_tokenizer

_BASE_MODEL = "Qwen/Qwen3.5-9B-Base"


@pytest.fixture(scope="module")
def renderer() -> RoleColonRenderer:
    return RoleColonRenderer(get_tokenizer(_BASE_MODEL))


class _StubProblem(ProblemEnv):
    def get_question(self) -> str:
        return "What is 2 + 2?"

    def check_answer(self, sample_str: str) -> bool:
        return "4" in sample_str

    def check_format(self, sample_str: str) -> bool:
        return sample_str.strip() != ""

    def get_reference_answer(self) -> str:
        return "4"


def _step(env: ProblemEnv, tokens: list[int]):
    return asyncio.run(env.step(tokens))


def test_eos_only_response_is_format_failure_by_default(renderer: RoleColonRenderer):
    """R1-Zero strict mode: EOS-only termination should NOT earn format reward.
    Preserves the pre-PR penalty so existing R1-Zero recipes are unchanged."""
    env = _StubProblem(renderer, format_coef=0.1)
    tokens = renderer.tokenizer.encode("The answer is 4.", add_special_tokens=False)
    assert isinstance(tokens, list)
    eos = renderer.tokenizer.eos_token_id
    assert isinstance(eos, int)
    tokens.append(eos)

    result = _step(env, tokens)

    assert result.metrics["format"] == 0.0
    assert result.metrics["correct"] == 1.0
    # reward = 0.1 * (0 - 1) + 1.0 = 0.9
    assert result.reward == pytest.approx(0.9)


def test_eos_only_response_can_earn_format_when_lenient(renderer: RoleColonRenderer):
    """Eval-aligned mode: when require_stop_sequence_for_format=False,
    EOS-only termination earns format reward (matches eval grading semantics)."""
    env = _StubProblem(renderer, format_coef=0.1, require_stop_sequence_for_format=False)
    tokens = renderer.tokenizer.encode("The answer is 4.", add_special_tokens=False)
    assert isinstance(tokens, list)
    eos = renderer.tokenizer.eos_token_id
    assert isinstance(eos, int)
    tokens.append(eos)

    result = _step(env, tokens)

    assert result.metrics["format"] == 1.0
    assert result.metrics["correct"] == 1.0
    # reward = 0.1 * (1 - 1) + 1.0 = 1.0
    assert result.reward == pytest.approx(1.0)


def test_stop_sequence_response_is_format_success(renderer: RoleColonRenderer):
    """Stop-sequence termination earns format reward in both modes."""
    text = "The answer is 4.\n\nUser:"
    tokens = renderer.tokenizer.encode(text, add_special_tokens=False)
    assert isinstance(tokens, list)

    for require_stop_seq in (True, False):
        env = _StubProblem(
            renderer,
            format_coef=0.1,
            require_stop_sequence_for_format=require_stop_seq,
        )
        result = _step(env, tokens)
        assert result.metrics["format"] == 1.0, f"require_stop_seq={require_stop_seq}"
        assert result.metrics["correct"] == 1.0


def test_truncated_response_is_format_failure(renderer: RoleColonRenderer):
    """No terminator at all (mid-sentence cutoff) is a format failure regardless of mode."""
    tokens = renderer.tokenizer.encode("The answer is", add_special_tokens=False)
    assert isinstance(tokens, list)

    for require_stop_seq in (True, False):
        env = _StubProblem(
            renderer,
            format_coef=0.1,
            require_stop_sequence_for_format=require_stop_seq,
        )
        result = _step(env, tokens)
        assert result.metrics["format"] == 0.0, f"require_stop_seq={require_stop_seq}"
