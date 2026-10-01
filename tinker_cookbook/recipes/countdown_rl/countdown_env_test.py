"""Unit tests for Countdown environment reward logic."""

from tinker_cookbook.recipes.countdown_rl.countdown_env import (
    evaluate_countdown_expression,
    extract_answer,
)


def test_evaluate_valid_expression():
    is_correct, score = evaluate_countdown_expression("44 + 19 + 35", [44, 19, 35], 98)
    assert is_correct
    assert score == 1.0


def test_evaluate_with_multiplication():
    is_correct, score = evaluate_countdown_expression("3 * 2 + 7", [3, 7, 2], 13)
    assert is_correct
    assert score == 1.0


def test_evaluate_with_parentheses():
    is_correct, score = evaluate_countdown_expression("(10 + 5) * 2", [10, 5, 2], 30)
    assert is_correct
    assert score == 1.0


def test_evaluate_wrong_result():
    is_correct, score = evaluate_countdown_expression("44 + 19", [44, 19, 35], 98)
    assert not is_correct
    # Valid expression with correct numbers but wrong result gets partial credit
    assert 0.3 <= score < 1.0


def test_evaluate_reuses_number():
    # Uses 44 twice but only one 44 is available
    is_correct, score = evaluate_countdown_expression("44 + 44 + 10", [44, 19, 35], 98)
    assert not is_correct
    assert score == 0.0


def test_evaluate_uses_unavailable_number():
    is_correct, score = evaluate_countdown_expression("50 + 48", [44, 19, 35], 98)
    assert not is_correct
    assert score == 0.0


def test_evaluate_division():
    is_correct, score = evaluate_countdown_expression("10 / 2 + 5", [10, 2, 5], 10)
    assert is_correct
    assert score == 1.0


def test_evaluate_invalid_expression():
    is_correct, score = evaluate_countdown_expression("hello world", [44, 19, 35], 98)
    assert not is_correct
    assert score == 0.0


def test_partial_credit_close_result():
    """Expression evaluates to 97 instead of 98 — should get high partial credit."""
    is_correct, score = evaluate_countdown_expression("44 + 19 + 35 - 1", [44, 19, 35, 1], 98)
    # 44+19+35-1=97, target=98, relative error = 1/98 ≈ 0.01
    assert not is_correct
    assert score > 0.55  # 0.3 base + high proximity bonus


def test_partial_credit_far_result():
    """Expression evaluates to something far from target — lower partial credit."""
    is_correct, score = evaluate_countdown_expression("44 + 19", [44, 19, 35], 98)
    # 44+19=63, target=98, relative error = 35/98 ≈ 0.36
    assert not is_correct
    assert 0.3 < score < 0.6


def test_extract_answer_boxed():
    response = "Let me think...\n\\boxed{44 + 19 + 35}"
    assert extract_answer(response) == "44 + 19 + 35"


def test_extract_answer_fallback():
    response = "The answer is:\n44 + 19 + 35"
    assert extract_answer(response) == "44 + 19 + 35"


def test_extract_answer_none():
    response = "I don't know how to solve this."
    assert extract_answer(response) is None


def test_extract_answer_boxed_with_parens():
    response = "\\boxed{(10 + 5) * 2}"
    assert extract_answer(response) == "(10 + 5) * 2"
