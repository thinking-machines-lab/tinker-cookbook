"""Tests for math reward functions."""

import pytest

from tinker_cookbook.rewards.math_rewards import (
    extract_boxed_answer,
    extract_gsm8k_answer,
    grade_math_answer,
    grade_math_answer_safe,
    normalize_answer,
)


class TestExtractBoxedAnswer:
    def test_simple(self):
        assert extract_boxed_answer("The answer is \\boxed{42}") == "42"

    def test_nested_braces(self):
        assert extract_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_multiple_boxed_returns_last(self):
        assert extract_boxed_answer("\\boxed{1} and \\boxed{2}") == "2"

    def test_no_braces(self):
        assert extract_boxed_answer("\\boxed 42") == "42"

    def test_no_boxed_raises(self):
        with pytest.raises(ValueError, match="No boxed strings found"):
            extract_boxed_answer("no boxed answer here")


class TestNormalizeAnswer:
    def test_none(self):
        assert normalize_answer(None) is None

    def test_strip_whitespace(self):
        assert normalize_answer("  42  ") == "42"

    def test_remove_text_wrapper(self):
        assert normalize_answer("\\text{hello}") == "hello"

    def test_fix_frac(self):
        assert normalize_answer("\\frac12") == "\\frac{1}{2}"

    def test_half_decimal(self):
        assert normalize_answer("0.5") == "\\frac{1}{2}"

    def test_slash_to_frac(self):
        assert normalize_answer("3/4") == "\\frac{3}{4}"


class TestGradeMathAnswer:
    def test_exact_match(self):
        assert grade_math_answer("42", "42") is True

    def test_equivalent_fraction(self):
        assert grade_math_answer("0.5", "\\frac{1}{2}") is True

    def test_wrong_answer(self):
        assert grade_math_answer("43", "42") is False

    def test_none_answer(self):
        assert grade_math_answer(None, "42") is False

    def test_empty_string(self):
        assert grade_math_answer("", "42") is False

    def test_tuple_match(self):
        assert grade_math_answer("(1, 2)", "(1, 2)") is True

    def test_tuple_mismatch(self):
        assert grade_math_answer("(1, 2)", "(1, 3)") is False


class TestGradeMathAnswerSafe:
    def test_correct(self):
        assert grade_math_answer_safe("42", "42", grader="sympy", timeout=2.0) is True

    def test_incorrect(self):
        assert grade_math_answer_safe("43", "42", grader="sympy", timeout=2.0) is False

    def test_invalid_grader(self):
        with pytest.raises(ValueError, match="Invalid grader"):
            grade_math_answer_safe("42", "42", grader="invalid")


class TestExtractGsm8kAnswer:
    def test_standard_format(self):
        assert extract_gsm8k_answer("blah blah\n#### 42") == "42"

    def test_with_commas(self):
        assert extract_gsm8k_answer("#### 1,234") == "1234"

    def test_no_answer(self):
        with pytest.raises(ValueError, match="No GSM8K final answer found"):
            extract_gsm8k_answer("no answer here")

    def test_colon_after_hashes(self):
        assert extract_gsm8k_answer("####: 7") == "7"
