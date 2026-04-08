"""Unit tests for math grading, answer extraction, and environment logic.

These tests exercise pure functions with no network or API dependencies.
The MathDatasetBuilder integration test (which downloads from HuggingFace)
lives in tests/integration/test_math_dataset_builder.py.
"""

import pytest

from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed,
    grade_answer,
    normalize_answer,
    split_tuple,
)


class TestExtractBoxed:
    def test_simple(self):
        assert extract_boxed("The answer is \\boxed{42}") == "42"

    def test_nested_braces(self):
        assert extract_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_multiple_boxed_returns_last(self):
        assert extract_boxed("\\boxed{1} then \\boxed{2}") == "2"

    def test_boxed_without_braces(self):
        assert extract_boxed("\\boxed 7") == "7"

    def test_no_boxed_raises(self):
        with pytest.raises(ValueError, match="No boxed"):
            extract_boxed("no answer here")

    def test_empty_boxed(self):
        assert extract_boxed("\\boxed{}") == ""

    def test_boxed_with_latex(self):
        assert extract_boxed("\\boxed{x^2 + 1}") == "x^2 + 1"

    def test_deeply_nested(self):
        assert extract_boxed("\\boxed{\\sqrt{\\frac{3}{4}}}") == "\\sqrt{\\frac{3}{4}}"


class TestExtractGsm8kFinalAnswer:
    def test_standard_format(self):
        text = "Some reasoning.\n#### 42"
        assert extract_gsm8k_final_answer(text) == "42"

    def test_with_commas_stripped(self):
        text = "Calculation.\n#### 1,234"
        assert extract_gsm8k_final_answer(text) == "1234"

    def test_with_colon(self):
        text = "####: 55"
        assert extract_gsm8k_final_answer(text) == "55"

    def test_multiple_lines_returns_last(self):
        text = "#### 10\nMore reasoning.\n#### 20"
        assert extract_gsm8k_final_answer(text) == "20"

    def test_no_answer_raises(self):
        with pytest.raises(ValueError, match="No GSM8K final answer"):
            extract_gsm8k_final_answer("no hash marks here")

    def test_inline_format(self):
        text = "The total is #### 99 dollars."
        assert extract_gsm8k_final_answer(text) == "99 dollars."


class TestNormalizeAnswer:
    def test_strips_whitespace(self):
        assert normalize_answer("  42  ") == "42"

    def test_removes_text_wrapper(self):
        assert normalize_answer("\\text{hello}") == "hello"

    def test_none_returns_none(self):
        assert normalize_answer(None) is None

    def test_fraction_normalization(self):
        result = normalize_answer("\\frac12")
        assert result is not None
        assert "frac" in result
        assert "{1}" in result
        assert "{2}" in result

    def test_removes_dollars(self):
        result = normalize_answer("\\$100")
        assert result is not None
        assert "\\$" not in result

    def test_removes_percent(self):
        result = normalize_answer("50\\%")
        assert result is not None
        assert "%" not in result


class TestGradeAnswer:
    def test_exact_match(self):
        assert grade_answer("42", "42") is True

    def test_none_given(self):
        assert grade_answer(None, "42") is False  # type: ignore[arg-type]

    def test_wrong_answer(self):
        assert grade_answer("43", "42") is False

    def test_latex_fraction_vs_decimal(self):
        # The grader normalizes \frac{1}{2} and 0.5 as equivalent (special case)
        assert grade_answer("\\frac{1}{2}", "0.5") is True
        # \frac{3}{4} vs 0.75 is caught by the sympy fallback
        assert grade_answer("\\frac{3}{4}", "0.75") is True

    def test_integer_mismatch_strict(self):
        # If ground truth is an integer, given answer must also be an integer
        assert grade_answer("5.0", "5") is True
        assert grade_answer("5.1", "5") is False

    def test_sympy_equivalence(self):
        # Sympy-based equivalence only applies when both sides are non-integer
        assert grade_answer("\\frac{2}{4}", "\\frac{1}{2}") is False

    def test_tuple_matching(self):
        assert grade_answer("(1, 2)", "(1, 2)") is True
        assert grade_answer("(1, 2)", "(2, 1)") is False

    def test_with_text_wrapper(self):
        assert grade_answer("\\text{yes}", "yes") is True


class TestSplitTuple:
    def test_single_value(self):
        assert split_tuple("42") == ["42"]

    def test_parenthesized_tuple(self):
        assert split_tuple("(1, 2, 3)") == ["1", "2", "3"]

    def test_bracketed(self):
        assert split_tuple("[1, 2]") == ["1", "2"]

    def test_empty(self):
        assert split_tuple("") == []

    def test_comma_in_number(self):
        assert split_tuple("1,000") == ["1000"]
