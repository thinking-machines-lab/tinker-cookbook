"""Unit tests for math RFT recipe (no API key needed)."""

import pytest

from tinker_cookbook.recipes.math_rft.datasets import (
    _parse_gsm8k_row,
    _parse_math_row,
)
from tinker_cookbook.recipes.math_rft.grading import grade_response
from tinker_cookbook.recipes.math_rft.train import Config, _get_question_suffix


class TestConfig:
    def test_defaults(self):
        config = Config()
        assert config.model_name == "Qwen/Qwen3-8B"
        assert config.env == "math"
        assert config.group_size == 16
        assert config.groups_per_batch == 32
        assert config.learning_rate == 1e-4
        assert config.max_tokens == 2048
        assert config.max_length == 3072
        assert config.behavior_if_log_dir_exists == "resume"

    def test_question_suffix_gsm8k(self):
        suffix = _get_question_suffix("gsm8k")
        assert "\\boxed{}" in suffix
        assert "numerical" in suffix

    def test_question_suffix_math(self):
        suffix = _get_question_suffix("math")
        assert "\\boxed{}" in suffix


class TestGrading:
    def test_correct_boxed_answer(self):
        assert grade_response("The answer is \\boxed{42}.", "42")

    def test_incorrect_boxed_answer(self):
        assert not grade_response("The answer is \\boxed{43}.", "42")

    def test_no_boxed_answer(self):
        assert not grade_response("The answer is 42.", "42")

    def test_symbolic_equivalence(self):
        assert grade_response("The answer is \\boxed{\\frac{1}{2}}.", "0.5")

    def test_empty_response(self):
        assert not grade_response("", "42")

    def test_nested_boxed(self):
        # Model sometimes produces nested \boxed
        assert grade_response("So \\boxed{\\boxed{7}}.", "7")

    def test_fraction_match(self):
        assert grade_response("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}")

    def test_negative_number(self):
        assert grade_response("\\boxed{-5}", "-5")
        assert not grade_response("\\boxed{5}", "-5")


class TestParseMathRow:
    def test_basic_row(self):
        row = {
            "problem": "What is 2+2?",
            "solution": "We compute $2+2 = \\boxed{4}$.",
            "level": "Level 1",
            "type": "Algebra",
        }
        result = _parse_math_row(row)
        assert result is not None
        assert result["problem"] == "What is 2+2?"
        assert result["answer"] == "4"
        assert result["level"] == "1"
        assert result["category"] == "Algebra"

    def test_level_normalization(self):
        row = {
            "problem": "p",
            "solution": "\\boxed{1}",
            "level": "Level 5",
            "type": "Geometry",
        }
        result = _parse_math_row(row)
        assert result is not None
        assert result["level"] == "5"

    def test_level_already_numeric(self):
        row = {
            "problem": "p",
            "solution": "\\boxed{1}",
            "level": "3",
            "subject": "Algebra",
        }
        result = _parse_math_row(row)
        assert result is not None
        assert result["level"] == "3"
        assert result["category"] == "Algebra"

    def test_direct_answer_field(self):
        row = {
            "problem": "p",
            "answer": "42",
            "solution": "ignored",
            "level": "1",
        }
        result = _parse_math_row(row)
        assert result is not None
        assert result["answer"] == "42"

    def test_invalid_solution(self):
        row = {"problem": "p", "solution": "No boxed answer here", "level": "1"}
        result = _parse_math_row(row)
        assert result is None


class TestParseGsm8kRow:
    def test_basic_row(self):
        row = {"question": "How many apples?", "answer": "Janet has 4 apples.\n#### 4"}
        result = _parse_gsm8k_row(row)
        assert result is not None
        assert result["problem"] == "How many apples?"
        assert result["answer"] == "4"
        assert result["level"] == ""
        assert result["category"] == ""

    def test_invalid_answer(self):
        row = {"question": "q", "answer": "No final answer here"}
        result = _parse_gsm8k_row(row)
        assert result is None


class TestLoadDatasets:
    """Integration tests that actually load from HuggingFace.

    These are slow and require network access, so they're marked accordingly.
    Run with: pytest -m "not slow" to skip.
    """

    @pytest.mark.slow
    def test_load_math_problems(self):
        from tinker_cookbook.recipes.math_rft.datasets import load_math_problems

        train, test = load_math_problems(seed=42)
        assert len(train) > 7000  # ~7500 minus parse failures
        assert len(test) > 400  # ~5000 (full test) or 500 (MATH-500)
        # Check structure
        for item in train[:5]:
            assert "problem" in item
            assert "answer" in item
            assert "level" in item

    @pytest.mark.slow
    def test_load_gsm8k_problems(self):
        from tinker_cookbook.recipes.math_rft.datasets import load_gsm8k_problems

        train, test = load_gsm8k_problems(seed=42)
        assert len(train) > 7000
        assert len(test) > 1000
