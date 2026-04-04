"""Unit tests for reward functions with telemetry.

These tests verify both the core reward logic and the telemetry-instrumented
variants (trace spans, logtree logging, metrics computation).
"""

from __future__ import annotations

import asyncio

import pytest


# ======================================================================
# Math rewards
# ======================================================================


class TestMathRewards:
    def test_normalize_answer_basic(self):
        from tinker_cookbook.rewards.math_rewards import normalize_answer

        assert normalize_answer("  42  ") == "42"
        assert normalize_answer("\\frac{1}{2}") == "\\frac{1}{2}"
        assert normalize_answer("0.5") == "\\frac{1}{2}"

    def test_extract_boxed_answer_basic(self):
        from tinker_cookbook.rewards.math_rewards import extract_boxed_answer

        assert extract_boxed_answer(r"The answer is \boxed{42}") == "42"
        assert extract_boxed_answer(r"\boxed{x^2 + 1}") == "x^2 + 1"

    def test_extract_boxed_answer_fbox(self):
        """Verify that \\fbox{...} is also extracted (SLIME compatibility)."""
        from tinker_cookbook.rewards.math_rewards import extract_boxed_answer

        assert extract_boxed_answer(r"The answer is \fbox{42}") == "42"

    def test_extract_boxed_answer_last(self):
        from tinker_cookbook.rewards.math_rewards import extract_boxed_answer

        text = r"\boxed{1} and \boxed{2}"
        assert extract_boxed_answer(text) == "2"

    def test_extract_boxed_answer_no_match(self):
        from tinker_cookbook.rewards.math_rewards import extract_boxed_answer

        with pytest.raises(ValueError, match="No boxed"):
            extract_boxed_answer("no boxed here")

    def test_grade_math_answer_identical(self):
        from tinker_cookbook.rewards.math_rewards import grade_math_answer

        assert grade_math_answer("42", "42") is True

    def test_grade_math_answer_equivalent_fraction(self):
        from tinker_cookbook.rewards.math_rewards import grade_math_answer

        assert grade_math_answer("0.5", "\\frac{1}{2}") is True

    def test_grade_math_answer_wrong(self):
        from tinker_cookbook.rewards.math_rewards import grade_math_answer

        assert grade_math_answer("43", "42") is False

    def test_grade_math_answer_none(self):
        from tinker_cookbook.rewards.math_rewards import grade_math_answer

        assert grade_math_answer(None, "42") is False

    def test_grade_math_answer_safe_timeout(self):
        from tinker_cookbook.rewards.math_rewards import grade_math_answer_safe

        # Should not hang; returns True or False within timeout
        result = grade_math_answer_safe("42", "42", grader="sympy", timeout=2.0)
        assert result is True

    def test_extract_gsm8k_answer(self):
        from tinker_cookbook.rewards.math_rewards import extract_gsm8k_answer

        text = "Step 1: 10\nStep 2: 20\n#### 30"
        assert extract_gsm8k_answer(text) == "30"

    def test_extract_gsm8k_answer_with_comma(self):
        from tinker_cookbook.rewards.math_rewards import extract_gsm8k_answer

        text = "#### 1,234"
        assert extract_gsm8k_answer(text) == "1234"

    def test_extract_math_answer_boxed(self):
        from tinker_cookbook.rewards.math_rewards import extract_math_answer

        assert extract_math_answer(r"Thus \boxed{42}") == "42"

    def test_extract_math_answer_answer_prefix(self):
        from tinker_cookbook.rewards.math_rewards import extract_math_answer

        assert extract_math_answer("Answer: 42") == "42"

    def test_extract_math_answer_gsm8k(self):
        from tinker_cookbook.rewards.math_rewards import extract_math_answer

        assert extract_math_answer("Steps...\n#### 7") == "7"

    def test_extract_math_answer_none(self):
        from tinker_cookbook.rewards.math_rewards import extract_math_answer

        assert extract_math_answer("no answer here at all") is None

    def test_grade_math_answer_none_type(self):
        """grade_math_answer accepts None for given_answer (FIX 8)."""
        from tinker_cookbook.rewards.math_rewards import grade_math_answer

        assert grade_math_answer(None, "42") is False


class TestMathRewardTelemetry:
    def test_grade_math_answer_traced(self):
        from tinker_cookbook.rewards.math_rewards import grade_math_answer_traced

        reward, metrics = grade_math_answer_traced(
            "42", "42", log_to_logtree=False,
        )
        assert reward == 1.0
        assert "reward/math/computation_time" in metrics
        assert metrics["reward/math/computation_time"] >= 0

    def test_grade_math_answer_traced_wrong(self):
        from tinker_cookbook.rewards.math_rewards import grade_math_answer_traced

        reward, metrics = grade_math_answer_traced(
            "43", "42", log_to_logtree=False,
        )
        assert reward == 0.0

    def test_compute_math_reward_metrics(self):
        from tinker_cookbook.rewards.math_rewards import compute_math_reward_metrics

        metrics = compute_math_reward_metrics([1.0, 0.0, 1.0, 1.0])
        assert metrics["reward/math/mean"] == 0.75
        assert metrics["reward/math/fraction_correct"] == 0.75
        assert metrics["reward/math/std"] > 0

    def test_compute_math_reward_metrics_empty(self):
        from tinker_cookbook.rewards.math_rewards import compute_math_reward_metrics

        assert compute_math_reward_metrics([]) == {}

    def test_compute_math_reward_metrics_all_correct(self):
        from tinker_cookbook.rewards.math_rewards import compute_math_reward_metrics

        metrics = compute_math_reward_metrics([1.0, 1.0, 1.0])
        assert metrics["reward/math/mean"] == 1.0
        assert metrics["reward/math/fraction_correct"] == 1.0
        assert metrics["reward/math/std"] == 0.0


# ======================================================================
# Format rewards
# ======================================================================


class TestFormatRewards:
    def test_check_has_boxed(self):
        from tinker_cookbook.rewards.format_rewards import check_has_boxed

        assert check_has_boxed(r"\boxed{42}") is True
        assert check_has_boxed("no boxed") is False

    def test_check_has_code_block(self):
        from tinker_cookbook.rewards.format_rewards import check_has_code_block

        assert check_has_code_block("```python\nprint('hi')\n```") is True
        assert check_has_code_block("no code") is False



# ======================================================================
# Code rewards
# ======================================================================


class TestCodeRewards:
    def test_extract_code_block(self):
        from tinker_cookbook.rewards.code_rewards import extract_code_block

        text = "Here is the code:\n```python\ndef foo(): pass\n```"
        assert extract_code_block(text) == "def foo(): pass"

    def test_extract_code_block_none(self):
        from tinker_cookbook.rewards.code_rewards import extract_code_block

        assert extract_code_block("no code here") is None

    def test_grade_code_correctness(self):
        from tinker_cookbook.rewards.code_rewards import grade_code_correctness

        code, has_block = grade_code_correctness("```python\npass\n```")
        assert has_block is True
        assert code == "pass"

    def test_compute_code_reward_metrics(self):
        from tinker_cookbook.rewards.code_rewards import compute_code_reward_metrics

        metrics = compute_code_reward_metrics([1.0, 0.0, 1.0])
        assert abs(metrics["reward/code/mean"] - 2 / 3) < 1e-6
        assert abs(metrics["reward/code/fraction_correct"] - 2 / 3) < 1e-6


# ======================================================================
# LLM judge
# ======================================================================


class TestLlmJudge:
    def test_rubric_extract_score(self):
        from tinker_cookbook.rewards.llm_judge import Rubric

        rubric = Rubric(rubric_str="test")
        assert rubric.extract_score("The score is <score>0.8</score>") == 0.8
        assert rubric.extract_score("no score here") == 0.0
        assert rubric.extract_score("<score>invalid</score>") == 0.0

    def test_compute_llm_judge_metrics(self):
        from tinker_cookbook.rewards.llm_judge import compute_llm_judge_metrics

        metrics = compute_llm_judge_metrics([0.8, 0.6, 0.3, 1.0])
        assert 0.0 < metrics["reward/llm_judge/mean"] < 1.0
        assert metrics["reward/llm_judge/std"] > 0
        assert metrics["reward/llm_judge/fraction_correct"] == 0.75

    def test_compute_llm_judge_metrics_empty(self):
        from tinker_cookbook.rewards.llm_judge import compute_llm_judge_metrics

        assert compute_llm_judge_metrics([]) == {}


# ======================================================================
# __init__ imports
# ======================================================================


class TestInitImports:
    """Verify that all public symbols are importable from the package."""

    def test_import_traced_functions(self):
        from tinker_cookbook.rewards import (
            compute_code_reward_metrics,
            compute_llm_judge_metrics,
            compute_math_reward_metrics,
            extract_math_answer,
            grade_math_answer_traced,
            score_code_sandbox_traced,
            score_with_rubric_traced,
        )

        # Just verify they're callable
        assert callable(grade_math_answer_traced)
        assert callable(score_code_sandbox_traced)
        assert callable(score_with_rubric_traced)
        assert callable(compute_math_reward_metrics)
        assert callable(compute_code_reward_metrics)
        assert callable(compute_llm_judge_metrics)
        assert callable(extract_math_answer)

    def test_import_new_names(self):
        from tinker_cookbook.rewards import (
            Rubric,
            extract_boxed_answer,
            grade_math_answer,
            grade_math_answer_safe,
            score_with_rubric,
        )

        assert callable(grade_math_answer)
        assert callable(grade_math_answer_safe)
        assert callable(extract_boxed_answer)
        assert callable(score_with_rubric)
        assert Rubric is not None

    def test_import_deprecated_aliases(self):
        """Verify backward-compatible aliases are still importable."""
        from tinker_cookbook.rewards import (
            Rubric,
            extract_boxed,
            grade_answer,
            grade_with_rubric,
            safe_grade,
        )

        assert callable(grade_answer)
        assert callable(safe_grade)
        assert callable(extract_boxed)
        assert callable(grade_with_rubric)
        assert Rubric is not None

    def test_import_compute_reward_metrics(self):
        from tinker_cookbook.rewards import compute_reward_metrics

        assert callable(compute_reward_metrics)


# ======================================================================
# Shared metrics (_metrics.py)
# ======================================================================


class TestSharedMetrics:
    def test_compute_reward_metrics(self):
        from tinker_cookbook.rewards._metrics import compute_reward_metrics

        metrics = compute_reward_metrics([1.0, 0.0, 1.0, 1.0], "test")
        assert metrics["reward/test/mean"] == 0.75
        assert metrics["reward/test/fraction_correct"] == 0.75
        assert metrics["reward/test/std"] > 0

    def test_empty(self):
        from tinker_cookbook.rewards._metrics import compute_reward_metrics

        assert compute_reward_metrics([], "test") == {}

    def test_delegates_math(self):
        """compute_math_reward_metrics delegates to shared implementation."""
        from tinker_cookbook.rewards.math_rewards import compute_math_reward_metrics

        metrics = compute_math_reward_metrics([1.0, 0.0])
        assert "reward/math/mean" in metrics

    def test_delegates_code(self):
        """compute_code_reward_metrics delegates to shared implementation."""
        from tinker_cookbook.rewards.code_rewards import compute_code_reward_metrics

        metrics = compute_code_reward_metrics([1.0, 0.0])
        assert "reward/code/mean" in metrics

    def test_delegates_llm_judge(self):
        """compute_llm_judge_metrics delegates to shared implementation."""
        from tinker_cookbook.rewards.llm_judge import compute_llm_judge_metrics

        metrics = compute_llm_judge_metrics([0.8, 0.2])
        assert "reward/llm_judge/mean" in metrics


# ======================================================================
# LLM judge error handling
# ======================================================================


class TestLlmJudgeErrorHandling:
    def test_score_with_rubric_api_failure(self):
        """score_with_rubric returns default_on_error on API failure."""
        from tinker_cookbook.rewards.llm_judge import Rubric, score_with_rubric

        rubric = Rubric(rubric_str="test")
        convo = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        async def failing_llm(_msgs):
            raise ConnectionError("API down")

        score, text = asyncio.run(
            score_with_rubric(convo, rubric, failing_llm)
        )
        assert score == 0.0
        assert "[error]" in text

    def test_score_with_rubric_custom_default(self):
        """score_with_rubric respects custom default_on_error."""
        from tinker_cookbook.rewards.llm_judge import Rubric, score_with_rubric

        rubric = Rubric(rubric_str="test")
        convo = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        async def failing_llm(_msgs):
            raise RuntimeError("timeout")

        score, _ = asyncio.run(
            score_with_rubric(convo, rubric, failing_llm, default_on_error=-1.0)
        )
        assert score == -1.0


# ======================================================================
# Rubric extraction regex (non-greedy)
# ======================================================================


class TestRubricNonGreedyRegex:
    def test_multiple_score_tags(self):
        """Non-greedy regex extracts first score, not spanning across tags."""
        from tinker_cookbook.rewards.llm_judge import Rubric

        rubric = Rubric(rubric_str="test")
        # With greedy .*, this would match "0.3</score> ... <score>0.9"
        # With non-greedy .*?, it correctly matches "0.3"
        text = "<score>0.3</score> some text <score>0.9</score>"
        assert rubric.extract_score(text) == 0.3
