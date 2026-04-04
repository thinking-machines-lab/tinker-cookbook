"""Unit tests for reward functions with telemetry.

These tests verify both the core reward logic and the telemetry-instrumented
variants (trace spans, logtree logging, metrics computation).
"""

from __future__ import annotations

import asyncio
import math

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

    def test_extract_boxed_basic(self):
        from tinker_cookbook.rewards.math_rewards import extract_boxed

        assert extract_boxed(r"The answer is \boxed{42}") == "42"
        assert extract_boxed(r"\boxed{x^2 + 1}") == "x^2 + 1"

    def test_extract_boxed_fbox(self):
        """Verify that \\fbox{...} is also extracted (SLIME compatibility)."""
        from tinker_cookbook.rewards.math_rewards import extract_boxed

        assert extract_boxed(r"The answer is \fbox{42}") == "42"

    def test_extract_boxed_last(self):
        from tinker_cookbook.rewards.math_rewards import extract_boxed

        text = r"\boxed{1} and \boxed{2}"
        assert extract_boxed(text) == "2"

    def test_extract_boxed_no_match(self):
        from tinker_cookbook.rewards.math_rewards import extract_boxed

        with pytest.raises(ValueError, match="No boxed"):
            extract_boxed("no boxed here")

    def test_grade_answer_identical(self):
        from tinker_cookbook.rewards.math_rewards import grade_answer

        assert grade_answer("42", "42") is True

    def test_grade_answer_equivalent_fraction(self):
        from tinker_cookbook.rewards.math_rewards import grade_answer

        assert grade_answer("0.5", "\\frac{1}{2}") is True

    def test_grade_answer_wrong(self):
        from tinker_cookbook.rewards.math_rewards import grade_answer

        assert grade_answer("43", "42") is False

    def test_grade_answer_none(self):
        from tinker_cookbook.rewards.math_rewards import grade_answer

        assert grade_answer(None, "42") is False

    def test_safe_grade_timeout(self):
        from tinker_cookbook.rewards.math_rewards import safe_grade

        # Should not hang; returns True or False within timeout
        result = safe_grade("42", "42", grader="sympy", timeout=2.0)
        assert result is True

    def test_extract_gsm8k_final_answer(self):
        from tinker_cookbook.rewards.math_rewards import extract_gsm8k_final_answer

        text = "Step 1: 10\nStep 2: 20\n#### 30"
        assert extract_gsm8k_final_answer(text) == "30"

    def test_extract_gsm8k_with_comma(self):
        from tinker_cookbook.rewards.math_rewards import extract_gsm8k_final_answer

        text = "#### 1,234"
        assert extract_gsm8k_final_answer(text) == "1234"

    def test_extract_answer_flexible_boxed(self):
        from tinker_cookbook.rewards.math_rewards import extract_answer_flexible

        assert extract_answer_flexible(r"Thus \boxed{42}") == "42"

    def test_extract_answer_flexible_answer_prefix(self):
        from tinker_cookbook.rewards.math_rewards import extract_answer_flexible

        assert extract_answer_flexible("Answer: 42") == "42"

    def test_extract_answer_flexible_gsm8k(self):
        from tinker_cookbook.rewards.math_rewards import extract_answer_flexible

        assert extract_answer_flexible("Steps...\n#### 7") == "7"

    def test_extract_answer_flexible_none(self):
        from tinker_cookbook.rewards.math_rewards import extract_answer_flexible

        assert extract_answer_flexible("no answer here at all") is None

    def test_is_numerically_close(self):
        from tinker_cookbook.rewards.math_rewards import _is_numerically_close

        assert _is_numerically_close("3.14", "3.14") is True
        assert _is_numerically_close("3.14", "3.140000001") is True
        assert _is_numerically_close("3.14", "3.15") is False
        assert _is_numerically_close("abc", "3.14") is False


class TestMathRewardTelemetry:
    def test_grade_answer_with_trace(self):
        from tinker_cookbook.rewards.math_rewards import grade_answer_with_trace

        reward, metrics = grade_answer_with_trace(
            "42", "42", log_to_logtree=False,
        )
        assert reward == 1.0
        assert "reward/math/computation_time" in metrics
        assert metrics["reward/math/computation_time"] >= 0

    def test_grade_answer_with_trace_wrong(self):
        from tinker_cookbook.rewards.math_rewards import grade_answer_with_trace

        reward, metrics = grade_answer_with_trace(
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
    def test_has_boxed_answer(self):
        from tinker_cookbook.rewards.format_rewards import has_boxed_answer

        assert has_boxed_answer(r"\boxed{42}") is True
        assert has_boxed_answer("no boxed") is False

    def test_has_code_block(self):
        from tinker_cookbook.rewards.format_rewards import has_code_block

        assert has_code_block("```python\nprint('hi')\n```") is True
        assert has_code_block("no code") is False

    def test_format_reward(self):
        from tinker_cookbook.rewards.format_rewards import format_reward

        assert format_reward(r"\boxed{42}", check_fn="boxed") == 0.0
        assert format_reward("no boxed", check_fn="boxed") == -0.1

    def test_format_reward_with_trace(self):
        from tinker_cookbook.rewards.format_rewards import format_reward_with_trace

        reward, metrics = format_reward_with_trace(
            r"\boxed{42}", check_fn="boxed", log_to_logtree=False,
        )
        assert reward == 0.0
        assert "reward/format/computation_time" in metrics


# ======================================================================
# Code rewards
# ======================================================================


class TestCodeRewards:
    def test_extract_code_from_model(self):
        from tinker_cookbook.rewards.code_rewards import extract_code_from_model

        text = "Here is the code:\n```python\ndef foo(): pass\n```"
        assert extract_code_from_model(text) == "def foo(): pass"

    def test_extract_code_none(self):
        from tinker_cookbook.rewards.code_rewards import extract_code_from_model

        assert extract_code_from_model("no code here") is None

    def test_grade_code_response(self):
        from tinker_cookbook.rewards.code_rewards import grade_code_response

        code, has_block = grade_code_response("```python\npass\n```")
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
# Composite rewards
# ======================================================================


class TestCompositeRewards:
    def test_weighted_sum(self):
        from tinker_cookbook.rewards.composite import WeightedReward, weighted_sum

        rewards = [
            WeightedReward("r1", lambda t: 1.0, weight=0.5),
            WeightedReward("r2", lambda t: 0.0, weight=0.5),
        ]
        total, metrics = weighted_sum("test", rewards)
        assert total == 0.5
        assert metrics["r1"] == 1.0
        assert metrics["r2"] == 0.0

    def test_threshold(self):
        from tinker_cookbook.rewards.composite import threshold

        fn = threshold(lambda t: 0.7, cutoff=0.5)
        assert fn("test") == 1.0
        fn2 = threshold(lambda t: 0.3, cutoff=0.5)
        assert fn2("test") == 0.0

    def test_reward_min(self):
        from tinker_cookbook.rewards.composite import reward_min

        result = reward_min("test", [lambda t: 0.3, lambda t: 0.7])
        assert result == 0.3

    def test_reward_max(self):
        from tinker_cookbook.rewards.composite import reward_max

        result = reward_max("test", [lambda t: 0.3, lambda t: 0.7])
        assert result == 0.7

    def test_reward_product(self):
        from tinker_cookbook.rewards.composite import reward_product

        result = reward_product("test", [lambda t: 0.5, lambda t: 0.4])
        assert abs(result - 0.2) < 1e-6

    def test_weighted_sum_with_trace(self):
        from tinker_cookbook.rewards.composite import (
            WeightedReward,
            weighted_sum_with_trace,
        )

        rewards = [
            WeightedReward("correctness", lambda t: 1.0, weight=0.8),
            WeightedReward("format", lambda t: 0.0, weight=0.2),
        ]
        total, metrics = weighted_sum_with_trace("test", rewards, log_to_logtree=False)
        assert total == 0.8
        assert metrics["reward/composite/correctness"] == 1.0
        assert metrics["reward/composite/format"] == 0.0
        assert metrics["reward/composite/total"] == 0.8
        assert "reward/composite/computation_time" in metrics


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
            extract_answer_flexible,
            format_reward_with_trace,
            grade_answer_with_trace,
            grade_with_rubric_traced,
            sandbox_check_correctness_with_trace,
            weighted_sum_with_trace,
        )

        # Just verify they're callable
        assert callable(grade_answer_with_trace)
        assert callable(sandbox_check_correctness_with_trace)
        assert callable(grade_with_rubric_traced)
        assert callable(format_reward_with_trace)
        assert callable(weighted_sum_with_trace)
        assert callable(compute_math_reward_metrics)
        assert callable(compute_code_reward_metrics)
        assert callable(compute_llm_judge_metrics)
        assert callable(extract_answer_flexible)

    def test_import_original_functions(self):
        from tinker_cookbook.rewards import (
            Rubric,
            WeightedReward,
            extract_boxed,
            format_reward,
            grade_answer,
            grade_with_rubric,
            safe_grade,
            weighted_sum,
        )

        assert callable(grade_answer)
        assert callable(safe_grade)
        assert callable(extract_boxed)
        assert callable(format_reward)
        assert callable(grade_with_rubric)
        assert callable(weighted_sum)
        assert Rubric is not None
        assert WeightedReward is not None
