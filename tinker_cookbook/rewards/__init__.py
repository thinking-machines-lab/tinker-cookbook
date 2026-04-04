"""Shared reward function library for Tinker training recipes.

This package collects reusable reward functions so that recipes and
custom training loops do not have to reimplement common grading logic.

Submodules
----------
- :mod:`math_rewards` -- math answer verification (normalization, sympy, math_verify)
- :mod:`code_rewards` -- code execution rewards via sandbox

All reward functions have ``*_traced`` variants that emit telemetry
(Perfetto trace spans, logtree HTML reports, and metric dicts) and
``compute_*_metrics`` helpers for batch-level aggregation.
"""

# Shared metrics
from tinker_cookbook.rewards._metrics import compute_reward_metrics

# Math rewards
from tinker_cookbook.rewards.math_rewards import (
    compute_math_reward_metrics,
    extract_boxed_answer,
    extract_gsm8k_answer,
    extract_math_answer,
    grade_math_answer,
    grade_math_answer_safe,
    grade_math_answer_strict,
    grade_math_answer_traced,
    normalize_answer,
)

# Code rewards
from tinker_cookbook.rewards.code_rewards import (
    compute_code_reward_metrics,
    extract_code_block,
    grade_code_correctness,
    score_code_sandbox,
    score_code_sandbox_traced,
)

# Deprecated aliases (backward compatibility)
from tinker_cookbook.rewards.math_rewards import (
    extract_boxed_answer as extract_boxed,
    extract_gsm8k_answer as extract_gsm8k_final_answer,
    extract_math_answer as extract_answer_flexible,
    grade_math_answer as grade_answer,
    grade_math_answer_safe as safe_grade,
    grade_math_answer_strict as grade_answer_math_verify,
    grade_math_answer_traced as grade_answer_with_trace,
)
from tinker_cookbook.rewards.code_rewards import (
    extract_code_block as extract_code_from_model,
    grade_code_correctness as grade_code_response,
    score_code_sandbox as sandbox_check_correctness,
    score_code_sandbox_traced as sandbox_check_correctness_with_trace,
)

__all__ = [
    # shared metrics
    "compute_reward_metrics",
    # math (new names)
    "extract_boxed_answer",
    "extract_gsm8k_answer",
    "extract_math_answer",
    "grade_math_answer",
    "grade_math_answer_safe",
    "grade_math_answer_strict",
    "grade_math_answer_traced",
    "normalize_answer",
    "compute_math_reward_metrics",
    # code (new names)
    "extract_code_block",
    "grade_code_correctness",
    "score_code_sandbox",
    "score_code_sandbox_traced",
    "compute_code_reward_metrics",
    # Deprecated aliases
    "extract_boxed",
    "extract_gsm8k_final_answer",
    "extract_answer_flexible",
    "grade_answer",
    "safe_grade",
    "grade_answer_math_verify",
    "grade_answer_with_trace",
    "extract_code_from_model",
    "grade_code_response",
    "sandbox_check_correctness",
    "sandbox_check_correctness_with_trace",
]
