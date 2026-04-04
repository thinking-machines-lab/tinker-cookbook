"""Math grading utilities for RL training.

This module re-exports from ``tinker_cookbook.rewards.math_rewards`` for
backward compatibility. New code should import directly from
``tinker_cookbook.rewards``.
"""

# Re-export everything from the shared rewards library (new names)
from tinker_cookbook.rewards.math_rewards import (
    extract_boxed_answer,
    extract_gsm8k_answer,
    grade_math_answer,
    grade_math_answer_safe,
    grade_math_answer_strict,
    normalize_answer,
    run_with_timeout as run_with_timeout_signal,
)

# Deprecated aliases (backward compatibility)
extract_boxed = extract_boxed_answer
extract_gsm8k_final_answer = extract_gsm8k_answer
grade_answer = grade_math_answer
grade_answer_math_verify = grade_math_answer_strict
safe_grade = grade_math_answer_safe

__all__ = [
    "extract_boxed_answer",
    "extract_gsm8k_answer",
    "grade_math_answer",
    "grade_math_answer_safe",
    "grade_math_answer_strict",
    "normalize_answer",
    "run_with_timeout_signal",
    # Deprecated aliases
    "extract_boxed",
    "extract_gsm8k_final_answer",
    "grade_answer",
    "grade_answer_math_verify",
    "safe_grade",
]
