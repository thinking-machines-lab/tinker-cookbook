"""Math grading utilities for RL training.

This module re-exports from ``tinker_cookbook.rewards.math_rewards`` for
backward compatibility. New code should import directly from
``tinker_cookbook.rewards``.
"""

# Re-export everything from the shared rewards library
from tinker_cookbook.rewards.math_rewards import (
    extract_boxed,
    extract_gsm8k_final_answer,
    grade_answer,
    grade_answer_math_verify,
    normalize_answer,
    run_with_timeout as run_with_timeout_signal,
    safe_grade,
)

__all__ = [
    "extract_boxed",
    "extract_gsm8k_final_answer",
    "grade_answer",
    "grade_answer_math_verify",
    "normalize_answer",
    "run_with_timeout_signal",
    "safe_grade",
]
