"""Shared reward function library for Tinker training recipes.

This package collects reusable reward functions so that recipes and
custom training loops do not have to reimplement common grading logic.

Submodules
----------
- :mod:`math_rewards` -- math answer verification (normalization, sympy, math_verify)
- :mod:`code_rewards` -- code execution rewards via sandbox
- :mod:`format_rewards` -- format compliance checks (boxed, XML, JSON, code blocks)
- :mod:`llm_judge` -- LLM-as-judge rubric-based scoring
- :mod:`composite` -- combinators for building multi-signal rewards

All reward functions have ``*_with_trace`` variants that emit telemetry
(Perfetto trace spans, logtree HTML reports, and metric dicts) and
``compute_*_metrics`` helpers for batch-level aggregation.
"""

# Shared metrics
from tinker_cookbook.rewards._metrics import compute_reward_metrics

# Math rewards
from tinker_cookbook.rewards.math_rewards import (
    compute_math_reward_metrics,
    extract_answer_flexible,
    extract_boxed,
    extract_gsm8k_final_answer,
    grade_answer,
    grade_answer_math_verify,
    grade_answer_with_trace,
    normalize_answer,
    safe_grade,
)

# Code rewards
from tinker_cookbook.rewards.code_rewards import (
    compute_code_reward_metrics,
    extract_code_from_model,
    grade_code_response,
    sandbox_check_correctness,
    sandbox_check_correctness_with_trace,
)

# Format rewards
from tinker_cookbook.rewards.format_rewards import (
    extract_after_prefix,
    extract_xml_tag,
    format_reward,
    format_reward_with_trace,
    has_answer_prefix,
    has_boxed_answer,
    has_code_block,
    has_xml_tag,
    is_valid_json,
)

# LLM judge
from tinker_cookbook.rewards.llm_judge import (
    Rubric,
    compute_llm_judge_metrics,
    grade_with_rubric,
    grade_with_rubric_traced,
    grade_with_rubrics,
)

# Composite rewards
from tinker_cookbook.rewards.composite import (
    WeightedReward,
    reward_max,
    reward_min,
    reward_product,
    threshold,
    weighted_sum,
    weighted_sum_with_trace,
)

__all__ = [
    # shared metrics
    "compute_reward_metrics",
    # math
    "extract_answer_flexible",
    "extract_boxed",
    "extract_gsm8k_final_answer",
    "grade_answer",
    "grade_answer_math_verify",
    "grade_answer_with_trace",
    "normalize_answer",
    "safe_grade",
    "compute_math_reward_metrics",
    # code
    "extract_code_from_model",
    "grade_code_response",
    "sandbox_check_correctness",
    "sandbox_check_correctness_with_trace",
    "compute_code_reward_metrics",
    # format
    "extract_after_prefix",
    "extract_xml_tag",
    "format_reward",
    "format_reward_with_trace",
    "has_answer_prefix",
    "has_boxed_answer",
    "has_code_block",
    "has_xml_tag",
    "is_valid_json",
    # llm judge
    "Rubric",
    "grade_with_rubric",
    "grade_with_rubric_traced",
    "grade_with_rubrics",
    "compute_llm_judge_metrics",
    # composite
    "WeightedReward",
    "reward_max",
    "reward_min",
    "reward_product",
    "threshold",
    "weighted_sum",
    "weighted_sum_with_trace",
]
