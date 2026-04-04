"""Shared reward function library for Tinker training recipes.

This package collects reusable reward functions so that recipes and
custom training loops do not have to reimplement common grading logic.

Submodules
----------
- :mod:`math_rewards` -- math answer verification (normalization, sympy, math_verify)
- :mod:`code_rewards` -- code execution rewards via sandbox
- :mod:`format_rewards` -- format compliance checks (boxed, XML, JSON, code blocks)
- :mod:`llm_judge` -- LLM-as-judge rubric-based scoring

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

# Format rewards
from tinker_cookbook.rewards.format_rewards import (
    check_has_answer_prefix,
    check_has_boxed,
    check_has_code_block,
    check_has_xml_tag,
    check_is_valid_json,
    extract_after_prefix,
    extract_xml_content,
)

# LLM judge
from tinker_cookbook.rewards.llm_judge import (
    Rubric,
    compute_llm_judge_metrics,
    score_with_rubric,
    score_with_rubric_traced,
    score_with_rubrics,
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
from tinker_cookbook.rewards.format_rewards import (
    check_has_answer_prefix as has_answer_prefix,
    check_has_boxed as has_boxed_answer,
    check_has_code_block as has_code_block,
    check_has_xml_tag as has_xml_tag,
    check_is_valid_json as is_valid_json,
    extract_xml_content as extract_xml_tag,
)
from tinker_cookbook.rewards.llm_judge import (
    score_with_rubric as grade_with_rubric,
    score_with_rubric_traced as grade_with_rubric_traced,
    score_with_rubrics as grade_with_rubrics,
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
    # format (new names)
    "check_has_answer_prefix",
    "check_has_boxed",
    "check_has_code_block",
    "check_has_xml_tag",
    "check_is_valid_json",
    "extract_after_prefix",
    "extract_xml_content",
    # llm judge (new names)
    "Rubric",
    "score_with_rubric",
    "score_with_rubric_traced",
    "score_with_rubrics",
    "compute_llm_judge_metrics",
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
    "has_answer_prefix",
    "has_boxed_answer",
    "has_code_block",
    "has_xml_tag",
    "is_valid_json",
    "extract_xml_tag",
    "grade_with_rubric",
    "grade_with_rubric_traced",
    "grade_with_rubrics",
]
