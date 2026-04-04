"""
Grading utilities for math RFT recipe.

Extracts boxed answers from model responses and checks correctness
using symbolic math grading.
"""

from tinker_cookbook.recipes.math_rl.math_env import safe_grade
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed


def grade_response(response_text: str, ground_truth: str) -> bool:
    """Check if a response contains the correct answer in \\boxed{} format."""
    try:
        given_answer = extract_boxed(response_text)
    except ValueError:
        return False
    return safe_grade(given_answer, ground_truth, grader="sympy", timeout=1.0)
