"""Math answer verification rewards.

Provides functions for checking mathematical answers via normalization,
symbolic comparison (sympy), and the math_verify package. These are
extracted from ``tinker_cookbook.recipes.math_rl.math_grading`` so that
any recipe or custom training loop can reuse them without depending on
the math_rl recipe.
"""

from __future__ import annotations

import contextlib
import logging
import math
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

from tinker_cookbook.exceptions import ConfigurationError, DataFormatError
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Lazy optional imports -- only needed when the functions are actually called
# ---------------------------------------------------------------------------

def _require_sympy():
    """Import sympy and friends, raising a helpful error on failure."""
    try:
        import sympy  # noqa: F811
        from pylatexenc import latex2text  # noqa: F811
        from sympy.parsing import sympy_parser  # noqa: F811
        return sympy, latex2text, sympy_parser
    except ImportError:
        raise ImportError(
            "Math reward dependencies (sympy, pylatexenc) are required. "
            "Install them with: uv pip install sympy pylatexenc"
        ) from None


# ======================================================================
# Timeout helper
# ======================================================================


def run_with_timeout(
    func: Callable[..., T],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    timeout_seconds: int = 5,
) -> T | None:
    """Run *func* in a thread with a timeout.

    Returns ``None`` if the function times out or raises an exception.
    """
    if kwargs is None:
        kwargs = {}
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            logger.warning("Function timed out after %d seconds.", timeout_seconds)
            return None
        except Exception as e:
            logger.warning("Function raised an exception: %s", e)
            return None


# ======================================================================
# Normalization (from hendrycks MATH / minerva conventions)
# ======================================================================


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a_str = string.split("/")[0]
    b_str = string.split("/")[1]
    try:
        a = int(a_str)
        b = int(b_str)
        assert string == f"{a}/{b}"
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except ValueError:
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def normalize_answer(answer: str | None) -> str | None:
    """Normalize a math answer string using MATH-dataset conventions.

    Handles LaTeX fracs, sqrt, units, percentages, and common formatting
    variations so that semantically identical answers compare equal.
    """
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search("^\\\\text\\{(?P<text>.+?)\\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(str(answer))
    except Exception:
        return answer


# ======================================================================
# Extract boxed answer
# ======================================================================


def extract_boxed(text: str) -> str:
    r"""Extract the content of the last ``\boxed{...}`` or ``\fbox{...}`` in *text*.

    Handles both ``\boxed`` and ``\fbox`` commands (the latter is used
    by some MATH dataset variants and SLIME-style reward functions).

    Raises ``ValueError`` if no boxed expression is found.
    """
    boxed_strs: list[str] = []
    stack: list[int] = []
    for ichar in range(len(text)):
        if text[ichar] == "{":
            stack.append(ichar)
        elif text[ichar] == "}":
            if len(stack) == 0:
                raise DataFormatError("Unmatched }")
            last_open_start = stack.pop()
            prefix = text[:last_open_start]
            if prefix.endswith("\\boxed") or prefix.endswith("\\fbox"):
                boxed_strs.append(text[last_open_start + 1 : ichar])
    if len(boxed_strs) > 0:
        return boxed_strs[-1]
    match = re.search(r"\\boxed\s+([a-zA-Z0-9]+)", text)
    if match:
        return match.group(1)
    raise DataFormatError("No boxed strings found")


# ======================================================================
# Sympy-based grading
# ======================================================================

BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    sympy_mod, _, sympy_parser = _require_sympy()
    py_expr = expr.replace("^", "**")
    # Use a restricted namespace to prevent arbitrary code execution via sympy's parser,
    # matching the approach used in SLIME/DeepScaler.
    safe_dict = {k: v for k, v in sympy_mod.__dict__.items() if not k.startswith("_")}
    return sympy_parser.parse_expr(
        py_expr,
        local_dict=safe_dict,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    _, latex2text_mod, _ = _require_sympy()
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = latex2text_mod.LatexNodes2Text().latex_to_text(expr)
    expr = expr.replace("\u221a", "sqrt")
    expr = expr.replace("\u03c0", "pi")
    expr = expr.replace("\u221e", "inf")
    expr = expr.replace("\u222a", "U")
    expr = expr.replace("\u00b7", "*")
    expr = expr.replace("\u00d7", "*")
    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except (ValueError, OverflowError):
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except (ValueError, OverflowError):
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x_str: str) -> bool:
    try:
        x_str = _strip_properly_formatted_commas(x_str)
        x = float(x_str)
        return abs(x - int(round(x))) <= 1e-7
    except (ValueError, OverflowError):
        return False


def _str_to_int(x_str: str) -> int:
    x_str = x_str.replace(",", "")
    x = float(x_str)
    return int(x)


def _inject_implicit_mixed_number(step: str) -> str:
    p1 = re.compile("([0-9]) +([0-9])")
    return p1.sub("\\1+\\2", step)


def _strip_properly_formatted_commas(expr: str) -> str:
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str | None:
    if expr is None:
        return None
    m = re.search("^\\\\text\\{(?P<text>.+?)\\}$", expr)
    if m is not None:
        expr = m.group("text")
    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")
    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")
    for unit in [
        "degree", "cm", "centimeter", "meter", "mile", "second", "minute",
        "hour", "day", "week", "month", "year", "foot", "feet", "inch", "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub("\\^ *\\\\circ", "", expr)
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]
    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        with contextlib.suppress(Exception):
            expr = _parse_latex(expr)
    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")
    expr = expr.lower()
    if _str_is_int(expr):
        expr = str(_str_to_int(expr))
    return expr


def _count_unknown_letters_in_expr(expr: str) -> int:
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    return len({x for x in expr if x.isalpha()})


def _should_allow_eval(expr: str) -> bool:
    if _count_unknown_letters_in_expr(expr) > 2:
        return False
    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False
    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False
    return True


def _are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str) -> bool:
    sympy_mod, _, _ = _require_sympy()
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if _should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy_mod.simplify(sympy_diff)
            if simplified == 0:
                return True
    except Exception:
        pass
    return False


def _split_tuple(expr: str) -> list[str]:
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all(ch not in expr[1:-1] for ch in TUPLE_CHARS)
    ):
        return [elem.strip() for elem in expr[1:-1].split(",")]
    return [expr]


# ======================================================================
# Public grading API
# ======================================================================


def grade_answer(given_answer: str | None, ground_truth: str) -> bool:
    """Check if *given_answer* matches *ground_truth* using normalization and sympy.

    The answer is considered correct if either:
    (a) it normalizes to the same string as the ground truth, or
    (b) sympy can simplify the difference between the expressions to 0.
    """
    if given_answer is None:
        return False

    ground_truth_normalized_mathd = normalize_answer(ground_truth)
    given_answer_normalized_mathd = normalize_answer(given_answer)
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)
    if ground_truth_normalized is None:
        return False
    if ground_truth_normalized == given_normalized:
        return True
    if len(given_normalized) == 0:
        return False

    ground_truth_elems = _split_tuple(ground_truth_normalized)
    given_elems = _split_tuple(given_normalized)

    is_correct = False
    if (
        len(ground_truth_elems) > 1
        and (
            ground_truth_normalized[0] != given_normalized[0]
            or ground_truth_normalized[-1] != given_normalized[-1]
        )
    ) or len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for gt_elem, given_elem in zip(ground_truth_elems, given_elems, strict=True):
            if _is_frac(gt_elem) and _is_frac(given_elem):
                is_correct = gt_elem == given_elem
            elif _str_is_int(gt_elem) != _str_is_int(given_elem):
                is_correct = False
            else:
                is_correct = _are_equal_under_sympy(gt_elem, given_elem)
            if not is_correct:
                break
    return is_correct


def grade_answer_math_verify(given_answer: str, ground_truth: str) -> bool:
    """Check correctness using the ``math_verify`` package.

    Requires ``math-verify`` to be installed.

    .. note::

        This function may raise on malformed inputs. For fault-tolerant
        grading, use :func:`safe_grade` which wraps this with a timeout
        and exception handling.
    """
    from math_verify import parse, verify

    if not given_answer.startswith("$") and not given_answer.endswith("$"):
        given_answer = f"${given_answer}$"
    if not ground_truth.startswith("$") and not ground_truth.endswith("$"):
        ground_truth = f"${ground_truth}$"

    return verify(parse(given_answer), parse(ground_truth))


def safe_grade(
    given_answer: str,
    ground_truth: str,
    grader: str = "sympy",
    timeout: float = 1.0,
) -> bool:
    """Grade with a timeout -- returns ``False`` on timeout.

    Args:
        given_answer: The model's extracted answer string.
        ground_truth: The reference answer string.
        grader: ``"sympy"`` or ``"math_verify"``.
        timeout: Maximum seconds to spend grading.
    """
    if grader == "sympy":
        grader_func = grade_answer
    elif grader == "math_verify":
        grader_func = grade_answer_math_verify
    else:
        raise ConfigurationError(f"Invalid grader: {grader}")
    out = run_with_timeout(
        grader_func, args=(given_answer, ground_truth), timeout_seconds=int(math.ceil(timeout))
    )
    if out is None:
        logger.warning("Timeout grading %r against %r", given_answer, ground_truth)
        return False
    return out


def extract_gsm8k_final_answer(text: str) -> str:
    """Extract the final numeric answer from a GSM8K solution field.

    GSM8K places the final answer on a line starting with ``####``.
    """
    lines = text.splitlines()
    for line in reversed(lines):
        s = line.strip()
        if s.startswith("####"):
            content = s[4:].strip()
            if content.startswith(":"):
                content = content[1:].strip()
            content = content.replace(",", "").strip()
            return content
    matches = re.findall(r"####\s*(.+)", text)
    if matches:
        return matches[-1].strip()
    raise DataFormatError("No GSM8K final answer found")


def extract_answer_flexible(response: str) -> str | None:
    r"""Extract a math answer from a model response, trying multiple strategies.

    Tries, in order:
    1. ``\boxed{...}`` / ``\fbox{...}``
    2. ``Answer: ...`` pattern
    3. ``#### ...`` (GSM8K style)

    Returns ``None`` if no answer can be extracted.
    """
    # 1. Try boxed
    try:
        return extract_boxed(response)
    except ValueError:
        pass

    # 2. Try "Answer:" prefix
    match = re.search(r"(?i)Answer\s*:\s*([^\n]+)", response)
    if match:
        return match.group(1).strip()

    # 3. Try GSM8K style
    try:
        return extract_gsm8k_final_answer(response)
    except ValueError:
        pass

    return None


# ======================================================================
# Telemetry-instrumented grading
# ======================================================================


def grade_answer_with_trace(
    given_answer: str,
    ground_truth: str,
    *,
    grader: str = "sympy",
    timeout: float = 1.0,
    reward_name: str = "math",
    log_to_logtree: bool = True,
) -> tuple[float, dict[str, Any]]:
    """Grade a math answer with tracing, logtree logging, and metric computation.

    This wraps :func:`safe_grade` with telemetry instrumentation:

    - A ``scope_span_sync`` trace span named ``"compute_{reward_name}_reward"``
    - Logtree table with grading details (when a logtree trace is active)
    - A metrics dict with computation time

    Args:
        given_answer: The model's extracted answer string.
        ground_truth: The reference answer string.
        grader: ``"sympy"`` or ``"math_verify"``.
        timeout: Maximum seconds to spend grading.
        reward_name: Name for the reward (used in span names and metric keys).
        log_to_logtree: Whether to emit logtree output.

    Returns:
        Tuple of ``(reward_value, metrics_dict)`` where ``reward_value`` is
        1.0 (correct) or 0.0 (incorrect), and ``metrics_dict`` contains
        ``reward/{name}/computation_time``.
    """
    import time as _time

    from tinker_cookbook.utils import logtree as _logtree
    from tinker_cookbook.utils.trace import scope_span_sync as _scope_span_sync

    span_name = f"compute_{reward_name}_reward"
    t_start = _time.perf_counter()

    with _scope_span_sync(span_name):
        is_correct = safe_grade(
            given_answer, ground_truth, grader=grader, timeout=timeout
        )

    elapsed = _time.perf_counter() - t_start
    reward = 1.0 if is_correct else 0.0

    metrics = {
        f"reward/{reward_name}/computation_time": elapsed,
    }

    if log_to_logtree:
        with _logtree.scope_header("Reward Computation"):
            _logtree.table_from_dict({
                "reward_type": f"math_{grader}",
                "expected": ground_truth,
                "extracted": given_answer,
                "reward": reward,
                "computation_time": f"{elapsed:.4f}s",
            })

    return reward, metrics


def compute_math_reward_metrics(
    rewards: Sequence[float],
    *,
    reward_name: str = "math",
) -> dict[str, float]:
    """Compute aggregate metrics for a batch of math reward values.

    Thin wrapper around :func:`~tinker_cookbook.rewards._metrics.compute_reward_metrics`
    with ``reward_name`` defaulting to ``"math"``.
    """
    from tinker_cookbook.rewards._metrics import compute_reward_metrics

    return compute_reward_metrics(rewards, reward_name)
