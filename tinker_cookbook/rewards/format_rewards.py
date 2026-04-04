"""Format compliance rewards.

Pure functions for checking whether model outputs conform to expected
structural formats (boxed answers, XML tags, JSON validity, code blocks,
``Answer:`` prefix, etc.).  These can be used as building blocks in
composite reward functions.

Includes telemetry via ``tinker_cookbook.utils.trace`` (sync spans)
and ``tinker_cookbook.utils.logtree`` (HTML reports).
"""

from __future__ import annotations

import json
import re
import time

from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.trace import scope_span_sync


def has_boxed_answer(text: str) -> bool:
    r"""Return ``True`` if *text* contains at least one ``\boxed{...}`` expression."""
    from tinker_cookbook.rewards.math_rewards import extract_boxed

    try:
        extract_boxed(text)
        return True
    except ValueError:
        return False


def has_code_block(text: str) -> bool:
    """Return ``True`` if *text* contains a fenced code block (triple backticks)."""
    return bool(re.search(r"```(?:\w+)?\n.*?```", text, re.DOTALL))


def has_xml_tag(text: str, tag: str) -> bool:
    """Return ``True`` if *text* contains a matching ``<tag>...</tag>`` pair.

    Args:
        text: The text to search.
        tag: The XML tag name (without angle brackets).
    """
    pattern = rf"<{re.escape(tag)}>.*?</{re.escape(tag)}>"
    return bool(re.search(pattern, text, re.DOTALL))


def extract_xml_tag(text: str, tag: str) -> str | None:
    """Extract content between ``<tag>`` and ``</tag>``.

    Returns ``None`` if the tag pair is not found.  If multiple matches
    exist, returns the *last* one (consistent with ``extract_boxed``
    semantics).
    """
    pattern = rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


def is_valid_json(text: str) -> bool:
    """Return ``True`` if *text* is valid JSON."""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def has_answer_prefix(text: str, prefix: str = "Answer:") -> bool:
    """Return ``True`` if *text* contains *prefix* (e.g. ``Answer:``)."""
    return prefix in text


def extract_after_prefix(text: str, prefix: str = "Answer:") -> str | None:
    """Extract the text after *prefix*.

    Returns ``None`` if *prefix* is not found or appears more than once.
    """
    if prefix not in text:
        return None
    parts = text.split(prefix)
    if len(parts) != 2:
        return None
    return parts[1].strip()


def format_reward(
    text: str,
    check_fn: str = "boxed",
    format_coef: float = 0.1,
) -> float:
    """Compute a format-compliance reward term.

    Returns ``0.0`` when the format check passes (no penalty) and
    ``-format_coef`` when it fails.

    Args:
        text: The model response text.
        check_fn: One of ``"boxed"``, ``"code_block"``, ``"json"``.
        format_coef: Coefficient for the format penalty.
    """
    checkers = {
        "boxed": has_boxed_answer,
        "code_block": has_code_block,
        "json": is_valid_json,
    }
    if check_fn not in checkers:
        raise ValueError(f"Unknown check_fn: {check_fn!r}. Choose from {list(checkers)}")
    passed = checkers[check_fn](text)
    return format_coef * (float(passed) - 1.0)


# ======================================================================
# Telemetry-instrumented format reward
# ======================================================================


def format_reward_with_trace(
    text: str,
    check_fn: str = "boxed",
    format_coef: float = 0.1,
    *,
    reward_name: str = "format",
    log_to_logtree: bool = True,
) -> tuple[float, dict[str, float]]:
    """Compute a format-compliance reward with tracing and logtree logging.

    Wraps :func:`format_reward` with telemetry:

    - A ``scope_span_sync`` trace span named ``"compute_{reward_name}_reward"``
    - Logtree table with format check details
    - Metrics dict with computation time

    Args:
        text: The model response text.
        check_fn: One of ``"boxed"``, ``"code_block"``, ``"json"``.
        format_coef: Coefficient for the format penalty.
        reward_name: Name for the reward (used in span names and metric keys).
        log_to_logtree: Whether to emit logtree output.

    Returns:
        Tuple of ``(reward_value, metrics_dict)``.
    """
    t_start = time.perf_counter()

    with scope_span_sync(f"compute_{reward_name}_reward"):
        reward = format_reward(text, check_fn=check_fn, format_coef=format_coef)

    elapsed = time.perf_counter() - t_start
    passed = reward >= 0.0

    metrics = {
        f"reward/{reward_name}/computation_time": elapsed,
    }

    if log_to_logtree:
        with logtree.scope_header("Reward Computation"):
            logtree.table_from_dict({
                "reward_type": f"format_{check_fn}",
                "format_passed": passed,
                "reward": reward,
                "computation_time": f"{elapsed:.4f}s",
            })

    return reward, metrics
