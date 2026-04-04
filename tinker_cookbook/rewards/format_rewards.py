"""Format compliance checks.

Pure functions for checking whether model outputs conform to expected
structural formats (boxed answers, XML tags, JSON validity, code blocks,
``Answer:`` prefix, etc.).  These can be used as building blocks in
reward functions.
"""

from __future__ import annotations

import json
import re


def check_has_boxed(text: str) -> bool:
    r"""Return ``True`` if *text* contains at least one ``\boxed{...}`` expression."""
    from tinker_cookbook.rewards.math_rewards import extract_boxed_answer

    try:
        extract_boxed_answer(text)
        return True
    except ValueError:
        return False


def check_has_code_block(text: str) -> bool:
    """Return ``True`` if *text* contains a fenced code block (triple backticks)."""
    return bool(re.search(r"```(?:\w+)?\n.*?```", text, re.DOTALL))


def check_has_xml_tag(text: str, tag: str) -> bool:
    """Return ``True`` if *text* contains a matching ``<tag>...</tag>`` pair.

    Args:
        text: The text to search.
        tag: The XML tag name (without angle brackets).
    """
    pattern = rf"<{re.escape(tag)}>.*?</{re.escape(tag)}>"
    return bool(re.search(pattern, text, re.DOTALL))


def extract_xml_content(text: str, tag: str) -> str | None:
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


def check_is_valid_json(text: str) -> bool:
    """Return ``True`` if *text* is valid JSON."""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def check_has_answer_prefix(text: str, prefix: str = "Answer:") -> bool:
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


# ======================================================================
# Deprecated aliases (backward compatibility)
# ======================================================================

has_boxed_answer = check_has_boxed
has_code_block = check_has_code_block
has_xml_tag = check_has_xml_tag
has_answer_prefix = check_has_answer_prefix
is_valid_json = check_is_valid_json
extract_xml_tag = extract_xml_content
