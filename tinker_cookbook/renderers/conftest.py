"""Shared test utilities for renderer tests."""

from __future__ import annotations

from typing import Any


def extract_token_ids(result: Any) -> list[int]:
    """Extract token IDs from apply_chat_template result.

    transformers 4.x returns list[int], while 5.x returns BatchEncoding (dict-like
    with 'input_ids' and 'attention_mask' keys). This helper normalizes both to list[int].
    """
    if hasattr(result, "input_ids"):
        return list(result["input_ids"])
    return list(result)
