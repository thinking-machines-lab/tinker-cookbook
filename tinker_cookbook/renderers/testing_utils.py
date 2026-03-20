"""Shared test utilities for renderer tests."""

from __future__ import annotations

from typing import Any

import pytest
import transformers


def extract_token_ids(result: Any) -> list[int]:
    """Extract token IDs from apply_chat_template result.

    transformers 4.x returns list[int], while 5.x returns BatchEncoding (dict-like
    with 'input_ids' and 'attention_mask' keys). This helper normalizes both to list[int].
    """
    if hasattr(result, "input_ids"):
        return list(result["input_ids"])
    return list(result)


_DEEPSEEK_TOKENIZER_BUG = (
    "transformers 5.3.0 has a known bug with DeepSeek tokenizer that strips spaces during decode. "
    "See https://github.com/huggingface/transformers/pull/44801"
)

_HAS_DEEPSEEK_TOKENIZER_BUG = transformers.__version__ == "5.3.0"

skip_deepseek_tokenizer_bug = pytest.mark.skipif(
    _HAS_DEEPSEEK_TOKENIZER_BUG,
    reason=_DEEPSEEK_TOKENIZER_BUG,
)


def skip_if_deepseek_tokenizer_bug(model_name: str) -> None:
    """Skip the current test if running DeepSeek on transformers 5.3.0."""
    if _HAS_DEEPSEEK_TOKENIZER_BUG and "deepseek" in model_name.lower():
        pytest.skip(_DEEPSEEK_TOKENIZER_BUG)
