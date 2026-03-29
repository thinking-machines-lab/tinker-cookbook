"""Shared utilities for Nemotron-Cascade-2 recipe environments."""

import re


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output.

    Also strips unclosed <think> tags (from truncated responses).
    """
    stripped = re.sub(r'<think>[\s\S]*?</think>', '', text)
    # If the model started thinking but never closed the tag, strip from <think> onward
    stripped = re.sub(r'<think>[\s\S]*$', '', stripped)
    return stripped.strip()
