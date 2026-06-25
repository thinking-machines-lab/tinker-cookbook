"""Pull a runnable RILL program out of a model message."""

from __future__ import annotations

import re

# Matches a fenced code block, optionally tagged ```rill (or ```text, ```, etc.).
_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)


def extract_program(text: str) -> str:
    """Take the last fenced block (preferring a ```rill one); else the whole message.

    Models reason first and emit the final program last, so the last block is the
    intended answer. With no fences, treat the whole message as the program.
    """
    blocks = _FENCE_RE.findall(text)
    if blocks:
        rill_blocks = [body for tag, body in blocks if tag.lower() == "rill"]
        chosen = rill_blocks[-1] if rill_blocks else blocks[-1][1]
        return chosen.strip("\n")
    return text.strip()
