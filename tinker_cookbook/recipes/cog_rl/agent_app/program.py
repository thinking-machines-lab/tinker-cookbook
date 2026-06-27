"""Pull a runnable Cog program out of a model message."""

from __future__ import annotations

import re

# Matches a fenced code block, optionally tagged ```cog (or ```text, ```, etc.).
_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)


def extract_program(text: str) -> str:
    """Take the last fenced block (preferring a ```cog one); else the whole message.

    Models reason first and emit the final program last, so the last block is the
    intended answer. With no fences, treat the whole message as the program.
    """
    blocks = _FENCE_RE.findall(text)
    if blocks:
        cog_blocks = [body for tag, body in blocks if tag.lower() == "cog"]
        chosen = cog_blocks[-1] if cog_blocks else blocks[-1][1]
        return chosen.strip("\n")
    return text.strip()
