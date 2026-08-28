"""Shared grading helpers for the audio recipes: renderer-safe response
decoding, transcript normalization, and word error rate (WER).

Dataset-agnostic by design, so every audio recipe (and its evaluator) grades
transcripts the same way. Task-specific prompts, parsing, and rewards live
with each recipe (e.g. ``emotion/env.py``).
"""

from __future__ import annotations

import logging
import re

from tinker_cookbook import renderers

logger = logging.getLogger(__name__)

_SPECIAL_TOKEN_TEXT = re.compile(r"<\|[^|]+\|>")


def parse_response_text(
    renderer: renderers.Renderer, action: list[int]
) -> tuple[str, renderers.ParseTermination]:
    """Decode one sampled response to plain text, grading parser rejections
    (e.g. a rollout truncated mid-UTF-8 sequence at temperature 1) as
    malformed-and-empty instead of crashing the rollout or eval."""
    try:
        message, termination = renderer.parse_response(action)
    except ValueError as e:
        logger.warning(f"Renderer rejected a sampled response; grading as malformed: {e}")
        return "", renderers.ParseTermination.MALFORMED
    text = _SPECIAL_TOKEN_TEXT.sub(" ", renderers.get_text_content(message)).strip()
    return text, termination


def normalize_text(text: str) -> str:
    """Lowercase, strip non-alphanumerics, collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split())


def clip_wer(ref: str, hyp: str) -> float:
    """WER of one hypothesis against one reference, on normalized text."""
    import jiwer

    ref_norm = normalize_text(ref)
    return float(jiwer.wer(ref_norm, normalize_text(hyp))) if ref_norm else 0.0


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    """One corpus-level WER over normalized (ref, hyp) pairs, not a mean of per-clip WERs."""
    import jiwer

    pairs = [
        (rn, normalize_text(h))
        for r, h in zip(refs, hyps, strict=True)
        if (rn := normalize_text(r))
    ]
    if not pairs:
        return 0.0
    r, h = map(list, zip(*pairs))
    return float(jiwer.wer(r, h))
