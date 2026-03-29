"""Shared utilities for benchmark evaluation modules."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from typing import TypeVar

import tinker

from tinker_cookbook import renderers
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.renderers import Message, Renderer
from tinker_cookbook.recipes.nemotron_cascade.eval.base import EvalResult

logger = logging.getLogger(__name__)

T = TypeVar("T")


def make_completer(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    max_tokens: int,
    temperature: float = 0.6,
) -> TinkerMessageCompleter:
    """Create a TinkerMessageCompleter from a sampling client and renderer."""
    return TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=max_tokens,
        temperature=temperature,
    )


async def run_concurrent_eval(
    items: list[T],
    eval_fn: Callable[[T], Awaitable[dict | None]],
    concurrency: int = 128,
) -> list[dict | None]:
    """Run *eval_fn* over *items* with bounded concurrency.

    Each call to *eval_fn* should return a result dict or ``None`` on failure.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _guarded(item: T) -> dict | None:
        async with semaphore:
            return await eval_fn(item)

    return await asyncio.gather(*[_guarded(item) for item in items])


# ---------------------------------------------------------------------------
# Answer extraction helpers (shared across math benchmarks)
# ---------------------------------------------------------------------------


def extract_boxed(text: str) -> str | None:
    r"""Extract content from ``\boxed{...}`` handling nested braces."""
    idx = text.find("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1] if depth == 0 else None


def extract_number(text: str) -> str:
    """Extract a number from *text*, stripping LaTeX formatting."""
    cleaned = re.sub(r"\\text\{[^}]*\}", "", text)
    cleaned = re.sub(r"\\[a-zA-Z]+", "", cleaned)
    cleaned = cleaned.replace("{", "").replace("}", "").replace("$", "")
    cleaned = cleaned.replace(",", "").replace(" ", "")
    match = re.search(r"[-]?\d+\.?\d*", cleaned)
    return match.group(0) if match else cleaned.strip()


def extract_gsm8k_answer(text: str) -> str:
    """Extract a numeric answer from a model response.

    Tries (in order): ``\\boxed{}``, ``#### answer``, "the answer is X",
    and finally the last number in the text.
    """
    boxed = extract_boxed(text)
    if boxed:
        return extract_number(boxed)

    hash_match = re.search(r"####\s*(.+)", text)
    if hash_match:
        return extract_number(hash_match.group(1))

    answer_match = re.search(
        r"(?:answer is|answer:)\s*\$?([0-9,.-]+)", text, re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).strip().replace(",", "")

    numbers = re.findall(r"[-]?\d+[,\d]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def extract_mcq_answer(text: str, valid_letters: str = "ABCD") -> str:
    """Extract a multiple-choice letter from a model response.

    Tries boxed format, "answer is (X)" pattern, then last standalone letter.
    """
    pattern = f"[{valid_letters}]"

    boxed = extract_boxed(text)
    if boxed and re.fullmatch(pattern, boxed.strip().upper()):
        return boxed.strip().upper()

    answer_match = re.search(
        rf"(?:answer is|answer:)\s*\(?([{valid_letters}])\)?",
        text,
        re.IGNORECASE,
    )
    if answer_match:
        return answer_match.group(1).upper()

    letters = re.findall(rf"\b({pattern})\b", text[-300:])
    if letters:
        return letters[-1]
    return ""


def get_text(response: list) -> str:
    """Extract text content from a completer response."""
    return renderers.get_text_content(response)
