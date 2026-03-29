"""GSM8K benchmark -- grade-school math word problems."""

from __future__ import annotations

import logging
from typing import cast

import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook.renderers import Message, Renderer
from tinker_cookbook.recipes.nemotron_cascade.eval.base import EvalResult
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import (
    extract_gsm8k_answer,
    get_text,
    make_completer,
    run_concurrent_eval,
)

logger = logging.getLogger(__name__)


def _check_gsm8k(response: str, expected: str) -> bool:
    extracted = extract_gsm8k_answer(response)
    try:
        return abs(float(extracted) - float(expected.replace(",", ""))) < 1e-6
    except (ValueError, TypeError):
        return extracted.strip() == expected.strip()


async def evaluate(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    max_tokens: int = 32768,
    max_examples: int | None = None,
) -> EvalResult:
    """Evaluate on the GSM8K test set."""
    import re

    ds = cast(Dataset, load_dataset("openai/gsm8k", "main", split="test"))
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    completer = make_completer(sampling_client, renderer, max_tokens)

    async def eval_one(row: dict) -> dict | None:
        question = row["question"]
        answer_match = re.search(r"####\s*(.+)", row["answer"])
        expected = answer_match.group(1).strip().replace(",", "") if answer_match else ""
        messages: list[Message] = [
            {"role": "user", "content": question + " Show your work step by step, then give the final numerical answer."},
        ]
        try:
            response = await completer(messages)
            content = get_text(response)
            correct = _check_gsm8k(content, expected)
            return {"correct": correct, "input": question, "output": content, "expected": expected}
        except Exception as e:
            logger.warning(f"GSM8K eval failed: {e}")
            return None

    logger.info(f"GSM8K: evaluating {len(ds)} samples")
    results = await run_concurrent_eval(list(ds), eval_one)

    valid = [r for r in results if r is not None]
    num_correct = sum(1 for r in valid if r["correct"])
    accuracy = num_correct / len(valid) if valid else 0.0
    logger.info(f"GSM8K final: {num_correct}/{len(valid)} = {accuracy:.4f}")

    return EvalResult(
        benchmark="gsm8k",
        score=accuracy,
        num_examples=len(valid),
        num_correct=num_correct,
        metrics={"gsm8k/accuracy": accuracy},
        examples=valid,
    )
