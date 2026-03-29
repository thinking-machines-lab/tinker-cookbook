"""MATH-500 benchmark -- Hendrycks MATH test set."""

from __future__ import annotations

import logging
from typing import cast

import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook.renderers import Message, Renderer
from tinker_cookbook.recipes.nemotron_cascade.eval.base import EvalResult
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import (
    get_text,
    make_completer,
    run_concurrent_eval,
)

logger = logging.getLogger(__name__)


async def evaluate(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    max_tokens: int = 32768,
    max_examples: int | None = None,
) -> EvalResult:
    """Evaluate on MATH-500 with boxed-answer grading."""
    from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer

    ds = cast(Dataset, load_dataset("HuggingFaceH4/MATH-500", split="test"))
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    completer = make_completer(sampling_client, renderer, max_tokens)

    async def eval_one(row: dict) -> dict | None:
        problem = row["problem"]
        try:
            expected = extract_boxed(row["solution"])
        except ValueError:
            return None

        messages: list[Message] = [
            {"role": "user", "content": problem + " Put your final answer in \\boxed{}."},
        ]
        try:
            response = await completer(messages)
            content = get_text(response)
            try:
                given = extract_boxed(content)
                correct = grade_answer(given, expected)
            except ValueError:
                correct = False
            return {"correct": correct, "input": problem, "output": content, "expected": expected}
        except Exception as e:
            logger.warning(f"MATH-500 eval failed: {e}")
            return None

    logger.info(f"MATH-500: evaluating {len(ds)} samples")
    results = await run_concurrent_eval(list(ds), eval_one)

    valid = [r for r in results if r is not None]
    num_correct = sum(1 for r in valid if r["correct"])
    accuracy = num_correct / len(valid) if valid else 0.0
    logger.info(f"MATH-500 final: {num_correct}/{len(valid)} = {accuracy:.4f}")

    return EvalResult(
        benchmark="math500",
        score=accuracy,
        num_examples=len(valid),
        num_correct=num_correct,
        metrics={"math500/accuracy": accuracy},
        examples=valid,
    )
