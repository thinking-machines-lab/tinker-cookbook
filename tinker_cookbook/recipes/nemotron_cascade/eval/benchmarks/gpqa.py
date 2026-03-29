"""GPQA-Diamond benchmark -- graduate-level science QA (multiple choice)."""

from __future__ import annotations

import logging
from typing import cast

import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook.renderers import Message, Renderer
from tinker_cookbook.recipes.nemotron_cascade.eval.base import EvalResult
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import (
    extract_mcq_answer,
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
    """Evaluate on GPQA-Diamond (hard graduate-level science, A/B/C/D)."""
    ds = cast(Dataset, load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train"))
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    completer = make_completer(sampling_client, renderer, max_tokens)

    async def eval_one(row: dict) -> dict | None:
        question = row["Question"]
        correct_answer = row.get("Answer", row.get("Correct Answer", ""))

        # Build choices from the available choice columns
        choice_cols = [
            col for col in row.keys()
            if col.startswith("Choice") or col in ("choice_a", "choice_b", "choice_c", "choice_d")
        ]

        if choice_cols:
            choices = [row[c] for c in sorted(choice_cols) if row.get(c)]
        else:
            choices = [row.get("Correct Answer", "")]
            for i in range(1, 4):
                inc = row.get(f"Incorrect Answer {i}", "")
                if inc:
                    choices.append(inc)

        if not choices:
            return None

        # Determine expected letter
        if correct_answer in ("A", "B", "C", "D"):
            expected = correct_answer
        else:
            expected = "A"
            for i, c in enumerate(choices):
                if c.strip() == str(correct_answer).strip():
                    expected = chr(65 + i)
                    break

        choice_text = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
        prompt = (
            f"{question}\n\n{choice_text}\n\n"
            "Think step by step, then give your final answer as a single letter (A, B, C, or D)."
        )
        messages: list[Message] = [{"role": "user", "content": prompt}]
        try:
            response = await completer(messages)
            content = get_text(response)
            extracted = extract_mcq_answer(content)
            correct = extracted == expected
            return {"correct": correct, "input": prompt, "output": content, "expected": expected}
        except Exception as e:
            logger.warning(f"GPQA eval failed: {e}")
            return None

    logger.info(f"GPQA-Diamond: evaluating {len(ds)} samples")
    results = await run_concurrent_eval(list(ds), eval_one)

    valid = [r for r in results if r is not None]
    num_correct = sum(1 for r in valid if r["correct"])
    accuracy = num_correct / len(valid) if valid else 0.0
    logger.info(f"GPQA-Diamond final: {num_correct}/{len(valid)} = {accuracy:.4f}")

    return EvalResult(
        benchmark="gpqa_diamond",
        score=accuracy,
        num_examples=len(valid),
        num_correct=num_correct,
        metrics={"gpqa_diamond/accuracy": accuracy},
        examples=valid,
    )
