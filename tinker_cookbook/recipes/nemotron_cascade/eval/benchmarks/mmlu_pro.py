"""MMLU-Pro benchmark -- multi-task language understanding (professional)."""

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
    """Evaluate on MMLU-Pro (0-shot multiple choice)."""
    try:
        ds = cast(Dataset, load_dataset("TIGER-Lab/MMLU-Pro", split="test"))
    except Exception:
        ds = cast(Dataset, load_dataset("cais/mmlu", "all", split="test"))

    if max_examples:
        ds = ds.shuffle(seed=42).select(range(min(max_examples, len(ds))))

    completer = make_completer(sampling_client, renderer, max_tokens)

    # MMLU-Pro can have up to 10 options (A-J)
    valid_letters = "ABCDEFGHIJ"

    async def eval_one(row: dict) -> dict | None:
        question = row.get("question", row.get("input", ""))
        choices = row.get("options", row.get("choices", []))
        answer_idx = row.get("answer_index", row.get("answer", None))

        if choices:
            choice_text = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
            prompt = f"{question}\n\n{choice_text}\n\nAnswer with just the letter (A, B, C, D, etc.)."
        else:
            prompt = f"{question}\n\nAnswer with just the letter."

        if isinstance(answer_idx, int):
            expected = chr(65 + answer_idx)
        elif isinstance(answer_idx, str) and len(answer_idx) == 1:
            expected = answer_idx.upper()
        else:
            expected = str(answer_idx).strip().upper()

        messages: list[Message] = [{"role": "user", "content": prompt}]
        try:
            response = await completer(messages)
            content = get_text(response)
            extracted = extract_mcq_answer(content, valid_letters)
            correct = extracted == expected
            return {"correct": correct, "input": prompt, "output": content, "expected": expected}
        except Exception as e:
            logger.warning(f"MMLU-Pro eval failed: {e}")
            return None

    logger.info(f"MMLU-Pro: evaluating {len(ds)} samples")
    results = await run_concurrent_eval(list(ds), eval_one)

    valid = [r for r in results if r is not None]
    num_correct = sum(1 for r in valid if r["correct"])
    accuracy = num_correct / len(valid) if valid else 0.0
    logger.info(f"MMLU-Pro final: {num_correct}/{len(valid)} = {accuracy:.4f}")

    return EvalResult(
        benchmark="mmlu_pro",
        score=accuracy,
        num_examples=len(valid),
        num_correct=num_correct,
        metrics={"mmlu_pro/accuracy": accuracy},
        examples=valid,
    )
