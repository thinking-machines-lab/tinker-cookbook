"""AIME 2025 benchmark -- math competition problems with integer answers (0-999)."""

from __future__ import annotations

import logging
import re
from typing import cast

import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook.renderers import Message, Renderer
from tinker_cookbook.recipes.nemotron_cascade.eval.base import EvalResult
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import (
    extract_boxed,
    extract_gsm8k_answer,
    extract_number,
    get_text,
    make_completer,
    run_concurrent_eval,
)

logger = logging.getLogger(__name__)

# Known HuggingFace dataset IDs for AIME 2025 (tried in order)
_AIME_DATASET_IDS = (
    "HuggingFaceH4/aime-2025",
    "yentinglin/aime_2025",
    "Maxwell-Jia/AIME_2025",
    "opencompass/AIME2025",
    "di-zhang-fdu/AIME24-25",
)


def _load_aime_dataset() -> Dataset | None:
    for dataset_id in _AIME_DATASET_IDS:
        for split in ("test", "train"):
            try:
                ds = cast(Dataset, load_dataset(dataset_id, split=split))
                logger.info(f"Loaded AIME 2025 from {dataset_id}/{split} ({len(ds)} problems)")
                return ds
            except Exception:
                continue
    return None


async def evaluate(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    max_tokens: int = 32768,
    max_examples: int | None = None,
) -> EvalResult:
    """Evaluate on AIME 2025 -- math competition problems with integer answers."""
    ds = _load_aime_dataset()
    if ds is None:
        logger.warning("Could not load AIME 2025 dataset from HuggingFace. Skipping.")
        return EvalResult(benchmark="aime2025", score=0.0, num_examples=0, num_correct=0)

    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    completer = make_completer(sampling_client, renderer, max_tokens)

    async def eval_one(row: dict) -> dict | None:
        problem = row.get("problem", row.get("question", row.get("Problem", "")))
        expected_raw = row.get("answer", row.get("Answer", row.get("expected_answer", "")))
        if not problem or expected_raw is None:
            return None

        try:
            expected = int(str(expected_raw).strip())
        except (ValueError, TypeError):
            m = re.search(r"\d+", str(expected_raw))
            if m:
                expected = int(m.group(0))
            else:
                return None

        prompt = (
            f"{problem}\n\n"
            "This is an AIME problem. The answer is an integer from 000 to 999. "
            "Show your work step by step, then put your final answer in \\boxed{}."
        )
        messages: list[Message] = [{"role": "user", "content": prompt}]
        try:
            response = await completer(messages)
            content = get_text(response)
            boxed = extract_boxed(content)
            extracted_str = extract_number(boxed) if boxed else extract_gsm8k_answer(content)
            try:
                extracted_val = int(float(extracted_str))
                correct = extracted_val == expected
            except (ValueError, TypeError):
                correct = False
            return {"correct": correct, "input": problem, "output": content, "expected": str(expected)}
        except Exception as e:
            logger.warning(f"AIME 2025 eval failed: {e}")
            return None

    logger.info(f"AIME 2025: evaluating {len(ds)} samples")
    results = await run_concurrent_eval(list(ds), eval_one)

    valid = [r for r in results if r is not None]
    num_correct = sum(1 for r in valid if r["correct"])
    accuracy = num_correct / len(valid) if valid else 0.0
    logger.info(f"AIME 2025 final: {num_correct}/{len(valid)} = {accuracy:.4f}")

    return EvalResult(
        benchmark="aime2025",
        score=accuracy,
        num_examples=len(valid),
        num_correct=num_correct,
        metrics={"aime2025/accuracy": accuracy},
        examples=valid,
    )
