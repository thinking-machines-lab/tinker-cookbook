"""LongBench v2 benchmark -- long-context comprehension across multiple subtasks."""

from __future__ import annotations

import logging
import re
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


def _load_longbench() -> tuple[Dataset | None, str]:
    """Try loading LongBench v2, falling back to v1. Returns (dataset, version)."""
    for dataset_id, split in (
        ("THUDM/LongBench-v2", "test"),
        ("THUDM/LongBench-v2", "train"),
        ("THUDM/LongBench", "test"),
    ):
        try:
            ds = cast(Dataset, load_dataset(dataset_id, split=split))
            version = "v2" if "v2" in dataset_id else "v1"
            logger.info(f"Loaded LongBench from {dataset_id}/{split} ({len(ds)} examples)")
            return ds, version
        except Exception:
            try:
                ds = cast(Dataset, load_dataset(dataset_id, "default", split=split))
                version = "v2" if "v2" in dataset_id else "v1"
                logger.info(f"Loaded LongBench from {dataset_id}/default/{split} ({len(ds)} examples)")
                return ds, version
            except Exception:
                continue
    return None, "v2"


async def evaluate(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    max_tokens: int = 32768,
    max_examples: int | None = None,
) -> EvalResult:
    """Evaluate on LongBench v2 (long-context comprehension, multiple subtasks)."""
    ds, dataset_version = _load_longbench()
    if ds is None:
        logger.warning("Could not load LongBench dataset. Skipping.")
        return EvalResult(benchmark="longbench", score=0.0, num_examples=0, num_correct=0)

    if max_examples:
        ds = ds.shuffle(seed=42).select(range(min(max_examples, len(ds))))

    completer = make_completer(sampling_client, renderer, max_tokens)

    async def eval_one(row: dict) -> dict | None:
        context = row.get("context", "")
        subtask = row.get("domain", row.get("dataset", "unknown"))

        if dataset_version == "v2":
            question = row.get("question", row.get("input", ""))
            choices = []
            for letter in ("A", "B", "C", "D"):
                choice = row.get(f"choice_{letter}", "")
                if choice:
                    choices.append(choice)

            expected = str(row.get("answer", "")).strip().upper()

            if choices:
                choice_text = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
                user_content = (
                    f"Read the following text carefully, then answer the question.\n\n"
                    f"--- TEXT ---\n{context}\n--- END TEXT ---\n\n"
                    f"Question: {question}\n\n{choice_text}\n\n"
                    "Answer with just the letter (A, B, C, or D)."
                )
            else:
                user_content = (
                    f"Read the following text carefully, then answer the question.\n\n"
                    f"--- TEXT ---\n{context}\n--- END TEXT ---\n\n"
                    f"Question: {question}\n\nGive a concise answer."
                )
        else:
            question = row.get("input", "")
            expected_answers = row.get("answers", row.get("all_classes", []))
            expected = expected_answers[0] if expected_answers else ""
            user_content = (
                f"Read the following text carefully, then answer the question.\n\n"
                f"--- TEXT ---\n{context}\n--- END TEXT ---\n\n"
                f"Question: {question}\n\nGive a concise answer."
            )

        if not question or not context:
            return None

        messages: list[Message] = [{"role": "user", "content": user_content}]
        try:
            response = await completer(messages)
            content = get_text(response)

            if dataset_version == "v2" and expected in ("A", "B", "C", "D"):
                letters = re.findall(r"\b([A-D])\b", content[-300:])
                extracted = letters[-1] if letters else ""
                correct = extracted == expected
            else:
                correct = str(expected).strip().lower() in content.strip().lower()

            return {"correct": correct, "subtask": subtask, "input": question[:200], "output": content[:200]}
        except Exception as e:
            logger.warning(f"LongBench eval failed: {e}")
            return None

    logger.info(f"LongBench ({dataset_version}): evaluating {len(ds)} samples")
    results = await run_concurrent_eval(list(ds), eval_one)

    valid = [r for r in results if r is not None]
    num_correct = sum(1 for r in valid if r["correct"])
    accuracy = num_correct / len(valid) if valid else 0.0

    # Per-subtask breakdown
    subtask_results: dict[str, list[bool]] = {}
    for r in valid:
        st = r["subtask"]
        subtask_results.setdefault(st, []).append(r["correct"])

    metrics: dict[str, float] = {"longbench/accuracy": accuracy}
    for st, st_results in sorted(subtask_results.items()):
        metrics[f"longbench/{st}/accuracy"] = sum(st_results) / len(st_results) if st_results else 0.0

    logger.info(f"LongBench final: {num_correct}/{len(valid)} = {accuracy:.4f}")

    return EvalResult(
        benchmark="longbench",
        score=accuracy,
        num_examples=len(valid),
        num_correct=num_correct,
        metrics=metrics,
        examples=valid,
    )
