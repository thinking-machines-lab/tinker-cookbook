"""MMLU-Redux benchmark -- cleaner subset of MMLU with verified annotations.

Dataset: ``edinburgh-dawg/mmlu-redux`` on HuggingFace.
Metric: Multiple-choice accuracy (A/B/C/D).

MMLU-Redux re-annotates 3,000 MMLU examples (30 subjects x 100) and flags
questions with errors. We evaluate only on samples with ``error_type == "ok"``.
"""

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

# All 30 MMLU-Redux subjects
_SUBJECTS = [
    "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "conceptual_physics",
    "econometrics", "electrical_engineering", "formal_logic", "global_facts",
    "high_school_chemistry", "high_school_geography", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_physics", "high_school_statistics",
    "high_school_us_history", "human_aging", "logical_fallacies",
    "machine_learning", "miscellaneous", "philosophy",
    "professional_accounting", "professional_law", "public_relations", "virology",
]


def _load_mmlu_redux(max_examples: int | None) -> list[dict]:
    """Load all MMLU-Redux subjects, filtering to error_type=='ok'."""
    all_rows: list[dict] = []
    for subject in _SUBJECTS:
        try:
            ds = cast(Dataset, load_dataset("edinburgh-dawg/mmlu-redux", subject, split="test"))
            for row in ds:
                row_dict = dict(row)
                row_dict["_subject"] = subject
                # Only keep samples flagged as correct
                if row_dict.get("error_type", "ok") == "ok":
                    all_rows.append(row_dict)
        except Exception as e:
            logger.warning(f"Could not load MMLU-Redux/{subject}: {e}")
            continue

    if max_examples and len(all_rows) > max_examples:
        # Deterministic subsample
        import random
        rng = random.Random(42)
        rng.shuffle(all_rows)
        all_rows = all_rows[:max_examples]

    return all_rows


async def evaluate(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    max_tokens: int = 32768,
    max_examples: int | None = None,
) -> EvalResult:
    """Evaluate on MMLU-Redux (multiple choice accuracy, filtered to verified samples)."""
    rows = _load_mmlu_redux(max_examples)
    if not rows:
        logger.warning("Could not load MMLU-Redux dataset. Skipping.")
        return EvalResult(benchmark="mmlu_redux", score=0.0, num_examples=0, num_correct=0)

    completer = make_completer(sampling_client, renderer, max_tokens)

    async def eval_one(row: dict) -> dict | None:
        question = row.get("question", "")
        choices = row.get("choices", [])
        answer_idx = row.get("answer", 0)  # int index 0-3

        if not question or not choices:
            return None

        expected = chr(65 + int(answer_idx))
        choice_text = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
        prompt = f"{question}\n\n{choice_text}\n\nAnswer with just the letter (A, B, C, or D)."

        messages: list[Message] = [{"role": "user", "content": prompt}]
        try:
            response = await completer(messages)
            content = get_text(response)
            extracted = extract_mcq_answer(content)
            correct = extracted == expected
            return {
                "correct": correct,
                "input": prompt,
                "output": content,
                "expected": expected,
                "subject": row.get("_subject", "unknown"),
            }
        except Exception as e:
            logger.warning(f"MMLU-Redux eval failed: {e}")
            return None

    logger.info(f"MMLU-Redux: evaluating {len(rows)} samples")
    results = await run_concurrent_eval(rows, eval_one)

    valid = [r for r in results if r is not None]
    num_correct = sum(1 for r in valid if r["correct"])
    accuracy = num_correct / len(valid) if valid else 0.0

    # Per-subject breakdown
    subject_results: dict[str, list[bool]] = {}
    for r in valid:
        subj = r.get("subject", "unknown")
        subject_results.setdefault(subj, []).append(r["correct"])

    metrics: dict[str, float] = {"mmlu_redux/accuracy": accuracy}
    for subj, subj_res in sorted(subject_results.items()):
        metrics[f"mmlu_redux/{subj}/accuracy"] = sum(subj_res) / len(subj_res) if subj_res else 0.0

    logger.info(f"MMLU-Redux final: {num_correct}/{len(valid)} = {accuracy:.4f}")

    return EvalResult(
        benchmark="mmlu_redux",
        score=accuracy,
        num_examples=len(valid),
        num_correct=num_correct,
        metrics=metrics,
        examples=valid,
    )
