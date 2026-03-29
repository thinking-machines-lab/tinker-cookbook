"""IFEval benchmark -- instruction-following verification."""

from __future__ import annotations

import json
import logging
import os

import tinker

from tinker_cookbook.renderers import Message, Renderer
from tinker_cookbook.recipes.nemotron_cascade.eval.base import EvalResult
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import (
    get_text,
    make_completer,
    run_concurrent_eval,
)

logger = logging.getLogger(__name__)


def _parse_kwargs(raw_kwargs: list, instruction_ids: list) -> list[dict]:
    kwargs_list = []
    for kw in raw_kwargs:
        if kw is None:
            kwargs_list.append({})
        elif isinstance(kw, str):
            try:
                kwargs_list.append(json.loads(kw))
            except json.JSONDecodeError:
                kwargs_list.append({})
        elif isinstance(kw, dict):
            kwargs_list.append(kw)
        else:
            kwargs_list.append({})
    while len(kwargs_list) < len(instruction_ids):
        kwargs_list.append({})
    return kwargs_list


async def evaluate(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    max_tokens: int = 32768,
    max_examples: int | None = None,
) -> EvalResult:
    """Evaluate on IFEval using the Nemotron-Cascade IF verifier."""
    from tinker_cookbook.recipes.nemotron_cascade.rl.envs.if_rl import verify_all_instructions

    ifeval_path = os.path.expanduser("~/data/nemotron-cascade-2/rl_if_rl.jsonl")
    rows: list[dict] = []
    with open(ifeval_path) as f:
        for line in f:
            rows.append(json.loads(line))

    if max_examples:
        rows = rows[:max_examples]

    completer = make_completer(sampling_client, renderer, max_tokens)

    async def eval_one(row: dict) -> dict | None:
        prompt_messages = row["responses_create_params"]["input"]
        instruction_ids = row.get("instruction_id_list", [])
        raw_kwargs = row.get("kwargs", [])
        kwargs_list = _parse_kwargs(raw_kwargs, instruction_ids)

        messages: list[Message] = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in prompt_messages
        ]
        try:
            response = await completer(messages)
            content = get_text(response)
            fraction, _ = verify_all_instructions(content, instruction_ids, kwargs_list)
            return {
                "correct": fraction == 1.0,
                "fraction": fraction,
                "input": messages[-1]["content"] if messages else "",
                "output": content,
            }
        except Exception as e:
            logger.warning(f"IFEval eval failed: {e}")
            return None

    logger.info(f"IFEval: evaluating {len(rows)} samples")
    results = await run_concurrent_eval(rows, eval_one)

    valid = [r for r in results if r is not None]
    total_score = sum(r["fraction"] for r in valid)
    strict_correct = sum(1 for r in valid if r["correct"])
    loose_acc = total_score / len(valid) if valid else 0.0
    strict_acc = strict_correct / len(valid) if valid else 0.0
    logger.info(f"IFEval final: loose={loose_acc:.4f}, strict={strict_acc:.4f} ({len(valid)} prompts)")

    return EvalResult(
        benchmark="ifeval",
        score=strict_acc,
        num_examples=len(valid),
        num_correct=strict_correct,
        metrics={
            "ifeval/loose_accuracy": loose_acc,
            "ifeval/strict_accuracy": strict_acc,
        },
        examples=valid,
    )
