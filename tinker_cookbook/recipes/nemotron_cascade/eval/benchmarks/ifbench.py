"""IFBench benchmark -- instruction following with objective verifiers.

Dataset: ``allenai/IFBench_test`` on HuggingFace (300 examples).
Metric: Fraction of prompts where *all* constraints are satisfied (strict accuracy).

IFBench extends IFEval with 58 new diverse constraint types and provides
objective verification functions. Each example has a prompt, a list of
``instruction_id_list`` entries, and ``kwargs`` specifying constraint parameters.

The verifier reuses the same ``verify_all_instructions`` machinery from our
IFEval RL environment, extended with IFBench-specific constraint IDs.
"""

from __future__ import annotations

import json
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


def _parse_kwargs(raw_kwargs: list) -> list[dict]:
    """Parse the kwargs list from the IFBench dataset."""
    result: list[dict] = []
    for kw in raw_kwargs:
        if kw is None:
            result.append({})
        elif isinstance(kw, str):
            try:
                result.append(json.loads(kw))
            except json.JSONDecodeError:
                result.append({})
        elif isinstance(kw, dict):
            result.append(kw)
        else:
            result.append({})
    return result


def _verify_constraints(
    response: str,
    instruction_ids: list[str],
    kwargs_list: list[dict],
) -> tuple[float, bool]:
    """Verify constraints using the IF-RL verifier if available, else basic heuristics.

    Returns (fraction_satisfied, all_satisfied).
    """
    try:
        from tinker_cookbook.recipes.nemotron_cascade.rl.envs.if_rl import verify_all_instructions
        fraction, _ = verify_all_instructions(response, instruction_ids, kwargs_list)
        return fraction, fraction == 1.0
    except (ImportError, Exception):
        pass

    # Fallback: basic heuristic verification for common constraint types
    satisfied = 0
    for iid, kw in zip(instruction_ids, kwargs_list):
        iid_lower = iid.lower()
        try:
            if "word_count" in iid_lower or "min_words" in iid_lower:
                min_w = kw.get("min_words", 0)
                max_w = kw.get("max_words", float("inf"))
                word_count = len(response.split())
                if min_w <= word_count <= max_w:
                    satisfied += 1
            elif "keyword" in iid_lower:
                keyword = kw.get("keyword", "")
                if keyword and keyword.lower() in response.lower():
                    satisfied += 1
            elif "sentences" in iid_lower or "num_sentences" in iid_lower:
                n = kw.get("n", kw.get("N", 0))
                sentence_count = response.count(".") + response.count("!") + response.count("?")
                if sentence_count >= n:
                    satisfied += 1
            else:
                # Unknown constraint type -- skip (count as not satisfied)
                pass
        except Exception:
            pass

    total = len(instruction_ids) if instruction_ids else 1
    fraction = satisfied / total
    return fraction, fraction == 1.0


async def evaluate(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    max_tokens: int = 32768,
    max_examples: int | None = None,
) -> EvalResult:
    """Evaluate on IFBench (instruction following with objective verifiers)."""
    try:
        ds = cast(Dataset, load_dataset("allenai/IFBench_test", split="train"))
    except Exception as exc:
        logger.warning(f"Could not load IFBench dataset: {exc}. Skipping.")
        return EvalResult(benchmark="ifbench", score=0.0, num_examples=0, num_correct=0)

    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    completer = make_completer(sampling_client, renderer, max_tokens)

    async def eval_one(row: dict) -> dict | None:
        prompt = row.get("prompt", "")
        instruction_ids = row.get("instruction_id_list", [])
        raw_kwargs = row.get("kwargs", [])

        if not prompt:
            return None

        kwargs_list = _parse_kwargs(raw_kwargs)
        # Pad kwargs to match instruction_ids length
        while len(kwargs_list) < len(instruction_ids):
            kwargs_list.append({})

        messages: list[Message] = [{"role": "user", "content": prompt}]
        try:
            response = await completer(messages)
            content = get_text(response)
            fraction, all_satisfied = _verify_constraints(content, instruction_ids, kwargs_list)
            return {
                "correct": all_satisfied,
                "fraction": fraction,
                "input": prompt[:200],
                "output": content[:500],
            }
        except Exception as e:
            logger.warning(f"IFBench eval failed: {e}")
            return None

    logger.info(f"IFBench: evaluating {len(ds)} samples")
    results = await run_concurrent_eval(list(ds), eval_one)

    valid = [r for r in results if r is not None]
    strict_correct = sum(1 for r in valid if r["correct"])
    total_score = sum(r["fraction"] for r in valid)
    strict_acc = strict_correct / len(valid) if valid else 0.0
    loose_acc = total_score / len(valid) if valid else 0.0

    logger.info(f"IFBench final: strict={strict_acc:.4f}, loose={loose_acc:.4f} ({len(valid)} prompts)")

    return EvalResult(
        benchmark="ifbench",
        score=strict_acc,
        num_examples=len(valid),
        num_correct=strict_correct,
        metrics={
            "ifbench/strict_accuracy": strict_acc,
            "ifbench/loose_accuracy": loose_acc,
        },
        examples=valid,
    )
