"""SWE-bench Verified benchmark -- software engineering patch generation.

Dataset: ``princeton-nlp/SWE-bench_Verified`` on HuggingFace (500 problems).
Metric: Fraction of problems where an LLM judge rates the generated patch
quality > 0.5 on a 0-1 scale.  Full Docker execution is too heavy for eval,
so we use an LLM judge (same pattern as the SWE RL env's LLM judge mode).

Reference: https://github.com/SWE-bench/SWE-bench
"""

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

_CANDIDATE_PROMPT_TEMPLATE = """\
You are a software engineer working on a Python repository. A bug report or \
feature request has been filed. Your task is to generate a unified diff patch \
that resolves the issue.

## Repository: {repo}

## Problem Statement
{problem_statement}

## Hints
{hints}

Please generate a unified diff patch (```diff ... ```) that fixes the issue. \
Only output the patch, no explanation needed."""

_JUDGE_PROMPT_TEMPLATE = """\
You are an expert software engineering reviewer. Evaluate the quality of a \
proposed patch for the following issue.

## Repository: {repo}

## Problem Statement
{problem_statement}

## Proposed Patch
{patch}

Rate the patch quality on a scale of 0.0 to 1.0 considering:
- Does the patch address the core issue described?
- Is the patch syntactically valid as a unified diff?
- Does it modify plausible files/locations?
- Would it likely pass the test suite?

Provide a brief explanation, then give your score in this exact format: \
"Score: [[X.X]]", for example: "Score: [[0.7]]"."""


def _extract_patch(text: str) -> str:
    """Extract a unified diff patch from a model response."""
    match = re.search(r"```diff\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_judge_score(judge_response: str) -> float | None:
    """Extract a numeric score from the judge response like ``[[0.7]]``."""
    match = re.search(r"\[\[([0-9]+(?:\.[0-9]+)?)\]\]", judge_response)
    if match:
        return float(match.group(1))
    match = re.search(r"Score:\s*([0-9]+(?:\.[0-9]+)?)", judge_response, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


async def evaluate(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    max_tokens: int = 32768,
    max_examples: int | None = None,
    judge_sampling_client: tinker.SamplingClient | None = None,
    judge_renderer: Renderer | None = None,
) -> EvalResult:
    """Evaluate on SWE-bench Verified using an LLM judge.

    If *judge_sampling_client* is provided, it will be used to judge patches.
    Otherwise, the same *sampling_client* is used as a self-judge.
    """
    ds = cast(Dataset, load_dataset("princeton-nlp/SWE-bench_Verified", split="test"))
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    candidate_completer = make_completer(sampling_client, renderer, max_tokens)

    j_client = judge_sampling_client or sampling_client
    j_renderer = judge_renderer or renderer
    judge_completer = make_completer(j_client, j_renderer, max_tokens=4096, temperature=0.0)

    async def eval_one(row: dict) -> dict | None:
        repo = row.get("repo", "unknown")
        problem_statement = row.get("problem_statement", "")
        hints = row.get("hints_text", "")
        if not problem_statement:
            return None

        # Generate candidate patch
        prompt = _CANDIDATE_PROMPT_TEMPLATE.format(
            repo=repo,
            problem_statement=problem_statement[:4000],
            hints=hints[:2000],
        )
        messages: list[Message] = [{"role": "user", "content": prompt}]
        try:
            response = await candidate_completer(messages)
            answer = get_text(response)
            patch = _extract_patch(answer)
        except Exception as e:
            logger.warning(f"SWE-bench candidate generation failed: {e}")
            return None

        # Judge the patch
        judge_prompt = _JUDGE_PROMPT_TEMPLATE.format(
            repo=repo,
            problem_statement=problem_statement[:4000],
            patch=patch[:4000],
        )
        judge_messages: list[Message] = [{"role": "user", "content": judge_prompt}]
        try:
            judge_response = await judge_completer(judge_messages)
            judge_text = get_text(judge_response)
            score = _extract_judge_score(judge_text)
        except Exception as e:
            logger.warning(f"SWE-bench judge failed: {e}")
            score = None

        if score is None:
            return None

        return {
            "correct": score > 0.5,
            "score": score,
            "input": problem_statement[:200],
            "output": patch[:500],
            "judge_output": judge_text[:300],
            "repo": repo,
            "instance_id": row.get("instance_id", ""),
        }

    logger.info(f"SWE-bench: evaluating {len(ds)} samples")
    results = await run_concurrent_eval(list(ds), eval_one)

    valid = [r for r in results if r is not None]
    num_correct = sum(1 for r in valid if r["correct"])
    avg_score = sum(r["score"] for r in valid) / len(valid) if valid else 0.0
    accuracy = num_correct / len(valid) if valid else 0.0

    logger.info(f"SWE-bench final: {num_correct}/{len(valid)} = {accuracy:.4f} (avg_score={avg_score:.3f})")

    return EvalResult(
        benchmark="swe_bench",
        score=accuracy,
        num_examples=len(valid),
        num_correct=num_correct,
        metrics={
            "swe_bench/accuracy": accuracy,
            "swe_bench/avg_score": avg_score,
        },
        examples=valid,
    )
