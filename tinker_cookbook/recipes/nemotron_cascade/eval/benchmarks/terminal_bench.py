"""Terminal-Bench 2.0 benchmark -- terminal command generation.

Dataset: ``ia03/terminal-bench`` on HuggingFace (89 tasks).
Evaluation: The model generates a solution script for each task. An LLM judge
compares the proposed solution to the reference solution.  Full terminal
sandbox execution is too heavy for eval.

Metric: Fraction of tasks where the judge rates the solution as correct.

Reference: https://github.com/laude-institute/terminal-bench
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
You are an expert Linux/Unix systems administrator. Solve the following \
terminal task by writing the exact command(s) or script needed.

## Task
{task_description}

{environment_info}

Write a solution script that accomplishes the task. Provide your commands in \
a ```bash code block. Be precise and use standard Unix utilities."""

_JUDGE_PROMPT_TEMPLATE = """\
You are an expert evaluator of terminal/shell solutions. Compare a candidate \
solution to a reference solution for the following task.

## Task
{task_description}

## Reference Solution
{reference_solution}

## Candidate Solution
{candidate_solution}

Evaluate whether the candidate solution would correctly accomplish the task. \
Consider:
- Does it achieve the same end result as the reference?
- Are the commands syntactically correct?
- Would it work in a standard Linux environment?
- Minor stylistic differences are acceptable if the result is the same.

Provide a brief explanation, then give your verdict in this exact format: \
"Verdict: [[CORRECT]]" or "Verdict: [[INCORRECT]]"."""


def _extract_script(text: str) -> str:
    """Extract a bash script from a model response."""
    match = re.search(r"```(?:bash|sh|shell)\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_judge_verdict(judge_response: str) -> bool | None:
    """Extract a CORRECT/INCORRECT verdict from the judge response."""
    match = re.search(r"\[\[(CORRECT|INCORRECT)\]\]", judge_response, re.IGNORECASE)
    if match:
        return match.group(1).upper() == "CORRECT"
    # Fallback patterns
    match = re.search(r"Verdict:\s*(CORRECT|INCORRECT)", judge_response, re.IGNORECASE)
    if match:
        return match.group(1).upper() == "CORRECT"
    return None


async def evaluate(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    max_tokens: int = 32768,
    max_examples: int | None = None,
    judge_sampling_client: tinker.SamplingClient | None = None,
    judge_renderer: Renderer | None = None,
) -> EvalResult:
    """Evaluate on Terminal-Bench using an LLM judge.

    If *judge_sampling_client* is provided, it will be used to judge solutions.
    Otherwise, the same *sampling_client* is used as a self-judge.
    """
    ds = cast(Dataset, load_dataset("ia03/terminal-bench", split="test"))
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    candidate_completer = make_completer(sampling_client, renderer, max_tokens)

    j_client = judge_sampling_client or sampling_client
    j_renderer = judge_renderer or renderer
    judge_completer = make_completer(j_client, j_renderer, max_tokens=4096, temperature=0.0)

    async def eval_one(row: dict) -> dict | None:
        task_description = row.get("task", row.get("description", row.get("prompt", "")))
        reference_solution = row.get("solution", row.get("reference", row.get("answer", "")))
        environment_info = row.get("environment", row.get("setup", ""))

        if not task_description:
            return None

        env_section = f"## Environment\n{environment_info}" if environment_info else ""
        prompt = _CANDIDATE_PROMPT_TEMPLATE.format(
            task_description=task_description[:3000],
            environment_info=env_section,
        )
        messages: list[Message] = [{"role": "user", "content": prompt}]

        try:
            response = await candidate_completer(messages)
            answer = get_text(response)
            script = _extract_script(answer)
        except Exception as e:
            logger.warning(f"Terminal-Bench candidate generation failed: {e}")
            return None

        # Judge the solution
        if reference_solution:
            judge_prompt = _JUDGE_PROMPT_TEMPLATE.format(
                task_description=task_description[:3000],
                reference_solution=str(reference_solution)[:3000],
                candidate_solution=script[:3000],
            )
            judge_messages: list[Message] = [{"role": "user", "content": judge_prompt}]
            try:
                judge_response = await judge_completer(judge_messages)
                judge_text = get_text(judge_response)
                correct = _extract_judge_verdict(judge_text)
            except Exception as e:
                logger.warning(f"Terminal-Bench judge failed: {e}")
                correct = None
        else:
            # No reference solution available, skip judging
            judge_text = ""
            correct = None

        if correct is None:
            return None

        return {
            "correct": correct,
            "input": task_description[:200],
            "output": script[:500],
            "judge_output": judge_text[:300],
            "reference": str(reference_solution)[:200],
        }

    logger.info(f"Terminal-Bench: evaluating {len(ds)} samples")
    results = await run_concurrent_eval(list(ds), eval_one)

    valid = [r for r in results if r is not None]
    num_correct = sum(1 for r in valid if r["correct"])
    accuracy = num_correct / len(valid) if valid else 0.0

    logger.info(f"Terminal-Bench final: {num_correct}/{len(valid)} = {accuracy:.4f}")

    return EvalResult(
        benchmark="terminal_bench",
        score=accuracy,
        num_examples=len(valid),
        num_correct=num_correct,
        metrics={"terminal_bench/accuracy": accuracy},
        examples=valid,
    )
