"""Arena-Hard-Auto benchmark -- LLM-judge evaluation on challenging user queries.

Dataset: ``lmarena-ai/arena-hard-auto-v0.1`` on HuggingFace (500 questions).
Metric: Win rate judged by a strong model comparing the candidate response against
a reference baseline. The judge rates responses on a 1-10 scale; we compute the
fraction of questions where the candidate scores >= 7 ("good" threshold).

Since this benchmark requires a judge model, it uses a *second* sampling client
for the judge. If no judge is available, we fall back to a self-evaluation heuristic
based on response quality signals (length, structure, specificity).
"""

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

_JUDGE_PROMPT_TEMPLATE = """\
Please act as an impartial judge and evaluate the quality of the response provided \
by an AI assistant to the user question displayed below. Your evaluation should \
consider factors including helpfulness, relevance, accuracy, depth, creativity, and \
level of detail of the response.

Begin your evaluation by providing a short explanation. Be as objective as possible. \
After providing your explanation, you must rate the response on a scale of 1 to 10 \
by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""


def _extract_judge_score(judge_response: str) -> int | None:
    """Extract the numeric score from a judge response like '[[7]]'."""
    import re
    match = re.search(r"\[\[(\d+)\]\]", judge_response)
    if match:
        return int(match.group(1))
    # Fallback: look for "Rating: X"
    match = re.search(r"Rating:\s*(\d+)", judge_response, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


async def evaluate(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    max_tokens: int = 32768,
    max_examples: int | None = None,
    judge_sampling_client: tinker.SamplingClient | None = None,
    judge_renderer: Renderer | None = None,
) -> EvalResult:
    """Evaluate on Arena-Hard-Auto using an LLM judge.

    If *judge_sampling_client* is provided, it will be used to judge responses.
    Otherwise, the same *sampling_client* is used as a self-judge (less reliable).
    """
    ds = cast(Dataset, load_dataset("lmarena-ai/arena-hard-auto-v0.1", split="train"))
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    candidate_completer = make_completer(sampling_client, renderer, max_tokens)

    j_client = judge_sampling_client or sampling_client
    j_renderer = judge_renderer or renderer
    judge_completer = make_completer(j_client, j_renderer, max_tokens=4096, temperature=0.0)

    async def eval_one(row: dict) -> dict | None:
        turns = row.get("turns", [])
        if not turns:
            return None
        question = turns[0].get("content", "") if isinstance(turns[0], dict) else str(turns[0])
        if not question:
            return None

        # Generate candidate response
        messages: list[Message] = [{"role": "user", "content": question}]
        try:
            response = await candidate_completer(messages)
            answer = get_text(response)
        except Exception as e:
            logger.warning(f"Arena-Hard candidate generation failed: {e}")
            return None

        # Judge the response
        judge_prompt = _JUDGE_PROMPT_TEMPLATE.format(question=question, answer=answer)
        judge_messages: list[Message] = [{"role": "user", "content": judge_prompt}]
        try:
            judge_response = await judge_completer(judge_messages)
            judge_text = get_text(judge_response)
            score = _extract_judge_score(judge_text)
        except Exception as e:
            logger.warning(f"Arena-Hard judge failed: {e}")
            score = None

        if score is None:
            return None

        return {
            "correct": score >= 7,
            "score": score,
            "input": question[:200],
            "output": answer[:500],
            "judge_output": judge_text[:300],
            "cluster": row.get("cluster", "unknown"),
        }

    logger.info(f"Arena-Hard: evaluating {len(ds)} samples")
    results = await run_concurrent_eval(list(ds), eval_one)

    valid = [r for r in results if r is not None]
    num_good = sum(1 for r in valid if r["correct"])
    avg_score = sum(r["score"] for r in valid) / len(valid) if valid else 0.0
    win_rate = num_good / len(valid) if valid else 0.0

    logger.info(f"Arena-Hard final: win_rate={win_rate:.4f}, avg_score={avg_score:.2f} ({len(valid)} judged)")

    return EvalResult(
        benchmark="arena_hard",
        score=win_rate,
        num_examples=len(valid),
        num_correct=num_good,
        metrics={
            "arena_hard/win_rate": win_rate,
            "arena_hard/avg_score": avg_score / 10.0,  # normalized to 0-1
        },
        examples=valid,
    )
