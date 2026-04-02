"""Arena-Hard-Auto v0.1 benchmark -- LLM-judge evaluation on challenging user queries.

Dataset: ``lmarena-ai/arena-hard-auto-v0.1`` on HuggingFace (500 questions).
Metric: Win rate -- fraction of questions where the judge scores >= 7 out of 10.
Pattern: Single-turn generate + LLM judge grading.

Requires ``config.judge_sampling_client`` for the judge model.

Note:
    The official Arena-Hard protocol uses GPT-4.1 or Gemini-2.5 as judge.
    Using the candidate model as its own judge (self-judging) inflates scores
    due to self-preference bias. Always use a separate, stronger judge model
    for meaningful results.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import cast

import tinker
from datasets import Dataset

from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.eval.benchmarks._common import (
    decode_response,
    limit_dataset,
    load_benchmark_dataset,
    make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import (
    BenchmarkBuilder,
    BenchmarkConfig,
    BenchmarkResult,
)
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult

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
    match = re.search(r"\[\[(\d+)\]\]", judge_response)
    if match:
        return int(match.group(1))
    match = re.search(r"Rating:\s*(\d+)", judge_response, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class ArenaHardEnv(Env):
    """Single-turn env for one Arena-Hard question, graded by an LLM judge.

    The env generates a response, then uses the judge completer to score it.
    """

    def __init__(
        self,
        question: str,
        cluster: str,
        judge_completer: TinkerMessageCompleter,
        renderer: Renderer,
        example_id: str = "",
    ):
        self.question = question
        self.cluster = cluster
        self.judge_completer = judge_completer
        self.renderer = renderer
        self.example_id = example_id

    async def initial_observation(self):
        messages: list[Message] = [{"role": "user", "content": self.question}]
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        answer = decode_response(action, self.renderer)

        # Judge the response
        judge_prompt = _JUDGE_PROMPT_TEMPLATE.format(question=self.question, answer=answer)
        judge_messages: list[Message] = [{"role": "user", "content": judge_prompt}]
        try:
            from tinker_cookbook import renderers

            judge_response = await self.judge_completer(judge_messages)
            judge_text = renderers.get_text_content(judge_response)
            score = _extract_judge_score(judge_text)
        except Exception as e:
            logger.warning(f"Arena-Hard judge failed: {e}")
            score = None
            judge_text = ""

        correct = score is not None and score >= 7
        return StepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={
                "correct": float(correct),
                "score": float(score) if score is not None else 0.0,
            },
            logs={
                "example_id": self.example_id,
                "input": self.question[:200],
                "output": answer[:500],
                "judge_output": judge_text[:300],
                "score": score if score is not None else -1,
                "cluster": self.cluster,
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class ArenaHardBenchmarkBuilder(BenchmarkBuilder):
    """Arena-Hard-Auto: challenging open-ended questions judged by LLM (500 questions)."""

    name = "arena_hard"
    requires_judge = True

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = cast(Dataset, load_benchmark_dataset("lmarena-ai/arena-hard-auto-v0.1", split="train"))
        ds = limit_dataset(ds, config.max_examples)

        # Build judge completer
        j_client = config.judge_sampling_client
        j_renderer = config.judge_renderer or renderer
        if j_client is None:
            raise ValueError(
                "arena_hard requires config.judge_sampling_client for LLM-as-judge grading. "
                "Create a sampling client for the judge model and pass it via BenchmarkConfig."
            )

        judge_completer = TinkerMessageCompleter(
            sampling_client=j_client,  # type: ignore[arg-type]
            renderer=j_renderer,
            max_tokens=4096,
            temperature=0.0,
        )

        envs = []
        for row in ds:
            row = dict(row)
            turns = row.get("turns", [])
            if not turns:
                continue
            question = turns[0].get("content", "") if isinstance(turns[0], dict) else str(turns[0])
            if not question:
                continue
            cluster = row.get("cluster", "unknown")
            example_id = make_example_id("arena_hard", question)
            envs.append(
                ArenaHardEnv(question, cluster, judge_completer, renderer, example_id=example_id)
            )
        return envs

    def aggregate(self, rewards: list[float], metrics_list: list[dict]) -> BenchmarkResult:
        """Aggregate with win rate and average score."""
        num_good = sum(1 for r in rewards if r > 0)
        win_rate = num_good / len(rewards) if rewards else 0.0

        scores = [m.get("score", 0.0) for m in metrics_list]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return BenchmarkResult(
            name=self.name,
            score=win_rate,
            num_examples=len(rewards),
            num_correct=num_good,
            metrics={
                "arena_hard/win_rate": win_rate,
                "arena_hard/avg_score": avg_score / 10.0,
            },
        )


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(ArenaHardBenchmarkBuilder())
