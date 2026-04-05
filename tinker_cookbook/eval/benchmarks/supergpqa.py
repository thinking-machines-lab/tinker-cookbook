"""SuperGPQA benchmark -- graduate-level multi-domain QA (285 disciplines).

Dataset: ``m-a-p/SuperGPQA`` on HuggingFace (26,529 examples, 4-10 option MCQA).
Metric: Multiple-choice accuracy (A-J).
Pattern: Single-turn ``MessageEnv`` + programmatic grading.

Reference: https://arxiv.org/abs/2502.14739
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

from datasets import Dataset

from tinker_cookbook.eval.benchmarks._common import (
    build_messages,
    extract_mcq_answer,
    format_mcq_choices,
    limit_dataset,
    load_benchmark_dataset,
    make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import Env

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MessageEnv implementation
# ---------------------------------------------------------------------------


class SuperGPQAMessageEnv(MessageEnv):
    """Single-turn message env for one SuperGPQA question."""

    def __init__(
        self,
        prompt: str,
        expected: str,
        valid_letters: str = "ABCD",
        discipline: str = "",
        example_id: str = "",
        system_prompt: str | None = None,
    ):
        self.prompt = prompt
        self.expected = expected
        self.valid_letters = valid_letters
        self.discipline = discipline
        self.example_id = example_id
        self.system_prompt = system_prompt

    async def initial_observation(self) -> list[Message]:
        return build_messages(self.prompt, self.system_prompt)

    async def step(self, message: Message) -> MessageStepResult:
        response = get_text_content(message)
        extracted = extract_mcq_answer(response, self.valid_letters)
        correct = extracted == self.expected
        return MessageStepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_messages=[],
            metrics={"correct": float(correct)},
            logs={
                "example_id": self.example_id,
                "input": self.prompt[:200],
                "expected": self.expected,
                "extracted": extracted,
                "discipline": self.discipline,
                "output": response[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class SuperGPQABenchmarkBuilder(BenchmarkBuilder):
    """SuperGPQA: graduate-level QA across 285 disciplines (26K+ questions)."""

    name = "supergpqa"

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = cast(Dataset, load_benchmark_dataset("m-a-p/SuperGPQA", split="train"))
        ds = limit_dataset(ds, config.max_examples, shuffle_seed=42)

        envs: list[Env] = []
        for row in ds:
            row = dict(row)
            question = row.get("question", "")
            options = row.get("options", [])
            answer_letter = row.get("answer_letter", "")
            if not question or not options or not answer_letter:
                continue

            # Options can be 4-10 items
            valid_letters = "".join(chr(65 + i) for i in range(len(options)))
            prompt = (
                f"{question}\n\n{format_mcq_choices(options)}\n\n"
                f"Think step by step, then give your final answer as a single letter "
                f"({', '.join(valid_letters)})."
            )

            discipline = row.get("discipline", "")
            example_id = row.get("uuid", "") or make_example_id("supergpqa", question)
            msg_env = SuperGPQAMessageEnv(
                prompt,
                answer_letter.strip().upper(),
                valid_letters=valid_letters,
                discipline=discipline,
                example_id=example_id,
                system_prompt=config.system_prompt,
            )
            envs.append(
                EnvFromMessageEnv(
                    renderer=renderer,
                    message_env=msg_env,
                    failed_parse_reward=0.0,
                    context_overflow_reward=0.0,
                )
            )
        return envs

    # Per-discipline breakdown is available via load_trajectories() + logs["discipline"].
    # Default aggregate (accuracy) is sufficient for the top-level score.


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(SuperGPQABenchmarkBuilder())
