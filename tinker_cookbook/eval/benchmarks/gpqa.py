"""GPQA-Diamond benchmark -- graduate-level science QA (multiple choice).

Dataset: ``Idavidrein/gpqa`` (gpqa_diamond config) on HuggingFace.
Metric: Multiple-choice accuracy (A/B/C/D).
Pattern: Single-turn ``MessageEnv`` + programmatic grading.
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


class GPQAMessageEnv(MessageEnv):
    """Single-turn message env for one GPQA-Diamond question.

    Receives a parsed ``Message`` (thinking already stripped by
    ``EnvFromMessageEnv``), grades the answer, and returns the result.
    """

    def __init__(
        self,
        prompt: str,
        expected: str,
        example_id: str = "",
        system_prompt: str | None = None,
    ):
        self.prompt = prompt
        self.expected = expected
        self.example_id = example_id
        self.system_prompt = system_prompt

    async def initial_observation(self) -> list[Message]:
        return build_messages(self.prompt, self.system_prompt)

    async def step(self, message: Message) -> MessageStepResult:
        response = get_text_content(message)
        extracted = extract_mcq_answer(response)
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
                "output": response[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class GPQABenchmarkBuilder(BenchmarkBuilder):
    """GPQA-Diamond: graduate-level science multiple-choice (198 questions)."""

    name = "gpqa"

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = cast(
            Dataset, load_benchmark_dataset("Idavidrein/gpqa", name="gpqa_diamond", split="train")
        )
        ds = limit_dataset(ds, config.max_examples)

        envs: list[Env] = []
        for row in ds:
            row = dict(row)
            question = row["Question"]
            correct_answer = row.get("Answer", row.get("Correct Answer", ""))

            choice_cols = [
                col
                for col in row
                if col.startswith("Choice")
                or col in ("choice_a", "choice_b", "choice_c", "choice_d")
            ]

            if choice_cols:
                choices = [row[c] for c in sorted(choice_cols) if row.get(c)]
            else:
                choices = [row.get("Correct Answer", "")]
                for i in range(1, 4):
                    inc = row.get(f"Incorrect Answer {i}", "")
                    if inc:
                        choices.append(inc)

            if not choices:
                continue

            if correct_answer in ("A", "B", "C", "D"):
                expected = correct_answer
            else:
                expected = "A"
                for i, c in enumerate(choices):
                    if c.strip() == str(correct_answer).strip():
                        expected = chr(65 + i)
                        break

            prompt = (
                f"{question}\n\n{format_mcq_choices(choices)}\n\n"
                "Think step by step, then give your final answer as a single letter (A, B, C, or D)."
            )
            example_id = make_example_id("gpqa", question)
            msg_env = GPQAMessageEnv(
                prompt,
                expected,
                example_id=example_id,
                system_prompt=config.system_prompt,
            )
            envs.append(
                EnvFromMessageEnv(
                    renderer=renderer,
                    message_env=msg_env,
                    failed_parse_reward=0.0,
                )
            )
        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(GPQABenchmarkBuilder())
