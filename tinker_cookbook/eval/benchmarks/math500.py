"""MATH-500 benchmark -- Hendrycks MATH test set.

Dataset: ``HuggingFaceH4/MATH-500`` on HuggingFace.
Metric: Accuracy -- fraction of problems where the extracted boxed answer matches ground truth.
Pattern: Single-turn ``MessageEnv`` + programmatic grading.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

from datasets import Dataset

from tinker_cookbook.eval.benchmarks._common import (
    build_messages,
    limit_dataset,
    load_benchmark_dataset,
    make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import (
    BenchmarkBuilder,
    BenchmarkConfig,
)
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import Env

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MessageEnv implementation
# ---------------------------------------------------------------------------


class MATH500MessageEnv(MessageEnv):
    """Single-turn message env for one MATH-500 problem.

    Receives a parsed ``Message`` (thinking already stripped by
    ``EnvFromMessageEnv``), grades the answer, and returns the result.
    """

    def __init__(
        self,
        problem: str,
        expected: str,
        example_id: str = "",
        system_prompt: str | None = None,
    ):
        self.problem = problem
        self.expected = expected
        self.example_id = example_id
        self.system_prompt = system_prompt

    async def initial_observation(self) -> list[Message]:
        prompt = self.problem + " Put your final answer in \\boxed{}."
        return build_messages(prompt, self.system_prompt)

    async def step(self, message: Message) -> MessageStepResult:
        from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer

        response = get_text_content(message)
        try:
            given = extract_boxed(response)
            correct = grade_answer(given, self.expected)
        except ValueError:
            given = ""
            correct = False
        return MessageStepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_messages=[],
            metrics={"correct": float(correct)},
            logs={
                "example_id": self.example_id,
                "input": self.problem[:200],
                "expected": self.expected,
                "extracted": str(given)[:200],
                "output": response[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class MATH500BenchmarkBuilder(BenchmarkBuilder):
    """MATH-500: 500 competition math problems from the Hendrycks MATH dataset."""

    name = "math500"
    recommended_system_prompt = "Put your final answer in \\boxed{}."

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

        ds = cast(Dataset, load_benchmark_dataset("HuggingFaceH4/MATH-500"))
        ds = limit_dataset(ds, config.max_examples)

        envs: list[Env] = []
        for row in ds:
            row = dict(row)
            try:
                expected = extract_boxed(row["solution"])
            except ValueError:
                continue
            example_id = make_example_id("math500", row["problem"])
            msg_env = MATH500MessageEnv(
                row["problem"],
                expected,
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


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(MATH500BenchmarkBuilder())
