"""MATH-500 benchmark -- Hendrycks MATH test set.

Dataset: ``HuggingFaceH4/MATH-500`` on HuggingFace.
Metric: Accuracy -- fraction of problems where the extracted boxed answer matches ground truth.
Pattern: Single-turn generate + programmatic grading.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

import tinker
from datasets import Dataset

from tinker_cookbook.eval.benchmarks._common import (
    decode_response,
    limit_dataset,
    load_benchmark_dataset,
    make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class MATH500Env(Env):
    """Single-turn env for one MATH-500 problem."""

    def __init__(self, problem: str, expected: str, renderer: Renderer, example_id: str = ""):
        self.problem = problem
        self.expected = expected
        self.renderer = renderer
        self.example_id = example_id

    async def initial_observation(self):
        prompt = self.problem + " Put your final answer in \\boxed{}."
        messages: list[Message] = [{"role": "user", "content": prompt}]
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer

        response = decode_response(action, self.renderer)
        try:
            given = extract_boxed(response)
            correct = grade_answer(given, self.expected)
        except ValueError:
            given = ""
            correct = False
        return StepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
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

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

        ds = cast(Dataset, load_benchmark_dataset("HuggingFaceH4/MATH-500"))
        ds = limit_dataset(ds, config.max_examples)

        envs = []
        for row in ds:
            row = dict(row)
            try:
                expected = extract_boxed(row["solution"])
            except ValueError:
                continue
            example_id = make_example_id("math500", row["problem"])
            envs.append(MATH500Env(row["problem"], expected, renderer, example_id=example_id))
        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(MATH500BenchmarkBuilder())
