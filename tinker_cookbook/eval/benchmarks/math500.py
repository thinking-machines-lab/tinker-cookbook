"""MATH-500 benchmark -- Hendrycks MATH test set.

Dataset: ``HuggingFaceH4/MATH-500`` on HuggingFace.
Metric: Accuracy -- fraction of problems where the extracted boxed answer matches ground truth.
Pattern: Single-turn generate + programmatic grading.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class MATH500Env(Env):
    """Single-turn env for one MATH-500 problem."""

    def __init__(self, problem: str, expected: str, renderer: Renderer):
        self.problem = problem
        self.expected = expected
        self.renderer = renderer

    async def initial_observation(self):
        prompt = self.problem + " Put your final answer in \\boxed{}."
        messages: list[Message] = [{"role": "user", "content": prompt}]
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer

        response = self.renderer.tokenizer.decode(action)
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

        ds = cast(Dataset, load_dataset("HuggingFaceH4/MATH-500", split="test"))
        if config.max_examples is not None:
            ds = ds.select(range(min(config.max_examples, len(ds))))

        envs = []
        for row in ds:
            try:
                expected = extract_boxed(row["solution"])
            except ValueError:
                continue
            envs.append(MATH500Env(row["problem"], expected, renderer))
        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(MATH500BenchmarkBuilder())
