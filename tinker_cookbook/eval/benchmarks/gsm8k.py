"""GSM8K benchmark — grade-school math word problems.

Dataset: ``openai/gsm8k`` (main split, test).
Metric: Accuracy — fraction of problems where the extracted numeric answer matches the ground truth.
Pattern: Single-turn generate + programmatic grading.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence

import tinker

from tinker_cookbook.eval.benchmarks._common import (
    check_gsm8k,
    extract_boxed,
    extract_gsm8k_answer,
    extract_number,
    load_benchmark_dataset,
)
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult

logger = logging.getLogger(__name__)

# Re-export for backward compatibility (used by benchmark_test.py and others)
__all__ = ["extract_boxed", "extract_number", "extract_gsm8k_answer", "check_gsm8k"]


# ---------------------------------------------------------------------------
# Env implementation
# ---------------------------------------------------------------------------


class GSM8KEnv(Env):
    """Single-turn env for one GSM8K problem."""

    def __init__(self, question: str, expected: str, renderer: Renderer, example_id: str = ""):
        self.question = question
        self.expected = expected
        self.renderer = renderer
        self.example_id = example_id

    async def initial_observation(self):
        messages: list[Message] = [{"role": "user", "content": self.question}]
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        response = self.renderer.tokenizer.decode(action)
        correct = check_gsm8k(response, self.expected)
        return StepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={"correct": float(correct)},
            logs={
                "example_id": self.example_id,
                "input": self.question[:200],
                "expected": self.expected,
                "extracted": extract_gsm8k_answer(response),
                "output": response[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class GSM8KBenchmarkBuilder(BenchmarkBuilder):
    """GSM8K: 1,319 grade-school math word problems."""

    name = "gsm8k"

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = load_benchmark_dataset("openai/gsm8k", name="main")
        if config.max_examples is not None:
            ds = ds.select(range(min(config.max_examples, len(ds))))

        envs = []
        for i, row in enumerate(ds):
            expected = row["answer"].split("####")[-1].strip()
            example_id = f"gsm8k_{hashlib.md5(row['question'].encode()).hexdigest()[:12]}"
            envs.append(GSM8KEnv(row["question"], expected, renderer, example_id=example_id))
        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(GSM8KBenchmarkBuilder())
