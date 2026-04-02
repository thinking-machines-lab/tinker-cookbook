"""GSM8K benchmark — grade-school math word problems.

Dataset: ``openai/gsm8k`` (main split, test).
Metric: Accuracy — fraction of problems where the extracted numeric answer matches the ground truth.
Pattern: Single-turn ``MessageEnv`` + programmatic grading.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from tinker_cookbook.eval.benchmarks._common import (
    build_messages,
    check_gsm8k,
    extract_boxed,
    extract_gsm8k_answer,
    extract_number,
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

# Re-export for backward compatibility (used by benchmark_test.py and others)
__all__ = ["extract_boxed", "extract_number", "extract_gsm8k_answer", "check_gsm8k"]


# ---------------------------------------------------------------------------
# MessageEnv implementation
# ---------------------------------------------------------------------------


class GSM8KMessageEnv(MessageEnv):
    """Single-turn message env for one GSM8K problem.

    Receives a parsed ``Message`` (thinking already stripped by
    ``EnvFromMessageEnv``), grades the answer, and returns the result.
    """

    def __init__(
        self,
        question: str,
        expected: str,
        example_id: str = "",
        system_prompt: str | None = None,
    ):
        self.question = question
        self.expected = expected
        self.example_id = example_id
        self.system_prompt = system_prompt

    async def initial_observation(self) -> list[Message]:
        return build_messages(self.question, self.system_prompt)

    async def step(self, message: Message) -> MessageStepResult:
        response = get_text_content(message)
        correct = check_gsm8k(response, self.expected)
        return MessageStepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_messages=[],
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
        ds = limit_dataset(ds, config.max_examples)

        envs: list[Env] = []
        for row in ds:
            row = dict(row)
            expected = row["answer"].split("####")[-1].strip()
            example_id = make_example_id("gsm8k", row["question"])
            msg_env = GSM8KMessageEnv(
                row["question"],
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

register(GSM8KBenchmarkBuilder())
