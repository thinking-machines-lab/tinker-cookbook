"""MMLU-Pro benchmark -- multi-task language understanding (professional).

Dataset: ``TIGER-Lab/MMLU-Pro`` on HuggingFace.
Metric: Multiple-choice accuracy (A-J, up to 10 options).
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

# Re-export for backward compatibility (used by benchmark_test.py)
__all__ = ["extract_mcq_answer"]


# ---------------------------------------------------------------------------
# MessageEnv implementation
# ---------------------------------------------------------------------------


class MMLUProMessageEnv(MessageEnv):
    """Single-turn message env for one MMLU-Pro question.

    Receives a parsed ``Message`` (thinking already stripped by
    ``EnvFromMessageEnv``), grades the answer, and returns the result.
    """

    def __init__(
        self,
        prompt: str,
        expected: str,
        valid_letters: str,
        example_id: str = "",
        system_prompt: str | None = None,
    ):
        self.prompt = prompt
        self.expected = expected
        self.valid_letters = valid_letters
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
                "output": response[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class MMLUProBenchmarkBuilder(BenchmarkBuilder):
    """MMLU-Pro: professional-level multiple-choice questions (up to 10 options)."""

    name = "mmlu_pro"

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = cast(Dataset, load_benchmark_dataset("TIGER-Lab/MMLU-Pro"))

        ds = limit_dataset(ds, config.max_examples, shuffle_seed=42)

        valid_letters = "ABCDEFGHIJ"
        envs: list[Env] = []
        for row in ds:
            row = dict(row)
            question = row.get("question", row.get("input", ""))
            choices = row.get("options", row.get("choices", []))
            answer_idx = row.get("answer_index", row.get("answer"))

            if choices:
                prompt = f"{question}\n\n{format_mcq_choices(choices)}\n\nAnswer with just the letter (A, B, C, D, etc.)."
            else:
                prompt = f"{question}\n\nAnswer with just the letter."

            if isinstance(answer_idx, int):
                expected = chr(65 + answer_idx)
            elif isinstance(answer_idx, str) and len(answer_idx) == 1:
                expected = answer_idx.upper()
            else:
                expected = str(answer_idx).strip().upper()

            question_id = row.get("question_id")
            if question_id is not None:
                example_id = f"mmlu_pro_{question_id}"
            else:
                example_id = make_example_id("mmlu_pro", question)
            msg_env = MMLUProMessageEnv(
                prompt,
                expected,
                valid_letters,
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

register(MMLUProBenchmarkBuilder())
