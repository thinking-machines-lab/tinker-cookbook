"""IFEval benchmark -- instruction-following verification.

Dataset: ``google/IFEval`` on HuggingFace (541 examples, split="train").
Metric: Strict accuracy (fraction of prompts where ALL constraints are satisfied).
Pattern: Single-turn ``MessageEnv`` + programmatic grading (multi-verifier).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from tinker_cookbook.eval.benchmarks._common import (
    build_messages,
    limit_dataset,
    load_benchmark_dataset,
    parse_kwargs,
)
from tinker_cookbook.eval.benchmarks._types import (
    BenchmarkBuilder,
    BenchmarkConfig,
    BenchmarkResult,
)
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import Env

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MessageEnv implementation
# ---------------------------------------------------------------------------


class IFEvalMessageEnv(MessageEnv):
    """Single-turn message env for one IFEval prompt with multiple instruction constraints.

    Receives a parsed ``Message`` (thinking already stripped by
    ``EnvFromMessageEnv``), verifies all constraints, and returns the result.
    """

    def __init__(
        self,
        messages: list[Message],
        instruction_ids: list[str],
        kwargs_list: list[dict],
        example_id: str = "",
    ):
        self._messages = messages
        self.instruction_ids = instruction_ids
        self.kwargs_list = kwargs_list
        self.example_id = example_id

    async def initial_observation(self) -> list[Message]:
        return self._messages

    async def step(self, message: Message) -> MessageStepResult:
        from tinker_cookbook.eval.benchmarks._ifeval_verify import verify_all_instructions

        response = get_text_content(message)
        fraction, _ = verify_all_instructions(response, self.instruction_ids, self.kwargs_list)
        correct = fraction == 1.0
        return MessageStepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_messages=[],
            metrics={"correct": float(correct), "fraction": fraction},
            logs={
                "example_id": self.example_id,
                "input": str(self._messages[-1]["content"])[:200] if self._messages else "",
                "fraction": fraction,
                "output": response[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class IFEvalBenchmarkBuilder(BenchmarkBuilder):
    """IFEval: instruction-following evaluation with multi-constraint verification."""

    name = "ifeval"

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = load_benchmark_dataset("google/IFEval", split="train")
        ds = limit_dataset(ds, config.max_examples)

        envs: list[Env] = []
        for row in ds:
            row = dict(row)
            instruction_ids = row["instruction_id_list"]
            raw_kwargs = row["kwargs"]
            kwargs_list = parse_kwargs(raw_kwargs, instruction_ids)

            messages = build_messages(row["prompt"], config.system_prompt)
            example_id = f"ifeval_{row['key']}"
            msg_env = IFEvalMessageEnv(
                messages,
                instruction_ids,
                kwargs_list,
                example_id=example_id,
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

    def aggregate(self, rewards: list[float], metrics_list: list[dict]) -> BenchmarkResult:
        """Aggregate with both strict and loose accuracy."""
        strict_correct = sum(1 for r in rewards if r > 0)
        strict_acc = strict_correct / len(rewards) if rewards else 0.0

        total_fraction = sum(m.get("fraction", 0.0) for m in metrics_list)
        loose_acc = total_fraction / len(metrics_list) if metrics_list else 0.0

        return BenchmarkResult(
            name=self.name,
            score=strict_acc,
            num_examples=len(rewards),
            num_correct=strict_correct,
            metrics={
                "ifeval/strict_accuracy": strict_acc,
                "ifeval/loose_accuracy": loose_acc,
            },
        )


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(IFEvalBenchmarkBuilder())
