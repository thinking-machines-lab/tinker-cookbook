"""IFEval benchmark -- instruction-following verification.

Dataset: ``google/IFEval`` on HuggingFace (541 examples, split="train").
Metric: Strict accuracy (fraction of prompts where ALL constraints are satisfied).
Pattern: Single-turn generate + programmatic grading (multi-verifier).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import tinker

from tinker_cookbook.eval.benchmarks._common import (
    limit_dataset,
    load_benchmark_dataset,
    parse_kwargs,
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


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class IFEvalEnv(Env):
    """Single-turn env for one IFEval prompt with multiple instruction constraints."""

    def __init__(
        self,
        messages: list[Message],
        instruction_ids: list[str],
        kwargs_list: list[dict],
        renderer: Renderer,
        example_id: str = "",
    ):
        self.messages = messages
        self.instruction_ids = instruction_ids
        self.kwargs_list = kwargs_list
        self.renderer = renderer
        self.example_id = example_id

    async def initial_observation(self):
        model_input = self.renderer.build_generation_prompt(self.messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        from tinker_cookbook.eval.benchmarks._ifeval_verify import verify_all_instructions

        response = self.renderer.tokenizer.decode(action)
        fraction, _ = verify_all_instructions(response, self.instruction_ids, self.kwargs_list)
        correct = fraction == 1.0
        return StepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={"correct": float(correct), "fraction": fraction},
            logs={
                "example_id": self.example_id,
                "input": self.messages[-1]["content"][:200] if self.messages else "",
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

        envs = []
        for row in ds:
            instruction_ids = row["instruction_id_list"]
            raw_kwargs = row["kwargs"]
            kwargs_list = parse_kwargs(raw_kwargs, instruction_ids)

            messages: list[Message] = [{"role": "user", "content": row["prompt"]}]
            example_id = f"ifeval_{row['key']}"
            envs.append(
                IFEvalEnv(messages, instruction_ids, kwargs_list, renderer, example_id=example_id)
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
