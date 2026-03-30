"""IFEval benchmark -- instruction-following verification.

Dataset: Loaded from local JSONL (Nemotron-Cascade RL format).
Metric: Strict accuracy (fraction of prompts where ALL constraints are satisfied).
Pattern: Single-turn generate + programmatic grading (multi-verifier).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence

import tinker

from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig, BenchmarkResult
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult

logger = logging.getLogger(__name__)


def _parse_kwargs(raw_kwargs: list, instruction_ids: list) -> list[dict]:
    kwargs_list = []
    for kw in raw_kwargs:
        if kw is None:
            kwargs_list.append({})
        elif isinstance(kw, str):
            try:
                kwargs_list.append(json.loads(kw))
            except json.JSONDecodeError:
                kwargs_list.append({})
        elif isinstance(kw, dict):
            kwargs_list.append(kw)
        else:
            kwargs_list.append({})
    while len(kwargs_list) < len(instruction_ids):
        kwargs_list.append({})
    return kwargs_list


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
    ):
        self.messages = messages
        self.instruction_ids = instruction_ids
        self.kwargs_list = kwargs_list
        self.renderer = renderer

    async def initial_observation(self):
        model_input = self.renderer.build_generation_prompt(self.messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        from tinker_cookbook.recipes.nemotron_cascade.rl.envs.if_rl import verify_all_instructions

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
        ifeval_path = os.path.expanduser("~/data/nemotron-cascade-2/rl_if_rl.jsonl")
        rows: list[dict] = []
        with open(ifeval_path) as f:
            for line in f:
                rows.append(json.loads(line))

        if config.max_examples is not None:
            rows = rows[:config.max_examples]

        envs = []
        for row in rows:
            prompt_messages = row["responses_create_params"]["input"]
            instruction_ids = row.get("instruction_id_list", [])
            raw_kwargs = row.get("kwargs", [])
            kwargs_list = _parse_kwargs(raw_kwargs, instruction_ids)

            messages: list[Message] = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in prompt_messages
            ]
            envs.append(IFEvalEnv(messages, instruction_ids, kwargs_list, renderer))
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
