"""IFBench benchmark -- instruction following with objective verifiers.

Dataset: ``allenai/IFBench_test`` on HuggingFace (300 examples).
Metric: Strict accuracy (fraction of prompts where ALL constraints are satisfied).
Pattern: Single-turn generate + programmatic grading.

IFBench extends IFEval with 58 new diverse constraint types and provides
objective verification functions.
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
    parse_kwargs,
)
from tinker_cookbook.eval.benchmarks._types import (
    BenchmarkBuilder,
    BenchmarkConfig,
    BenchmarkResult,
    Metrics,
)
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult

logger = logging.getLogger(__name__)


try:
    from instructions_registry import (  # pyright: ignore[reportMissingImports]
        INSTRUCTION_DICT as _IFBENCH_CHECKERS,
    )
except ImportError:
    _IFBENCH_CHECKERS = None


def _verify_constraints(
    response: str,
    instruction_ids: list[str],
    kwargs_list: list[dict],
    prompt: str = "",
) -> tuple[float, bool]:
    """Verify IFBench constraints using the official IFBench checkers.

    Requires the ``ifbench`` package::

        uv pip install 'ifbench @ git+https://github.com/allenai/IFBench.git'

    Raises:
        ImportError: If the ``ifbench`` package is not installed.
    """
    if _IFBENCH_CHECKERS is None:
        raise ImportError(
            "IFBench requires the official checker library. Install it with:\n"
            "  uv pip install 'ifbench @ git+https://github.com/allenai/IFBench.git'"
        )

    results = []
    for iid, kw in zip(instruction_ids, kwargs_list):
        if iid not in _IFBENCH_CHECKERS:
            results.append(False)
            continue
        checker = _IFBENCH_CHECKERS[iid](iid)
        # Filter None values from kwargs
        clean_kw = {k: v for k, v in kw.items() if v is not None}
        checker.build_description(**clean_kw)
        # Some checkers need the prompt for context
        args = checker.get_instruction_args()
        if args and "prompt" in args:
            checker.build_description(prompt=prompt)
        try:
            results.append(bool(checker.check_following(response)))
        except Exception:
            results.append(False)

    if not results:
        return 0.0, False
    fraction = sum(results) / len(results)
    return fraction, all(results)


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class IFBenchEnv(Env):
    """Single-turn env for one IFBench prompt with constraint verification."""

    def __init__(
        self,
        prompt: str,
        instruction_ids: list[str],
        kwargs_list: list[dict],
        renderer: Renderer,
        example_id: str = "",
    ):
        self.prompt = prompt
        self.instruction_ids = instruction_ids
        self.kwargs_list = kwargs_list
        self.renderer = renderer
        self.example_id = example_id

    async def initial_observation(self):
        messages: list[Message] = [{"role": "user", "content": self.prompt}]
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        response = decode_response(action, self.renderer)
        fraction, all_satisfied = _verify_constraints(
            response, self.instruction_ids, self.kwargs_list, prompt=self.prompt
        )
        return StepResult(
            reward=1.0 if all_satisfied else 0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={"correct": float(all_satisfied), "fraction": fraction},
            logs={
                "example_id": self.example_id,
                "input": self.prompt[:200],
                "fraction": fraction,
                "output": response[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class IFBenchBenchmarkBuilder(BenchmarkBuilder):
    """IFBench: instruction following with 58 diverse constraint types (300 examples)."""

    name = "ifbench"

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        try:
            ds = cast(Dataset, load_benchmark_dataset("allenai/IFBench_test", split="train"))
        except Exception as exc:
            logger.warning(f"Could not load IFBench dataset: {exc}.")
            return []

        ds = limit_dataset(ds, config.max_examples)

        envs = []
        for row in ds:
            row = dict(row)
            prompt = row.get("prompt", "")
            instruction_ids = row.get("instruction_id_list", [])
            raw_kwargs = row.get("kwargs", [])
            if not prompt:
                continue

            kwargs_list = parse_kwargs(raw_kwargs, instruction_ids)

            example_id = make_example_id("ifbench", prompt)
            envs.append(
                IFBenchEnv(prompt, instruction_ids, kwargs_list, renderer, example_id=example_id)
            )
        return envs

    def aggregate(self, rewards: list[float], metrics_list: list[Metrics]) -> BenchmarkResult:
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
                "ifbench/strict_accuracy": strict_acc,
                "ifbench/loose_accuracy": loose_acc,
            },
        )


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(IFBenchBenchmarkBuilder())
