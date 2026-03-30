"""MBPP benchmark -- Mostly Basic Python Programming via code execution.

Dataset: ``google-research-datasets/mbpp`` (sanitized) on HuggingFace.
Metric: Pass@1 -- fraction of problems where generated code passes assertion tests.
Pattern: Single-turn generate + subprocess execution.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from collections.abc import Sequence
from typing import cast

import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Code extraction and execution
# ---------------------------------------------------------------------------


def _extract_python_code(text: str) -> str:
    """Extract Python code from a model response."""
    match = re.search(r"```python\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    matches = re.findall(r"```(?:\w*)\s*\n(.*?)\n```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


def _run_python_tests(code: str, test_assertions: list[str], timeout: int = 15) -> bool:
    """Run code + assertion tests in a subprocess. Returns True if all pass."""
    test_code = code + "\n\n" + "\n".join(test_assertions)
    try:
        result = subprocess.run(
            ["python3", "-c", test_code],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class MBPPEnv(Env):
    """Single-turn env for one MBPP problem with execution-based grading."""

    def __init__(
        self,
        prompt: str,
        task_prompt: str,
        test_list: list[str],
        renderer: Renderer,
    ):
        self.prompt = prompt
        self.task_prompt = task_prompt
        self.test_list = test_list
        self.renderer = renderer

    async def initial_observation(self):
        messages: list[Message] = [{"role": "user", "content": self.prompt}]
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        response = self.renderer.tokenizer.decode(action)
        code = _extract_python_code(response)

        loop = asyncio.get_event_loop()
        passed = await loop.run_in_executor(None, _run_python_tests, code, self.test_list)

        return StepResult(
            reward=1.0 if passed else 0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={"correct": float(passed)},
            logs={
                "input": self.task_prompt[:200],
                "output": response[:500],
                "code": code[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class MBPPBenchmarkBuilder(BenchmarkBuilder):
    """MBPP: Mostly Basic Python Programming with execution-based testing."""

    name = "mbpp"

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        try:
            ds = cast(Dataset, load_dataset("google-research-datasets/mbpp", "sanitized", split="test"))
        except Exception:
            ds = cast(Dataset, load_dataset("google-research-datasets/mbpp", "full", split="test"))

        if config.max_examples is not None:
            ds = ds.select(range(min(config.max_examples, len(ds))))

        envs = []
        for row in ds:
            task_prompt = row.get("prompt", row.get("text", ""))
            test_list = row.get("test_list", [])
            if not task_prompt or not test_list:
                continue

            example_test = test_list[0] if test_list else ""
            prompt = (
                f"{task_prompt}\n\n"
                f"Example test: `{example_test}`\n\n"
                "Write a Python function that satisfies the requirements. "
                "Provide ONLY the function definition in a ```python code block."
            )
            envs.append(MBPPEnv(prompt, task_prompt, test_list, renderer))
        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(MBPPBenchmarkBuilder())
