"""LiveCodeBench benchmark -- competitive programming with execution-based testing.

Dataset: ``livecodebench/code_generation_lite`` on HuggingFace.
Metric: Pass@1 -- fraction of problems where generated code passes all test cases.
Pattern: Single-turn generate + subprocess execution (stdin/stdout).

LiveCodeBench continuously collects problems from LeetCode, AtCoder, and Codeforces.
Each problem has a description, starter code (optional), and JSON-encoded test cases
(stdin/stdout pairs).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
from collections.abc import Sequence
from typing import cast

import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig, BenchmarkResult
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


def _run_with_stdin(code: str, stdin: str, timeout: int = 30) -> tuple[bool, str]:
    """Run *code* with *stdin* and return (success, stdout)."""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            input=stdin,
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        return result.returncode == 0, result.stdout.strip()
    except (subprocess.TimeoutExpired, Exception):
        return False, ""


def _check_solution(code: str, test_cases_json: str, timeout: int = 30) -> bool:
    """Run the solution against all test cases encoded as JSON."""
    try:
        tests = json.loads(test_cases_json)
    except (json.JSONDecodeError, TypeError):
        return False

    inputs = tests.get("inputs", tests.get("input", []))
    outputs = tests.get("outputs", tests.get("output", []))

    if not inputs or not outputs or len(inputs) != len(outputs):
        return False

    for inp, expected_out in zip(inputs, outputs):
        stdin_str = inp if isinstance(inp, str) else str(inp)
        expected_str = expected_out.strip() if isinstance(expected_out, str) else str(expected_out).strip()

        ok, actual = _run_with_stdin(code, stdin_str, timeout=timeout)
        if not ok or actual.strip() != expected_str:
            return False

    return True


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class LiveCodeBenchEnv(Env):
    """Single-turn env for one LiveCodeBench problem with execution-based grading."""

    def __init__(
        self,
        prompt: str,
        question: str,
        test_cases_json: str,
        difficulty: str,
        renderer: Renderer,
    ):
        self.prompt = prompt
        self.question = question
        self.test_cases_json = test_cases_json
        self.difficulty = difficulty
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
        passed = await loop.run_in_executor(None, _check_solution, code, self.test_cases_json)

        return StepResult(
            reward=1.0 if passed else 0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={"correct": float(passed), "difficulty": self.difficulty},
            logs={
                "input": self.question[:200],
                "output": response[:500],
                "code": code[:500],
                "difficulty": self.difficulty,
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class LiveCodeBenchBenchmarkBuilder(BenchmarkBuilder):
    """LiveCodeBench: competitive programming with execution-based Pass@1."""

    name = "livecodebench"

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        try:
            ds = cast(
                Dataset,
                load_dataset("livecodebench/code_generation_lite", split="test", trust_remote_code=True),
            )
        except Exception:
            try:
                ds = cast(
                    Dataset,
                    load_dataset(
                        "livecodebench/code_generation_lite",
                        "release_v6",
                        split="test",
                        trust_remote_code=True,
                    ),
                )
            except Exception as exc:
                logger.warning(f"Could not load LiveCodeBench dataset: {exc}.")
                return []

        if config.max_examples is not None:
            ds = ds.select(range(min(config.max_examples, len(ds))))

        envs = []
        for row in ds:
            question = row.get("question_content", row.get("question", ""))
            starter_code = row.get("starter_code", "")
            test_cases_json = row.get("input_output", "")
            difficulty = row.get("difficulty", "unknown")

            if not question or not test_cases_json:
                continue

            prompt_parts = [question]
            if starter_code:
                prompt_parts.append(f"\nStarter code:\n```python\n{starter_code}\n```")
            prompt_parts.append(
                "\nWrite a complete Python solution. Read input from stdin and write output to stdout. "
                "Provide your solution in a ```python code block."
            )
            prompt = "\n".join(prompt_parts)

            envs.append(LiveCodeBenchEnv(prompt, question, test_cases_json, difficulty, renderer))
        return envs

    def aggregate(self, rewards: list[float], metrics_list: list[dict]) -> BenchmarkResult:
        """Aggregate with per-difficulty breakdown."""
        num_correct = sum(1 for r in rewards if r > 0)
        accuracy = num_correct / len(rewards) if rewards else 0.0

        difficulty_results: dict[str, list[bool]] = {}
        for r, m in zip(rewards, metrics_list):
            d = m.get("difficulty", "unknown")
            if isinstance(d, str):
                difficulty_results.setdefault(d, []).append(r > 0)

        metrics: dict[str, float] = {"livecodebench/pass_at_1": accuracy}
        for d, d_results in sorted(difficulty_results.items()):
            metrics[f"livecodebench/{d}/pass_at_1"] = sum(d_results) / len(d_results) if d_results else 0.0

        return BenchmarkResult(
            name=self.name,
            score=accuracy,
            num_examples=len(rewards),
            num_correct=num_correct,
            metrics=metrics,
        )


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(LiveCodeBenchBenchmarkBuilder())
