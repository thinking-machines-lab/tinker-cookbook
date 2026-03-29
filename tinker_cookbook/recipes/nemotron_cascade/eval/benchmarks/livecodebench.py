"""LiveCodeBench benchmark -- competitive programming with execution-based testing.

Dataset: ``livecodebench/code_generation_lite`` on HuggingFace.
Metric: Pass@1 -- fraction of problems where the generated code passes all test cases.

LiveCodeBench continuously collects problems from LeetCode, AtCoder, and Codeforces.
Each problem has a description, starter code (optional), and JSON-encoded test cases
(stdin/stdout pairs). We generate a solution, then verify it by running against the
test cases in a subprocess.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
from typing import cast

import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook.renderers import Message, Renderer
from tinker_cookbook.recipes.nemotron_cascade.eval.base import EvalResult
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import (
    get_text,
    make_completer,
    run_concurrent_eval,
)

logger = logging.getLogger(__name__)


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
    """Run the solution against all test cases encoded as JSON.

    The ``input_output`` field in LiveCodeBench is a JSON string containing
    a dict with ``inputs`` and ``outputs`` lists (stdin/stdout pairs).
    """
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


async def evaluate(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    max_tokens: int = 32768,
    max_examples: int | None = None,
) -> EvalResult:
    """Evaluate on LiveCodeBench (Pass@1 via code execution)."""
    try:
        ds = cast(
            Dataset,
            load_dataset("livecodebench/code_generation_lite", split="test", trust_remote_code=True),
        )
    except Exception:
        # Try the release_v6 config
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
            logger.warning(f"Could not load LiveCodeBench dataset: {exc}. Skipping.")
            return EvalResult(benchmark="livecodebench", score=0.0, num_examples=0, num_correct=0)

    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    completer = make_completer(sampling_client, renderer, max_tokens)

    async def eval_one(row: dict) -> dict | None:
        # Field names: question_title, question_content, platform, difficulty,
        # starter_code, input_output (JSON string of test cases)
        question = row.get("question_content", row.get("question", ""))
        starter_code = row.get("starter_code", "")
        test_cases_json = row.get("input_output", "")

        if not question or not test_cases_json:
            return None

        prompt_parts = [question]
        if starter_code:
            prompt_parts.append(f"\nStarter code:\n```python\n{starter_code}\n```")
        prompt_parts.append(
            "\nWrite a complete Python solution. Read input from stdin and write output to stdout. "
            "Provide your solution in a ```python code block."
        )
        prompt = "\n".join(prompt_parts)

        messages: list[Message] = [{"role": "user", "content": prompt}]
        try:
            response = await completer(messages)
            content = get_text(response)
            code = _extract_python_code(content)

            loop = asyncio.get_event_loop()
            passed = await loop.run_in_executor(None, _check_solution, code, test_cases_json)
            return {
                "correct": passed,
                "input": question[:200],
                "output": content[:500],
                "difficulty": row.get("difficulty", "unknown"),
            }
        except Exception as e:
            logger.warning(f"LiveCodeBench eval failed: {e}")
            return None

    logger.info(f"LiveCodeBench: evaluating {len(ds)} samples")
    results = await run_concurrent_eval(list(ds), eval_one)

    valid = [r for r in results if r is not None]
    num_correct = sum(1 for r in valid if r["correct"])
    accuracy = num_correct / len(valid) if valid else 0.0

    # Per-difficulty breakdown
    difficulty_results: dict[str, list[bool]] = {}
    for r in valid:
        d = r.get("difficulty", "unknown")
        difficulty_results.setdefault(d, []).append(r["correct"])

    metrics: dict[str, float] = {"livecodebench/pass_at_1": accuracy}
    for d, d_results in sorted(difficulty_results.items()):
        metrics[f"livecodebench/{d}/pass_at_1"] = sum(d_results) / len(d_results) if d_results else 0.0

    logger.info(f"LiveCodeBench final: {num_correct}/{len(valid)} = {accuracy:.4f}")

    return EvalResult(
        benchmark="livecodebench",
        score=accuracy,
        num_examples=len(valid),
        num_correct=num_correct,
        metrics=metrics,
        examples=valid,
    )
