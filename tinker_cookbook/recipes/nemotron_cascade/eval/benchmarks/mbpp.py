"""MBPP benchmark -- Mostly Basic Python Programming via code execution."""

from __future__ import annotations

import asyncio
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


async def evaluate(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    max_tokens: int = 32768,
    max_examples: int | None = None,
) -> EvalResult:
    """Evaluate on MBPP via code execution with assertion-based tests."""
    try:
        ds = cast(Dataset, load_dataset("google-research-datasets/mbpp", "sanitized", split="test"))
    except Exception:
        ds = cast(Dataset, load_dataset("google-research-datasets/mbpp", "full", split="test"))

    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    completer = make_completer(sampling_client, renderer, max_tokens)

    async def eval_one(row: dict) -> dict | None:
        task_prompt = row.get("prompt", row.get("text", ""))
        test_list = row.get("test_list", [])
        if not task_prompt or not test_list:
            return None

        example_test = test_list[0] if test_list else ""
        prompt = (
            f"{task_prompt}\n\n"
            f"Example test: `{example_test}`\n\n"
            "Write a Python function that satisfies the requirements. "
            "Provide ONLY the function definition in a ```python code block."
        )
        messages: list[Message] = [{"role": "user", "content": prompt}]
        try:
            response = await completer(messages)
            content = get_text(response)
            code = _extract_python_code(content)

            loop = asyncio.get_event_loop()
            passed = await loop.run_in_executor(None, _run_python_tests, code, test_list)
            return {"correct": passed, "input": task_prompt, "output": content}
        except Exception as e:
            logger.warning(f"MBPP eval failed: {e}")
            return None

    logger.info(f"MBPP: evaluating {len(ds)} samples")
    results = await run_concurrent_eval(list(ds), eval_one)

    valid = [r for r in results if r is not None]
    num_correct = sum(1 for r in valid if r["correct"])
    accuracy = num_correct / len(valid) if valid else 0.0
    logger.info(f"MBPP final: {num_correct}/{len(valid)} = {accuracy:.4f}")

    return EvalResult(
        benchmark="mbpp",
        score=accuracy,
        num_examples=len(valid),
        num_correct=num_correct,
        metrics={"mbpp/accuracy": accuracy},
        examples=valid,
    )
