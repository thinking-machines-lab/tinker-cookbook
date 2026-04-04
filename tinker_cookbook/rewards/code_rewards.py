"""Code execution rewards.

Provides functions for verifying code correctness by running test cases
in a sandbox. Extracted from ``tinker_cookbook.recipes.code_rl`` so that
any recipe can reuse code-grading logic.

Includes telemetry via ``tinker_cookbook.utils.trace`` (Perfetto spans),
``tinker_cookbook.utils.logtree`` (HTML reports), and metric computation
helpers for logging reward statistics.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Sequence
from typing import Any

from tinker_cookbook.sandbox import SandboxBackend
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.trace import scope_span


def extract_code_from_model(model_response: str) -> str | None:
    """Extract the last fenced code block from a model response.

    Returns ``None`` if no code block is found.
    """
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


async def sandbox_check_correctness(
    sample: list[dict[str, Any]],
    generation: str,
    timeout: int = 6,
    backend: SandboxBackend | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Check correctness of generated code using sandbox execution.

    This is a thin re-export of the implementation in
    ``tinker_cookbook.recipes.code_rl.code_grading`` to give a stable
    import path from the rewards library.

    Args:
        sample: List of test cases in LiveCodeBench format.
        generation: Generated code to test.
        timeout: Per-test timeout in seconds.
        backend: Sandbox backend to use (defaults to ``"sandboxfusion"``).

    Returns:
        Tuple of ``(all_passed, details)``.
    """
    from tinker_cookbook.recipes.code_rl.code_grading import (
        sandbox_check_correctness as _impl,
    )
    return await _impl(sample, generation, timeout=timeout, backend=backend)


def grade_code_response(
    response_text: str,
) -> tuple[str | None, bool]:
    """Extract code from a model response and report whether a code block was found.

    Returns:
        Tuple of ``(extracted_code_or_none, has_code_block)``.
    """
    code = extract_code_from_model(response_text)
    return code, code is not None


def taco_to_lcb_format(tests: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert TACO-style tests to LiveCodeBench format.

    Re-exported from ``tinker_cookbook.recipes.code_rl.code_grading``.
    """
    from tinker_cookbook.recipes.code_rl.code_grading import (
        taco_to_lcb_format as _impl,
    )
    return _impl(tests)


# ======================================================================
# Telemetry-instrumented code grading
# ======================================================================


async def sandbox_check_correctness_with_trace(
    sample: list[dict[str, Any]],
    generation: str,
    *,
    timeout: int = 6,
    backend: SandboxBackend | None = None,
    reward_name: str = "code",
    log_to_logtree: bool = True,
) -> tuple[float, dict[str, Any]]:
    """Check code correctness with tracing, logtree logging, and metrics.

    Wraps :func:`sandbox_check_correctness` with telemetry:

    - An async ``scope_span`` named ``"compute_{reward_name}_reward"``
    - Logtree table with grading details
    - Metrics dict with computation time

    Args:
        sample: List of test cases in LiveCodeBench format.
        generation: Generated code to test.
        timeout: Per-test timeout in seconds.
        backend: Sandbox backend to use.
        reward_name: Name for the reward (used in span names and metric keys).
        log_to_logtree: Whether to emit logtree output.

    Returns:
        Tuple of ``(reward_value, metrics_dict)`` where ``reward_value`` is
        1.0 (all tests passed) or 0.0, and ``metrics_dict`` contains
        ``reward/{name}/computation_time``.
    """
    t_start = time.perf_counter()

    async with scope_span(f"compute_{reward_name}_reward"):
        all_passed, details = await sandbox_check_correctness(
            sample, generation, timeout=timeout, backend=backend,
        )

    elapsed = time.perf_counter() - t_start
    reward = 1.0 if all_passed else 0.0

    metrics: dict[str, Any] = {
        f"reward/{reward_name}/computation_time": elapsed,
    }

    if log_to_logtree:
        code_snippet = generation[:200] + "..." if len(generation) > 200 else generation
        with logtree.scope_header("Reward Computation"):
            logtree.table_from_dict({
                "reward_type": "code_execution",
                "all_passed": all_passed,
                "reward": reward,
                "num_tests": len(sample),
                "code_preview": code_snippet,
                "computation_time": f"{elapsed:.4f}s",
            })

    return reward, metrics


def compute_code_reward_metrics(
    rewards: Sequence[float],
    *,
    reward_name: str = "code",
) -> dict[str, float]:
    """Compute aggregate metrics for a batch of code reward values.

    Returns a dict with keys:

    - ``reward/{name}/mean``
    - ``reward/{name}/std``
    - ``reward/{name}/fraction_correct``

    Args:
        rewards: Sequence of reward values (typically 0.0 or 1.0).
        reward_name: Name prefix for the metric keys.
    """
    if not rewards:
        return {}

    n = len(rewards)
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(variance)
    fraction_correct = sum(1.0 for r in rewards if r > 0.5) / n

    return {
        f"reward/{reward_name}/mean": mean,
        f"reward/{reward_name}/std": std,
        f"reward/{reward_name}/fraction_correct": fraction_correct,
    }
