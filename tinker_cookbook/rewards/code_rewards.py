"""Code execution rewards.

Provides functions for verifying code correctness by running test cases
in a sandbox. Extracted from ``tinker_cookbook.recipes.code_rl`` so that
any recipe can reuse code-grading logic.

Includes telemetry via ``tinker_cookbook.utils.trace`` (Perfetto spans),
``tinker_cookbook.utils.logtree`` (HTML reports), and metric computation
helpers for logging reward statistics.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from tinker_cookbook.utils import logtree

if TYPE_CHECKING:
    from tinker_cookbook.sandbox import SandboxBackend
from tinker_cookbook.utils.trace import scope_span


def extract_code_block(model_response: str) -> str | None:
    """Extract the last fenced code block from a model response.

    Returns ``None`` if no code block is found.
    """
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


async def score_code_sandbox(
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


def grade_code_correctness(
    response_text: str,
) -> tuple[str | None, bool]:
    """Extract code from a model response and report whether a code block was found.

    Returns:
        Tuple of ``(extracted_code_or_none, has_code_block)``.
    """
    code = extract_code_block(response_text)
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


async def score_code_sandbox_traced(
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
        all_passed, details = await score_code_sandbox(
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

    Thin wrapper around :func:`~tinker_cookbook.rewards._metrics.compute_reward_metrics`
    with ``reward_name`` defaulting to ``"code"``.
    """
    from tinker_cookbook.rewards._metrics import compute_reward_metrics

    return compute_reward_metrics(rewards, reward_name)


# ======================================================================
# Deprecated aliases (backward compatibility)
# ======================================================================

extract_code_from_model = extract_code_block
grade_code_response = grade_code_correctness
sandbox_check_correctness = score_code_sandbox
sandbox_check_correctness_with_trace = score_code_sandbox_traced
