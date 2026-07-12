from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, Any

from tinker_cookbook.recipes.code_rl.code_grading import (
    extract_code_from_model,
    sandbox_check_correctness,
)
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.types import Logs
from tinker_cookbook.sandbox import SandboxBackend
from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool
from tinker_cookbook.utils import logtree


@dataclass(frozen=True)
class DeepcoderTask:
    """A single code task with problem statement and test cases."""

    problem: str
    tests: list[dict[str, Any]]
    starter_code: str | None = None
    source: str | None = None
    """Sub-dataset the task came from (e.g. ``"taco"``, ``"lcbv5"``); used for
    token DB capture dimensions only."""


class DeepcoderTool:
    """Tool for testing code against a task's test cases.

    Each DeepcoderTool instance is bound to a specific task (its tests).
    """

    def __init__(
        self,
        task: DeepcoderTask,
        sandbox_backend: SandboxBackend | None = None,
        timeout: int = 6,
    ):
        self._task = task
        self._sandbox_backend = sandbox_backend
        self._timeout = timeout

    @tool
    async def check_solution(
        self,
        code: Annotated[str, "Python code implementing the solution."],
    ) -> ToolResult:
        """Execute the proposed solution against the task's test cases.

        Use this to test your code before providing your final answer.
        """
        try:
            passed, details = await sandbox_check_correctness(
                self._task.tests,
                code,
                timeout=self._timeout,
                backend=self._sandbox_backend,
            )
            content = json.dumps(
                {"passed": passed, "details": details},
                ensure_ascii=False,
            )
            return simple_tool_result(content)
        except Exception as e:
            return simple_tool_result(json.dumps({"error": str(e), "passed": False}))


# Keys kept when summarizing sandbox grading details for StepResult.logs.
# stdout/stderr are dropped: they are bulky run output, and the same test
# harness output is already rendered into the model's observation whenever the
# model exercises the check_solution tool.
_GRADING_DETAIL_KEYS = ("status", "message", "error", "exit_code", "return_code", "execution_time")


def _grading_details_summary(details: dict[str, Any]) -> dict[str, Any]:
    """Keep the verdict / error-class / timing fields of a sandbox result."""
    summary: dict[str, Any] = {
        key: details[key] for key in _GRADING_DETAIL_KEYS if details.get(key) is not None
    }
    run_result = details.get("run_result")
    if isinstance(run_result, dict):
        nested = {
            key: run_result[key] for key in _GRADING_DETAIL_KEYS if run_result.get(key) is not None
        }
        if nested:
            summary["run_result"] = nested
    return summary


@dataclass
class DeepcoderReward:
    """Reward function for code tasks.

    Grades the final answer by extracting code from the last assistant message
    and running it against the task's tests.

    Formula: format_coef * (has_code_block - 1) + correct

    Called once at episode end with the full message history.
    """

    task: DeepcoderTask
    sandbox_backend: SandboxBackend | None = None
    timeout: int = 6
    format_coef: float = 0.1

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float], Logs]:
        """Grade the completed episode by extracting code from final assistant message.

        Returns ``(reward, metrics, logs)``: metrics carry the reward components
        plus ``tests_total``; logs carry a grading-details summary (verdict,
        error class, timing) for display/capture.
        """
        tests_total = float(len(self.task.tests))

        # Find the last assistant message
        final_message = None
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                final_message = msg
                break

        if final_message is None:
            logtree.log_text("No assistant message found in history.")
            return 0.0, {"format": 0.0, "correct": 0.0, "tests_total": tests_total}, {}

        # Use get_text_content to properly handle thinking models (o1, o3)
        content = get_text_content(final_message)

        # Extract code from content
        code = extract_code_from_model(content)
        has_code_block = code is not None

        # Grade the code by running tests
        grading_logs: Logs = {}
        if code is not None:
            try:
                passed, details = await sandbox_check_correctness(
                    self.task.tests,
                    code,
                    timeout=self.timeout,
                    backend=self.sandbox_backend,
                )
                correct = float(passed)
                grading_logs["grading/verdict"] = "passed" if passed else "failed"
                summary = _grading_details_summary(details)
                if summary:
                    grading_logs["grading/details"] = json.dumps(
                        summary, ensure_ascii=False, default=str
                    )
            except Exception as e:
                logtree.log_text(f"Error running tests: {e}")
                correct = 0.0
                grading_logs["grading/verdict"] = "error"
                grading_logs["grading/error_type"] = type(e).__name__
        else:
            logtree.log_text("No code block detected in response.")
            correct = 0.0
            grading_logs["grading/verdict"] = "no_code_block"

        # Reward formula
        format_score = float(has_code_block)
        reward = self.format_coef * (format_score - 1.0) + correct

        # Log results
        logtree.log_text(f"Problem: {self.task.problem}")
        logtree.log_text(f"Response: {content}")
        logtree.log_text(
            f"Format Valid: {'✓' if has_code_block else '✗'}, "
            f"Correct: {'✓' if correct > 0 else '✗'}, "
            f"Reward: {reward:.2f}"
        )

        return (
            reward,
            {"format": format_score, "correct": correct, "tests_total": tests_total},
            grading_logs,
        )
