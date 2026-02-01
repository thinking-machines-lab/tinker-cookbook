from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, Any

from tinker_cookbook.recipes.code_rl.code_grading import (
    extract_code_from_model,
    sandbox_check_correctness,
)
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.sandbox import SandboxBackend
from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool
from tinker_cookbook.utils import logtree


@dataclass(frozen=True)
class DeepcoderTask:
    """A single code task with problem statement and test cases."""

    problem: str
    tests: list[dict[str, Any]]
    starter_code: str | None = None


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

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        """Grade the completed episode by extracting code from final assistant message."""
        # Find the last assistant message
        final_message = None
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                final_message = msg
                break

        if final_message is None:
            logtree.log_text("No assistant message found in history.")
            return 0.0, {"format": 0.0, "correct": 0.0}

        content = final_message.get("content", "")
        if not isinstance(content, str):
            content = str(content)

        # Extract code from content
        code = extract_code_from_model(content)
        has_code_block = code is not None

        # Grade the code by running tests
        if code is not None:
            try:
                passed, _details = await sandbox_check_correctness(
                    self.task.tests,
                    code,
                    timeout=self.timeout,
                    backend=self.sandbox_backend,
                )
                correct = float(passed)
            except Exception as e:
                logtree.log_text(f"Error running tests: {e}")
                correct = 0.0
        else:
            logtree.log_text("No code block detected in response.")
            correct = 0.0

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

        return reward, {"format": format_score, "correct": correct}
