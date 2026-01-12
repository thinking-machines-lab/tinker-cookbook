"""Code execution tool and reward function."""

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
from tinker_cookbook.tool_use import tool
from tinker_cookbook.utils import logtree


@dataclass(frozen=True)
class CodeTask:
    """A single code task with problem statement and test cases."""

    problem: str
    tests: list[dict[str, Any]]
    starter_code: str | None = None


class CodeTool:
    """Tool for testing code against a task's test cases.

    Each CodeTool instance is bound to a specific task (its tests).
    """

    def __init__(
        self,
        task: CodeTask,
        sandbox_backend: SandboxBackend | None = None,
        timeout: int = 6,
    ):
        self._task = task
        self._sandbox_backend = sandbox_backend
        self._timeout = timeout

    @tool
    async def run_python(
        self,
        code: Annotated[str, "Python code implementing the solution."],
    ) -> str:
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
            return json.dumps(
                {"passed": passed, "details": details},
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"error": str(e), "passed": False})

    async def grade_code(self, code: str) -> tuple[bool, dict]:
        """Grade code against tests. Used by reward_fn."""
        try:
            passed, details = await sandbox_check_correctness(
                self._task.tests,
                code,
                timeout=self._timeout,
                backend=self._sandbox_backend,
            )
            status = "✓" if passed else "✗"
            logtree.log_text(
                f"Sandbox result {status}: {'All tests passed' if passed else 'Failed'}"
            )
            return passed, details
        except Exception as exc:
            logtree.log_text(f"Sandbox check failed: {exc}")
            return False, {}


@dataclass
class CodeReward:
    """Reward function for code tasks.

    Grades the final answer by extracting code from message content
    and running it against the task's tests using the same tool.

    Formula: format_coef * (has_code_block - 1) + correct
    """

    code_tool: CodeTool
    format_coef: float = 0.1

    async def __call__(
        self, results: list[Message], message: Message
    ) -> tuple[float, bool, dict[str, float]]:
        # If message has tool calls, this is an intermediate step
        if message.get("tool_calls"):
            return 0.0, False, {}

        # No tool calls - this is the final answer, grade it
        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)

        # Extract code from content
        code = extract_code_from_model(content)
        has_code_block = code is not None

        # Grade the code using the same tool
        if code is not None:
            passed, _ = await self.code_tool.grade_code(code)
            correct = float(passed)
        else:
            logtree.log_text("No code block detected in response.")
            correct = 0.0

        # Reward formula matches old CodeEnv
        format_score = float(has_code_block)
        reward = self.format_coef * (format_score - 1.0) + correct

        # Log results (matching old CodeEnv behavior)
        logtree.log_text(f"Problem: {self.code_tool._task.problem}")
        logtree.log_text(f"Response: {content}")
        logtree.log_text(
            f"Format Valid: {'✓' if has_code_block else '✗'}, "
            f"Correct: {'✓' if correct > 0 else '✗'}, "
            f"Reward: {reward:.2f}"
        )

        return reward, True, {"format": format_score, "correct": correct}
