"""Code RL tools and reward function.

Provides @tool-decorated methods for code execution and submission,
using sandbox execution from tinker_cookbook.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, Iterable, cast

from tinker_cookbook.recipes.code.task import CodeRLTask
from tinker_cookbook.recipes.code_rl.code_grading import sandbox_check_correctness
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.sandbox import SandboxBackend
from tinker_cookbook.tool_use import ToolInterface, extract_tool_payload, tool


@dataclass
class SubmissionReward:
    """Reward function: reward on submit_answer success.

    For code_rl, termination happens when submit_answer is called.
    The assistant message is ignored (we only care about tool results).
    """

    success_reward: float = 1.0
    failure_reward: float = 0.0

    def __call__(
        self, results: list[Message], message: Message
    ) -> tuple[float, bool, dict[str, float]]:
        reward = 0.0
        done = False
        metrics: dict[str, float] = {}

        for result in results:
            content = result.get("content", "")
            payload = extract_tool_payload(content) if isinstance(content, str) else None
            if payload is None:
                continue

            name = result.get("name", "")
            if name in {"run_python", "submit_answer"}:
                passed_all = float(payload.get("passed_all", 0.0))
                metrics["passed_all"] = passed_all
                if "error" in payload:
                    metrics["error"] = 1.0

            if name == "submit_answer":
                done = True
                reward += self.success_reward if payload.get("passed_all") else self.failure_reward

        return reward, done, metrics


class CodeRLTools:
    """Tools for code RL tasks with sandbox execution."""

    def __init__(
        self,
        tasks: dict[str, CodeRLTask],
        sandbox_backend: SandboxBackend | None = None,
        timeout: int = 6,
    ):
        self._tasks = tasks
        self._sandbox_backend = sandbox_backend
        self._timeout = timeout

    @tool
    async def run_python(
        self,
        task_id: Annotated[str, "ID of the coding task to evaluate."],
        code: Annotated[str, "Python code implementing the solution."],
    ) -> str:
        """Execute the proposed solution against the task's test cases. Use this to test your code before submitting."""
        task = self._tasks.get(task_id)
        if task is None:
            return json.dumps({"error": f"Unknown task_id {task_id}"})

        try:
            passed, details = await sandbox_check_correctness(
                task.tests,
                code,
                timeout=self._timeout,
                backend=self._sandbox_backend,
            )
            return json.dumps(
                {"passed_all": passed, "details": _truncate_details(details)},
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"error": str(e), "passed_all": False})

    @tool
    async def submit_answer(
        self,
        task_id: Annotated[str, "ID of the coding task being submitted."],
        code: Annotated[str, "Python code implementing the final answer."],
    ) -> str:
        """Submit the final solution for grading. The code will be run against all tests to compute reward."""
        task = self._tasks.get(task_id)
        if task is None:
            return json.dumps({"error": f"Unknown task_id {task_id}"})

        try:
            passed, details = await sandbox_check_correctness(
                task.tests,
                code,
                timeout=self._timeout,
                backend=self._sandbox_backend,
            )
            return json.dumps(
                {"passed_all": passed, "details": _truncate_details(details)},
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"error": str(e), "passed_all": False})


def _truncate_details(details: dict) -> dict:
    """Truncate verbose details to avoid token bloat."""
    result = {}
    for key, value in details.items():
        if isinstance(value, str) and len(value) > 500:
            result[key] = value[:500] + "...[truncated]"
        else:
            result[key] = value
    return result


def build_tools(
    tasks: Iterable[CodeRLTask],
    sandbox_backend: SandboxBackend | None = None,
    timeout: int = 6,
) -> list[ToolInterface]:
    """Construct a list of code RL tools for the given tasks."""
    tasks_dict = {task.task_id: task for task in tasks}
    tools_obj = CodeRLTools(tasks_dict, sandbox_backend=sandbox_backend, timeout=timeout)
    return cast(list[ToolInterface], [tools_obj.run_python, tools_obj.submit_answer])
