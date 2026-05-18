"""Harbor bash tool and reward function for RL training."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from tinker_cookbook.renderers.base import Message
from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 16384


class HarborBashTool:
    """Bash tool that executes commands in a sandbox.

    Wraps a SandboxInterface as a tinker_cookbook Tool via the @tool decorator.
    """

    def __init__(self, sandbox: SandboxInterface, command_timeout: int = 120) -> None:
        self._sandbox = sandbox
        self._command_timeout = command_timeout

    @tool
    async def bash(
        self,
        command: Annotated[str, "The bash command to execute."],
    ) -> ToolResult:
        """Execute a bash command in the sandbox environment.

        Use this to run shell commands, install packages, edit files, etc.
        """
        result = await self._sandbox.run_command(
            command, workdir="/", timeout=self._command_timeout, max_output_bytes=MAX_OUTPUT_CHARS
        )
        stdout = result.stdout[:MAX_OUTPUT_CHARS]
        stderr = result.stderr[:MAX_OUTPUT_CHARS]
        output = json.dumps({"exit_code": result.exit_code, "stdout": stdout, "stderr": stderr})
        return simple_tool_result(output)


@dataclass
class HarborReward:
    """Reward function for Harbor tasks.

    Grades by uploading test files to the sandbox, running test.sh,
    and parsing reward from /logs/verifier/reward.txt or reward.json.

    Called once at episode end with the full message history.
    """

    tests_dir: Path
    sandbox: SandboxInterface
    grader_timeout: int = 60

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        """Grade the completed episode by running test.sh in the sandbox."""
        try:
            # 1. Upload test files to /tests/ in sandbox
            await self._upload_tests()

            # 2. Create log directory and run test.sh
            # Run from /root (not /) because test.sh checks if PWD=/ and exits early
            await self.sandbox.run_command("mkdir -p /logs/verifier", workdir="/root")
            result = await self.sandbox.run_command(
                "bash /tests/test.sh",
                workdir="/root",
                timeout=self.grader_timeout,
            )
            logger.info("test.sh completed with exit_code=%d", result.exit_code)
            if result.stdout:
                logger.debug("test.sh stdout: %s", result.stdout[:500])
            if result.stderr:
                logger.debug("test.sh stderr: %s", result.stderr[:500])

            # 3. Parse reward
            reward = await self._parse_reward()
            return reward, {"reward": reward, "test_passed": float(reward > 0)}

        except Exception as e:
            logger.error("Harbor grading failed: %s", e)
            return 0.0, {"reward": 0.0, "test_passed": 0.0, "grading_error": 1.0}

    async def _upload_tests(self) -> None:
        """Upload test files from local tests_dir to /tests/ in sandbox.

        Walks ``tests_dir`` recursively so nested fixture trees (e.g. a
        ``tests/repo_overlay/`` directory copied into the workspace by
        ``test.sh``) are preserved. ``Path.iterdir`` would silently drop them.
        """
        await self.sandbox.run_command("mkdir -p /tests", workdir="/")
        for file_path in self.tests_dir.rglob("*"):
            if not file_path.is_file():
                continue
            content = file_path.read_text()
            target = f"/tests/{file_path.relative_to(self.tests_dir)}"
            await self.sandbox.write_file(target, content, executable=(file_path.suffix == ".sh"))

    async def _parse_reward(self) -> float:
        """Parse reward from /logs/verifier/reward.txt or reward.json."""
        # Try reward.txt first
        result = await self.sandbox.read_file("/logs/verifier/reward.txt")
        if result.exit_code == 0 and result.stdout.strip():
            reward = float(result.stdout.strip())
            logger.info("Parsed reward from reward.txt: %s", reward)
            return reward

        # Try reward.json
        result = await self.sandbox.read_file("/logs/verifier/reward.json")
        if result.exit_code == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            reward = float(data.get("reward", 0.0))
            logger.info("Parsed reward from reward.json: %s", reward)
            return reward

        logger.warning("No reward file found at /logs/verifier/")
        return 0.0
