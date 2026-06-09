"""Unit tests for HarborReward, HarborBashTool, and HarborEnvGroupBuilder."""

import asyncio
import json
import pickle
from pathlib import Path

from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborEnvGroupBuilder, HarborTask
from tinker_cookbook.recipes.harbor_rl.harbor_tools import (
    MAX_OUTPUT_CHARS,
    HarborBashTool,
    HarborReward,
)
from tinker_cookbook.sandbox.sandbox_interface import SandboxResult
from tinker_cookbook.tool_use.types import ToolInput


class FakeSandbox:
    """In-memory sandbox for testing."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.executable_files: set[str] = set()
        self.commands_run: list[str] = []
        self._command_results: dict[str, SandboxResult] = {}
        self._default_result = SandboxResult(stdout="", stderr="", exit_code=0)

    @property
    def sandbox_id(self) -> str:
        return "fake-sandbox"

    def set_command_result(self, command: str, result: SandboxResult) -> None:
        self._command_results[command] = result

    async def send_heartbeat(self, timeout: int = 30) -> None:
        pass

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        self.commands_run.append(command)
        if command in self._command_results:
            return self._command_results[command]
        return self._default_result

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        if path in self.files:
            return SandboxResult(stdout=self.files[path], stderr="", exit_code=0)
        return SandboxResult(stdout="", stderr=f"No such file: {path}", exit_code=1)

    async def write_file(
        self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60
    ) -> SandboxResult:
        self.files[path] = content if isinstance(content, str) else content.decode()
        if executable:
            self.executable_files.add(path)
        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def cleanup(self) -> None:
        pass


# ---------------------------------------------------------------------------
# HarborReward tests
# ---------------------------------------------------------------------------


class TestHarborReward:
    def _make_reward(self, tmp_path: Path, sandbox: FakeSandbox, **kwargs) -> HarborReward:
        return HarborReward(tests_dir=tmp_path, sandbox=sandbox, **kwargs)

    def test_reward_from_txt(self, tmp_path: Path) -> None:
        sandbox = FakeSandbox()
        sandbox.files["/logs/verifier/reward.txt"] = "1.0"
        reward_fn = self._make_reward(tmp_path, sandbox)

        reward, info = asyncio.run(reward_fn([]))
        assert reward == 1.0
        assert info == {"reward": 1.0, "test_passed": 1.0}

    def test_reward_from_json(self, tmp_path: Path) -> None:
        sandbox = FakeSandbox()
        sandbox.files["/logs/verifier/reward.json"] = json.dumps({"reward": 0.75})
        reward_fn = self._make_reward(tmp_path, sandbox)

        reward, info = asyncio.run(reward_fn([]))
        assert reward == 0.75
        assert info == {"reward": 0.75, "test_passed": 1.0}

    def test_no_reward_file(self, tmp_path: Path) -> None:
        sandbox = FakeSandbox()
        reward_fn = self._make_reward(tmp_path, sandbox)

        reward, info = asyncio.run(reward_fn([]))
        assert reward == 0.0
        assert info == {"reward": 0.0, "test_passed": 0.0}

    def test_zero_reward(self, tmp_path: Path) -> None:
        sandbox = FakeSandbox()
        sandbox.files["/logs/verifier/reward.txt"] = "0.0"
        reward_fn = self._make_reward(tmp_path, sandbox)

        reward, info = asyncio.run(reward_fn([]))
        assert reward == 0.0
        assert info["test_passed"] == 0.0

    def test_grading_error(self, tmp_path: Path) -> None:
        """Sandbox exception during grading returns 0 reward with error flag."""

        class ExplodingSandbox(FakeSandbox):
            async def run_command(
                self,
                command: str,
                workdir: str | None = None,
                timeout: int = 60,
                max_output_bytes: int | None = None,
            ) -> SandboxResult:
                raise RuntimeError("sandbox died")

        sandbox = ExplodingSandbox()
        reward_fn = self._make_reward(tmp_path, sandbox)

        reward, info = asyncio.run(reward_fn([]))
        assert reward == 0.0
        assert info["grading_error"] == 1.0

    def test_upload_tests(self, tmp_path: Path) -> None:
        """Files are uploaded recursively; .sh files marked executable."""
        (tmp_path / "test.sh").write_text("#!/bin/bash\necho ok")
        (tmp_path / "helper.py").write_text("print('hi')")
        (tmp_path / "fixtures").mkdir()
        (tmp_path / "fixtures" / "expected.txt").write_text("expected")
        (tmp_path / "fixtures" / "nested").mkdir()
        (tmp_path / "fixtures" / "nested" / "deep.py").write_text("# deep")

        sandbox = FakeSandbox()
        sandbox.files["/logs/verifier/reward.txt"] = "1.0"
        reward_fn = self._make_reward(tmp_path, sandbox)

        asyncio.run(reward_fn([]))

        assert "/tests/test.sh" in sandbox.files
        assert "/tests/helper.py" in sandbox.files
        assert "/tests/fixtures/expected.txt" in sandbox.files
        assert "/tests/fixtures/nested/deep.py" in sandbox.files
        assert "/tests/test.sh" in sandbox.executable_files
        assert "/tests/helper.py" not in sandbox.executable_files
        assert "/tests/fixtures/expected.txt" not in sandbox.executable_files


# ---------------------------------------------------------------------------
# HarborBashTool tests
# ---------------------------------------------------------------------------


class TestHarborBashTool:
    def test_bash_tool_basic(self) -> None:
        sandbox = FakeSandbox()
        sandbox.set_command_result(
            "echo hello",
            SandboxResult(stdout="hello\n", stderr="", exit_code=0),
        )
        tool_obj = HarborBashTool(sandbox)

        result = asyncio.run(tool_obj.bash.run(ToolInput(arguments={"command": "echo hello"})))
        content = result.messages[0]["content"]
        assert isinstance(content, str)
        output = json.loads(content)
        assert output["exit_code"] == 0
        assert output["stdout"] == "hello\n"
        assert output["stderr"] == ""

    def test_bash_tool_truncation(self) -> None:
        long_stdout = "x" * (MAX_OUTPUT_CHARS + 100)
        long_stderr = "e" * (MAX_OUTPUT_CHARS + 100)
        sandbox = FakeSandbox()
        sandbox.set_command_result(
            "big_cmd",
            SandboxResult(stdout=long_stdout, stderr=long_stderr, exit_code=1),
        )
        tool_obj = HarborBashTool(sandbox)

        result = asyncio.run(tool_obj.bash.run(ToolInput(arguments={"command": "big_cmd"})))
        content = result.messages[0]["content"]
        assert isinstance(content, str)
        output = json.loads(content)
        assert len(output["stdout"]) == MAX_OUTPUT_CHARS
        assert len(output["stderr"]) == MAX_OUTPUT_CHARS
        assert output["exit_code"] == 1


# ---------------------------------------------------------------------------
# HarborEnvGroupBuilder pickle tests
# ---------------------------------------------------------------------------


class TestHarborEnvGroupBuilderPickle:
    def _make_task(self, tmp_path: Path) -> HarborTask:
        return HarborTask(
            task_name="test-task",
            instruction="Fix the bug",
            task_dir=tmp_path,
            config={"difficulty": "easy"},
        )

    def test_pickle_roundtrip(self, tmp_path: Path) -> None:
        """HarborEnvGroupBuilder survives pickle/unpickle with default sandbox_factory."""
        builder = HarborEnvGroupBuilder(
            task=self._make_task(tmp_path),
            model_name="Qwen/Qwen3.5-9B",
            renderer_name="qwen3_5_disable_thinking",
            max_turns=5,
            group_size=2,
        )

        restored = pickle.loads(pickle.dumps(builder))

        assert restored.task == builder.task
        assert restored.model_name == builder.model_name
        assert restored.renderer_name == builder.renderer_name
        assert restored.max_turns == builder.max_turns
        assert restored.group_size == builder.group_size
        assert restored.sandbox_timeout == builder.sandbox_timeout
        assert restored.command_timeout == builder.command_timeout
        assert restored.grader_timeout == builder.grader_timeout
        assert restored.max_trajectory_tokens == builder.max_trajectory_tokens
        assert restored.sandbox_factory is builder.sandbox_factory

    def test_pickle_with_custom_params(self, tmp_path: Path) -> None:
        """Non-default scalar parameters survive pickle roundtrip."""
        builder = HarborEnvGroupBuilder(
            task=self._make_task(tmp_path),
            model_name="Qwen/Qwen3.5-9B",
            renderer_name="qwen3_5_disable_thinking",
            max_turns=5,
            group_size=2,
            sandbox_timeout=300,
            command_timeout=60,
            grader_timeout=30,
            max_trajectory_tokens=16 * 1024,
        )

        restored = pickle.loads(pickle.dumps(builder))

        assert restored.sandbox_timeout == 300
        assert restored.command_timeout == 60
        assert restored.grader_timeout == 30
        assert restored.max_trajectory_tokens == 16 * 1024
