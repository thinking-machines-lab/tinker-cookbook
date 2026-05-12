"""Unit tests for Harbor tools and environment."""

import asyncio
import json
import pickle
from pathlib import Path

from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborEnvGroupBuilder, HarborTask
from tinker_cookbook.recipes.harbor_rl.harbor_tools import (
    MAX_OUTPUT_CHARS,
    HarborBashTool,
    HarborGlobTool,
    HarborGrepTool,
    HarborReadFileTool,
    HarborReward,
    HarborStrReplaceFileTool,
    HarborWriteFileTool,
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
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            renderer_name="llama3",
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
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            renderer_name="llama3",
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


# ---------------------------------------------------------------------------
# HarborGlobTool tests
# ---------------------------------------------------------------------------


class TestHarborGlobTool:
    def test_spec_shape(self) -> None:
        sandbox = FakeSandbox()
        tool_obj = HarborGlobTool(sandbox)
        spec = tool_obj.Glob.to_spec()
        assert spec["name"] == "Glob"
        assert "pattern" in spec["parameters"]["properties"]
        assert spec["parameters"]["required"] == ["pattern"]

    def test_glob_basic(self) -> None:
        sandbox = FakeSandbox()
        sandbox.set_command_result(
            sandbox._default_result.stdout,  # fallback
            SandboxResult(
                stdout='{"matches": ["a.py", "b.py"], "total": 2}', stderr="", exit_code=0
            ),
        )
        tool_obj = HarborGlobTool(sandbox)
        # The glob runs python3 inside sandbox; mock any python3 command
        sandbox._command_results.clear()
        sandbox._default_result = SandboxResult(
            stdout='{"matches": ["a.py", "b.py"], "total": 2}', stderr="", exit_code=0
        )

        result = asyncio.run(tool_obj.Glob.run(ToolInput(arguments={"pattern": "*.py"})))
        content = result.messages[0]["content"]
        assert "Found 2 matches" in content
        assert "a.py" in content

    def test_glob_rejects_double_star(self) -> None:
        sandbox = FakeSandbox()
        sandbox._default_result = SandboxResult(stdout="bin\netc\n", stderr="", exit_code=0)
        tool_obj = HarborGlobTool(sandbox)

        result = asyncio.run(tool_obj.Glob.run(ToolInput(arguments={"pattern": "**/*.py"})))
        content = result.messages[0]["content"]
        assert "ERROR" in content
        assert "not allowed" in content


# ---------------------------------------------------------------------------
# HarborGrepTool tests
# ---------------------------------------------------------------------------


class TestHarborGrepTool:
    def test_spec_shape(self) -> None:
        sandbox = FakeSandbox()
        tool_obj = HarborGrepTool(sandbox)
        spec = tool_obj.Grep.to_spec()
        assert spec["name"] == "Grep"
        assert "pattern" in spec["parameters"]["properties"]
        assert spec["parameters"]["required"] == ["pattern"]

    def test_grep_no_matches(self) -> None:
        sandbox = FakeSandbox()
        sandbox._default_result = SandboxResult(stdout="", stderr="", exit_code=1)
        tool_obj = HarborGrepTool(sandbox)

        result = asyncio.run(tool_obj.Grep.run(ToolInput(arguments={"pattern": "nonexistent"})))
        content = result.messages[0]["content"]
        assert "No matches found" in content

    def test_grep_error(self) -> None:
        sandbox = FakeSandbox()
        sandbox._default_result = SandboxResult(stdout="", stderr="bad regex", exit_code=2)
        tool_obj = HarborGrepTool(sandbox)

        result = asyncio.run(tool_obj.Grep.run(ToolInput(arguments={"pattern": "[invalid"})))
        content = result.messages[0]["content"]
        assert "ERROR" in content

    def test_grep_path_prefix_stripping(self) -> None:
        sandbox = FakeSandbox()
        sandbox._default_result = SandboxResult(
            stdout="/src/app/foo.py\n/src/app/bar.py\n", stderr="", exit_code=0
        )
        tool_obj = HarborGrepTool(sandbox)

        result = asyncio.run(
            tool_obj.Grep.run(ToolInput(arguments={"pattern": "test", "path": "/src/app"}))
        )
        content = result.messages[0]["content"]
        assert "foo.py" in content
        assert "/src/app/foo.py" not in content


# ---------------------------------------------------------------------------
# HarborReadFileTool tests
# ---------------------------------------------------------------------------


class TestHarborReadFileTool:
    def test_spec_shape(self) -> None:
        sandbox = FakeSandbox()
        tool_obj = HarborReadFileTool(sandbox)
        spec = tool_obj.ReadFile.to_spec()
        assert spec["name"] == "ReadFile"
        assert "path" in spec["parameters"]["properties"]
        assert spec["parameters"]["required"] == ["path"]

    def test_read_file_basic(self) -> None:
        sandbox = FakeSandbox()
        sandbox.files["/test.py"] = "line1\nline2\nline3\n"
        tool_obj = HarborReadFileTool(sandbox)

        result = asyncio.run(tool_obj.ReadFile.run(ToolInput(arguments={"path": "/test.py"})))
        content = result.messages[0]["content"]
        assert "3 lines read" in content
        assert "line1" in content
        assert "line3" in content

    def test_read_file_not_found(self) -> None:
        sandbox = FakeSandbox()
        tool_obj = HarborReadFileTool(sandbox)

        result = asyncio.run(
            tool_obj.ReadFile.run(ToolInput(arguments={"path": "/nonexistent.py"}))
        )
        content = result.messages[0]["content"]
        assert "ERROR" in content
        assert "does not exist" in content

    def test_read_file_with_offset(self) -> None:
        sandbox = FakeSandbox()
        sandbox.files["/test.py"] = "a\nb\nc\nd\ne\n"
        tool_obj = HarborReadFileTool(sandbox)

        result = asyncio.run(
            tool_obj.ReadFile.run(
                ToolInput(arguments={"path": "/test.py", "line_offset": 3, "n_lines": 2})
            )
        )
        content = result.messages[0]["content"]
        assert "2 lines read" in content
        assert "c" in content


# ---------------------------------------------------------------------------
# HarborWriteFileTool tests
# ---------------------------------------------------------------------------


class TestHarborWriteFileTool:
    def test_spec_shape(self) -> None:
        sandbox = FakeSandbox()
        tool_obj = HarborWriteFileTool(sandbox)
        spec = tool_obj.WriteFile.to_spec()
        assert spec["name"] == "WriteFile"
        assert "path" in spec["parameters"]["properties"]
        assert "content" in spec["parameters"]["properties"]

    def test_write_file_overwrite(self) -> None:
        sandbox = FakeSandbox()
        # Mock stat command for file size
        sandbox.set_command_result(
            "stat -c %s '/test.py'",
            SandboxResult(stdout="13", stderr="", exit_code=0),
        )
        tool_obj = HarborWriteFileTool(sandbox)

        result = asyncio.run(
            tool_obj.WriteFile.run(
                ToolInput(arguments={"path": "/test.py", "content": "hello world\n"})
            )
        )
        content = result.messages[0]["content"]
        assert "overwritten" in content
        assert sandbox.files["/test.py"] == "hello world\n"

    def test_write_file_append(self) -> None:
        sandbox = FakeSandbox()
        sandbox.files["/test.py"] = "existing\n"
        sandbox.set_command_result(
            "stat -c %s '/test.py'",
            SandboxResult(stdout="24", stderr="", exit_code=0),
        )
        tool_obj = HarborWriteFileTool(sandbox)

        result = asyncio.run(
            tool_obj.WriteFile.run(
                ToolInput(arguments={"path": "/test.py", "content": "new line\n", "mode": "append"})
            )
        )
        content = result.messages[0]["content"]
        assert "appended to" in content


# ---------------------------------------------------------------------------
# HarborStrReplaceFileTool tests
# ---------------------------------------------------------------------------


class TestHarborStrReplaceFileTool:
    def test_spec_shape(self) -> None:
        sandbox = FakeSandbox()
        tool_obj = HarborStrReplaceFileTool(sandbox)
        spec = tool_obj.StrReplaceFile.to_spec()
        assert spec["name"] == "StrReplaceFile"
        assert "path" in spec["parameters"]["properties"]
        assert "edit" in spec["parameters"]["properties"]

    def test_str_replace_basic(self) -> None:
        sandbox = FakeSandbox()
        sandbox.files["/test.py"] = "hello world\n"
        tool_obj = HarborStrReplaceFileTool(sandbox)

        result = asyncio.run(
            tool_obj.StrReplaceFile.run(
                ToolInput(
                    arguments={
                        "path": "/test.py",
                        "edit": {"old": "hello", "new": "goodbye"},
                    }
                )
            )
        )
        content = result.messages[0]["content"]
        assert "successfully edited" in content
        assert "1 total replacement" in content
        assert sandbox.files["/test.py"] == "goodbye world\n"

    def test_str_replace_not_found(self) -> None:
        sandbox = FakeSandbox()
        sandbox.files["/test.py"] = "hello world\n"
        tool_obj = HarborStrReplaceFileTool(sandbox)

        result = asyncio.run(
            tool_obj.StrReplaceFile.run(
                ToolInput(
                    arguments={
                        "path": "/test.py",
                        "edit": {"old": "nonexistent", "new": "replacement"},
                    }
                )
            )
        )
        content = result.messages[0]["content"]
        assert "ERROR" in content
        assert "not found" in content

    def test_str_replace_file_not_found(self) -> None:
        sandbox = FakeSandbox()
        tool_obj = HarborStrReplaceFileTool(sandbox)

        result = asyncio.run(
            tool_obj.StrReplaceFile.run(
                ToolInput(
                    arguments={
                        "path": "/nonexistent.py",
                        "edit": {"old": "a", "new": "b"},
                    }
                )
            )
        )
        content = result.messages[0]["content"]
        assert "ERROR" in content
        assert "does not exist" in content

    def test_str_replace_multiple_edits(self) -> None:
        sandbox = FakeSandbox()
        sandbox.files["/test.py"] = "aaa bbb ccc\n"
        tool_obj = HarborStrReplaceFileTool(sandbox)

        result = asyncio.run(
            tool_obj.StrReplaceFile.run(
                ToolInput(
                    arguments={
                        "path": "/test.py",
                        "edit": [
                            {"old": "aaa", "new": "xxx"},
                            {"old": "bbb", "new": "yyy"},
                        ],
                    }
                )
            )
        )
        content = result.messages[0]["content"]
        assert "2 edit(s)" in content
        assert sandbox.files["/test.py"] == "xxx yyy ccc\n"

    def test_str_replace_all(self) -> None:
        sandbox = FakeSandbox()
        sandbox.files["/test.py"] = "aaa bbb aaa\n"
        tool_obj = HarborStrReplaceFileTool(sandbox)

        result = asyncio.run(
            tool_obj.StrReplaceFile.run(
                ToolInput(
                    arguments={
                        "path": "/test.py",
                        "edit": {"old": "aaa", "new": "xxx", "replace_all": True},
                    }
                )
            )
        )
        content = result.messages[0]["content"]
        assert "2 total replacement" in content
        assert sandbox.files["/test.py"] == "xxx bbb xxx\n"
