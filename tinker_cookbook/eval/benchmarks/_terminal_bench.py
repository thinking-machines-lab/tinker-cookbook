"""Terminal-Bench benchmark — multi-turn terminal task solving.

Dataset: ``ia03/terminal-bench`` on HuggingFace.
Evaluation: Multi-turn interaction where the model executes shell commands
in a sandboxed environment, reads output, and iterates until the task is
complete. Graded by running test scripts after the agent finishes.

Metric: Fraction of tasks where all test cases pass.
Pattern: Multi-turn ``MessageEnv`` + sandbox Tool + test grading.

Uses the cookbook's ``MessageEnv`` / ``EnvFromMessageEnv`` pattern so that
the renderer handles thinking-token stripping and prompt construction
automatically. The model interacts with the sandbox via a ``bash`` tool.

Requires a sandbox backend (Modal or local Docker) for command execution.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import tarfile
from collections.abc import Sequence
from typing import Annotated, Any, cast

import yaml
from datasets import Dataset

from tinker_cookbook.eval.benchmarks._common import (
    get_sandbox_factory,
    limit_dataset,
    load_benchmark_dataset,
    make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.types import Env
from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.tool_use.agent_tool_message_env import AgentToolMessageEnv
from tinker_cookbook.tool_use.tools import simple_tool_result, tool
from tinker_cookbook.tool_use.types import ToolResult

logger = logging.getLogger(__name__)

MAX_TURNS = 40
"""Maximum number of agent turns before forced termination."""

_SYSTEM_PROMPT = """\
You are an expert Linux/Unix systems administrator solving a terminal task \
in a sandboxed environment. You have full shell access.

Use the bash tool to execute commands. You can run multiple commands \
across multiple turns. When you are done, stop calling tools.

Important:
- You are in a fresh Linux environment
- You can install packages, create files, run scripts
- Your working directory is /workspace
- Be precise and verify your work before stopping"""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class _BashTool:
    """Bash tool that executes commands in a sandbox."""

    def __init__(self, sandbox: SandboxInterface) -> None:
        self._sandbox = sandbox
        self.commands_executed: list[str] = []

    @tool
    async def bash(
        self,
        command: Annotated[str, "The bash command to execute."],
    ) -> ToolResult:
        """Execute a bash command in the sandbox environment."""
        self.commands_executed.append(command)
        result = await self._sandbox.run_command(
            command, workdir="/workspace", timeout=60, max_output_bytes=8000
        )
        output = json.dumps(
            {
                "exit_code": result.exit_code,
                "stdout": result.stdout[:4000],
                "stderr": result.stderr[:2000],
            }
        )
        return simple_tool_result(output)


# ---------------------------------------------------------------------------
# Reward function — runs test scripts to grade
# ---------------------------------------------------------------------------


class _TerminalBenchReward:
    """Reward function that runs test scripts in the sandbox."""

    def __init__(self, sandbox: SandboxInterface, test_script: str) -> None:
        self._sandbox = sandbox
        self._test_script = test_script

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        """Grade by running test scripts. Returns (reward, metrics)."""
        if not self._test_script:
            return 0.0, {"no_test_script": 1.0}

        try:
            result = await self._sandbox.run_command(
                "cd /workspace && bash run_tests.sh 2>&1",
                timeout=120,
            )
            passed = result.exit_code == 0
        except Exception as e:
            logger.warning(f"terminal_bench test execution error: {e}")
            passed = False

        # Count commands from bash tool
        num_commands = sum(
            1 for msg in history if msg.get("role") == "assistant" and msg.get("tool_calls")
        )
        num_turns = sum(1 for msg in history if msg.get("role") == "assistant")

        return (
            1.0 if passed else 0.0,
            {
                "correct": float(passed),
                "num_turns": float(num_turns),
                "num_commands": float(num_commands),
            },
        )


# ---------------------------------------------------------------------------
# Dataset parsing
# ---------------------------------------------------------------------------


def _parse_task(row: dict) -> tuple[str, str, dict[str, str]]:
    """Parse a terminal-bench dataset row into (instruction, test_script, setup_files)."""
    task_description = ""
    test_script = ""
    setup_files: dict[str, str] = {}

    task_yaml_str = row.get("task_yaml", "")
    if task_yaml_str:
        try:
            task_yaml = yaml.safe_load(task_yaml_str)
            if isinstance(task_yaml, dict):
                task_description = task_yaml.get("instruction", "")
        except yaml.YAMLError:
            pass

    archive_data = row.get("archive")
    if archive_data is not None:
        try:
            if isinstance(archive_data, bytes):
                raw = archive_data
            else:
                raw = bytes(archive_data)
            tar_bytes = gzip.decompress(raw)
            with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    name = member.name.lstrip("./")
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    try:
                        content = f.read().decode("utf-8", errors="replace")
                    except Exception:
                        continue
                    if name == "run-tests.sh" or name.endswith("/run-tests.sh"):
                        test_script = content
                    if not name.startswith("solution"):
                        setup_files[name] = content
        except Exception as e:
            logger.warning(f"terminal_bench: failed to extract archive: {e}")

    if not task_description:
        task_description = row.get("task", row.get("instruction", row.get("description", "")))
    if not test_script:
        test_script = str(row.get("test_script", row.get("run_tests", "")))

    return task_description, test_script, setup_files


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class TerminalBenchBenchmarkBuilder(BenchmarkBuilder):
    """Terminal-Bench: multi-turn terminal task solving.

    Uses the cookbook's ``MessageEnv`` / ``EnvFromMessageEnv`` pattern.
    The model interacts with the sandbox via a ``bash`` tool — the renderer
    handles tool call parsing and thinking-token stripping automatically.

    Requires a sandbox backend — uses Modal by default.
    """

    name = "terminal_bench"
    requires_sandbox = True
    multi_turn = True
    recommended_timeout = 1800

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = cast(Dataset, load_benchmark_dataset("ia03/terminal-bench"))
        ds = limit_dataset(ds, config.max_examples)

        sandbox_factory = get_sandbox_factory(
            config, packages=["git", "python3", "python3-pip", "curl", "wget", "build-essential"]
        )

        envs: list[Env] = []
        for row in ds:
            row = dict(row)
            task_id = row.get("task_id", "")
            task_description, test_script, setup_files = _parse_task(row)

            if not task_description:
                logger.warning(f"terminal_bench: skipping task {task_id} — no instruction found")
                continue

            if task_id:
                example_id = f"terminal_bench_{task_id}"
            else:
                example_id = make_example_id("terminal_bench", task_description)

            # Each env gets its own sandbox, tools, and reward function.
            # These are created lazily inside a wrapper that sets up the
            # sandbox on first use.
            envs.append(
                _TerminalBenchEnvFactory(
                    task_description=task_description,
                    test_script=test_script,
                    setup_files=setup_files,
                    sandbox_factory=sandbox_factory,
                    renderer=renderer,
                    example_id=example_id,
                    system_prompt=config.system_prompt,
                )
            )

        return envs


class _TerminalBenchEnvFactory(Env):
    """Wrapper that creates the sandbox and MessageEnv on first observation.

    We can't create the sandbox in ``make_envs`` (it's async and expensive).
    This wrapper delays sandbox creation until ``initial_observation``, then
    delegates to ``EnvFromMessageEnv`` for the actual multi-turn loop.
    """

    def __init__(
        self,
        task_description: str,
        test_script: str,
        setup_files: dict[str, str],
        sandbox_factory: Any,
        renderer: Renderer,
        example_id: str,
        system_prompt: str | None = None,
    ):
        self.task_description = task_description
        self.test_script = test_script
        self.setup_files = setup_files
        self.sandbox_factory = sandbox_factory
        self.renderer = renderer
        self.example_id = example_id
        self.system_prompt = system_prompt

        self._inner: EnvFromMessageEnv | None = None
        self._sandbox: SandboxInterface | None = None

    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        if self._sandbox is not None:
            try:
                await self._sandbox.cleanup()
            except Exception:
                logger.debug("Sandbox cleanup failed", exc_info=True)

    async def initial_observation(self):
        # Create sandbox
        self._sandbox = await self.sandbox_factory()

        # Write setup files
        for filepath, content in self.setup_files.items():
            try:
                dest = f"/workspace/{filepath}"
                await self._sandbox.write_file(dest, content, executable=filepath.endswith(".sh"))
            except Exception as e:
                logger.warning(f"Failed to write setup file {filepath}: {e}")

        # Write test script
        if self.test_script:
            await self._sandbox.write_file(
                "/workspace/run_tests.sh", self.test_script, executable=True
            )

        # Create tool and reward
        bash_tool = _BashTool(self._sandbox)
        reward_fn = _TerminalBenchReward(self._sandbox, self.test_script)

        # Build initial messages with tool specs in the renderer's native format
        system_content = self.system_prompt or _SYSTEM_PROMPT
        tool_specs = [bash_tool.bash.to_spec()]
        initial_messages = self.renderer.create_conversation_prefix_with_tools(
            tools=tool_specs, system_prompt=system_content,
        )
        initial_messages.append({"role": "user", "content": f"## Task\n{self.task_description}"})

        # Create the inner MessageEnv + adapter
        msg_env = AgentToolMessageEnv(
            tools=[bash_tool.bash],
            initial_messages=initial_messages,
            max_turns=MAX_TURNS,
            reward_fn=reward_fn,
        )
        self._inner = EnvFromMessageEnv(
            renderer=self.renderer,
            message_env=msg_env,
            failed_parse_reward=0.0,
            terminate_on_parse_error=False,
        )

        return await self._inner.initial_observation()

    async def step(self, action, *, extra=None):
        assert self._inner is not None
        result = await self._inner.step(action, extra=extra)
        # Inject example_id into logs for trajectory storage
        if result.episode_done:
            result.logs["example_id"] = self.example_id
        return result


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(TerminalBenchBenchmarkBuilder())
