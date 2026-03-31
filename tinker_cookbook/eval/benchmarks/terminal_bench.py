"""Terminal-Bench benchmark — multi-turn terminal task solving.

Dataset: ``ia03/terminal-bench`` on HuggingFace.
Evaluation: Multi-turn interaction where the model executes shell commands
in a sandboxed environment, reads output, and iterates until the task is
complete. Graded by running test scripts after the agent finishes.

Metric: Fraction of tasks where all test cases pass.
Pattern: Multi-turn generate + sandbox execution + test grading.

Requires a sandbox backend (Modal or local Docker) for command execution.
"""

from __future__ import annotations

import gzip
import io
import logging
import tarfile
from collections.abc import Sequence
from typing import cast

import yaml

import tinker
from datasets import Dataset

from tinker_cookbook.eval.benchmarks._common import SandboxMixin, extract_command, get_sandbox_factory, is_task_complete, limit_dataset, load_benchmark_dataset, make_example_id
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult

logger = logging.getLogger(__name__)

MAX_TURNS = 40
"""Maximum number of agent turns before forced termination."""

_SYSTEM_PROMPT = """\
You are an expert Linux/Unix systems administrator solving a terminal task \
in a sandboxed environment. You have full shell access.

To execute a command, write it in a ```bash code block:
```bash
your command here
```

You will see the command's stdout and stderr. You can run multiple commands \
across multiple turns. When you are done, write TASK_COMPLETE.

Important:
- You are in a fresh Linux environment
- You can install packages, create files, run scripts
- Your working directory is /workspace
- Be precise and verify your work before declaring completion"""


def _parse_task(row: dict) -> tuple[str, str, dict[str, str]]:
    """Parse a terminal-bench dataset row into (instruction, test_script, setup_files).

    The dataset stores tasks as:
    - ``task_yaml``: YAML string with the ``instruction`` field
    - ``archive``: gzipped tarball containing test scripts, Dockerfiles, etc.

    Falls back to flat field names if the archive format isn't present.
    """
    task_description = ""
    test_script = ""
    setup_files: dict[str, str] = {}

    # Parse task_yaml for the instruction
    task_yaml_str = row.get("task_yaml", "")
    if task_yaml_str:
        try:
            task_yaml = yaml.safe_load(task_yaml_str)
            if isinstance(task_yaml, dict):
                task_description = task_yaml.get("instruction", "")
        except yaml.YAMLError:
            pass

    # Extract files from archive tarball
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

                    # Collect test scripts
                    if name == "run-tests.sh" or name.endswith("/run-tests.sh"):
                        test_script = content
                    # Collect all files for setup (tests, configs, etc.)
                    if not name.startswith("solution"):
                        setup_files[name] = content
        except Exception as e:
            logger.warning(f"terminal_bench: failed to extract archive: {e}")

    # Fallback to flat field names
    if not task_description:
        task_description = row.get("task", row.get("instruction", row.get("description", "")))
    if not test_script:
        test_script = str(row.get("test_script", row.get("run_tests", "")))

    return task_description, test_script, setup_files


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class TerminalBenchEnv(SandboxMixin, Env):
    """Multi-turn env for one Terminal-Bench task.

    On each turn:
    1. Parse the model's response for a bash command
    2. Execute the command in the sandbox
    3. Return stdout/stderr as the next observation
    4. On completion or max turns, run test scripts and grade
    """

    def __init__(
        self,
        task_description: str,
        test_script: str,
        sandbox_factory,
        renderer: Renderer,
        example_id: str = "",
        setup_script: str = "",
        setup_files: dict[str, str] | None = None,
    ):
        self.task_description = task_description
        self.test_script = test_script
        self.sandbox_factory = sandbox_factory
        self.renderer = renderer
        self.example_id = example_id
        self.setup_script = setup_script
        self.setup_files = setup_files or {}

        # Runtime state
        self.messages: list[Message] = []
        self.turn_count = 0
        self.commands_executed: list[str] = []

    async def initial_observation(self):
        # Create sandbox
        self.sandbox = await self.sandbox_factory()

        # Write task files (tests, configs, etc.) extracted from archive
        for filepath, content in self.setup_files.items():
            try:
                dest = f"/workspace/{filepath}"
                await self.sandbox.write_file(dest, content, executable=filepath.endswith(".sh"))
            except Exception as e:
                logger.warning(f"Failed to write setup file {filepath}: {e}")

        # Run setup script if provided
        if self.setup_script:
            try:
                await self.sandbox.run_command(self.setup_script, workdir="/workspace", timeout=120)
            except Exception as e:
                logger.warning(f"Setup script failed: {e}")

        # Write test script to sandbox for later grading (may override archive version)
        if self.test_script:
            await self.sandbox.write_file("/workspace/run_tests.sh", self.test_script, executable=True)

        # Build initial prompt
        self.messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"## Task\n{self.task_description}"},
        ]

        model_input = self.renderer.build_generation_prompt(self.messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        self.turn_count += 1
        response_text = self.renderer.tokenizer.decode(action)

        self.messages.append({"role": "assistant", "content": response_text})

        # Check if task is complete
        if is_task_complete(response_text) or self.turn_count >= MAX_TURNS:
            return await self._finalize()

        # Extract and execute command
        command = extract_command(response_text)
        if command is None:
            # No command found — prompt the model to provide one
            feedback = (
                "I didn't see a bash command in your response. "
                "Please provide a command in a ```bash code block, "
                "or write TASK_COMPLETE if you are done."
            )
            self.messages.append({"role": "user", "content": feedback})
            model_input = self.renderer.build_generation_prompt(self.messages)
            stop = self.renderer.get_stop_sequences()
            return StepResult(
                reward=0.0,
                episode_done=False,
                next_observation=model_input,
                next_stop_condition=stop,
                metrics={"turn": self.turn_count},
                logs={},
            )

        # Execute in sandbox
        self.commands_executed.append(command)
        try:
            result = await self.sandbox.run_command(
                command, workdir="/workspace", timeout=60,
            )
            output_parts = []
            if result.stdout.strip():
                output_parts.append(f"stdout:\n{result.stdout[:4000]}")
            if result.stderr.strip():
                output_parts.append(f"stderr:\n{result.stderr[:2000]}")
            if result.exit_code != 0:
                output_parts.append(f"(exit code: {result.exit_code})")
            if not output_parts:
                output_parts.append("(command completed with no output)")

            feedback = "\n".join(output_parts)
        except Exception as e:
            feedback = f"Command execution error: {e}"

        self.messages.append({"role": "user", "content": feedback})
        model_input = self.renderer.build_generation_prompt(self.messages)
        stop = self.renderer.get_stop_sequences()
        return StepResult(
            reward=0.0,
            episode_done=False,
            next_observation=model_input,
            next_stop_condition=stop,
            metrics={"turn": self.turn_count},
            logs={},
        )

    async def _finalize(self) -> StepResult:
        """Run test scripts and grade the task."""
        test_passed = False
        test_output = ""

        if self.sandbox and self.test_script:
            try:
                result = await self.sandbox.run_command(
                    "cd /workspace && bash run_tests.sh 2>&1",
                    timeout=120,
                )
                test_output = (result.stdout + "\n" + result.stderr).strip()[:2000]
                test_passed = result.exit_code == 0
            except Exception as e:
                test_output = f"Test execution error: {e}"

        # Cleanup sandbox
        await self.cleanup()

        reward = 1.0 if test_passed else 0.0
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={
                "correct": float(test_passed),
                "num_turns": self.turn_count,
                "num_commands": len(self.commands_executed),
            },
            logs={
                "example_id": self.example_id,
                "task": self.task_description[:200],
                "test_output": test_output[:500],
                "num_turns": self.turn_count,
                "commands": "\n".join(self.commands_executed[-5:])[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class TerminalBenchBenchmarkBuilder(BenchmarkBuilder):
    """Terminal-Bench: multi-turn terminal task solving.

    The model executes shell commands in a sandbox, iterating until the task
    is complete. Graded by running test scripts. Requires a sandbox backend —
    uses Modal by default, falls back to a warning if unavailable.
    """

    name = "terminal_bench"
    requires_sandbox = True
    multi_turn = True
    recommended_timeout = 1800

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = cast(Dataset, load_benchmark_dataset("ia03/terminal-bench"))
        ds = limit_dataset(ds, config.max_examples)

        # Create sandbox factory
        sandbox_factory = get_sandbox_factory(config)

        envs = []
        for row in ds:
            task_id = row.get("task_id", "")
            task_description, test_script, setup_files = _parse_task(row)

            if not task_description:
                logger.warning(f"terminal_bench: skipping task {task_id} — no instruction found")
                continue

            if task_id:
                example_id = f"terminal_bench_{task_id}"
            else:
                example_id = make_example_id("terminal_bench", task_description)

            envs.append(TerminalBenchEnv(
                task_description=task_description,
                test_script=test_script,
                sandbox_factory=sandbox_factory,
                renderer=renderer,
                example_id=example_id,
                setup_script="",
                setup_files=setup_files,
            ))

        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(TerminalBenchBenchmarkBuilder())
