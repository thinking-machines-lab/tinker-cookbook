"""
SWE Agentic RL environment for Nemotron-Cascade-2 replication.

Multi-turn agentic environment where the model interacts with a codebase
through tool calls (read_file, write_file, run_command) to fix bugs.
After the model finishes (or max_turns reached), FAIL_TO_PASS tests
are executed in a Modal sandbox to determine the reward.

Paper hyperparameters (SWE Agentic RL):
  - Batch 16, Group 64, max_tokens 256K, max_turns 200, temp 0.8
  - Data: SWE-Gym + R2E-Subset (loaded from nvidia/Nemotron-Cascade-2-RL-data)
  - Reward: binary (tests pass = 1, fail = 0)

Key difference from swe_rl_env.py (agentless):
  - Agentless: single-turn, one-shot patch generation
  - Agentic: multi-turn interaction with read_file, write_file, run_command tools
"""

import json
import logging
import math
from collections.abc import Sequence
from typing import Annotated, cast

import chz
import modal
from datasets import Dataset

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
)
from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox
from tinker_cookbook.tool_use import ToolResult, build_agent_tool_env, simple_tool_result, tool
from tinker_cookbook.tool_use.agent_tool_message_env import RewardFn

logger = logging.getLogger(__name__)

# Maximum characters returned from tool outputs to avoid overwhelming context
MAX_OUTPUT_CHARS = 16384

# System prompt for the SWE agentic environment
SWE_AGENTIC_SYSTEM_PROMPT = (
    "You are an expert software engineer. You have access to a repository checkout "
    "in a sandboxed environment. Your task is to fix the described issue by reading "
    "relevant files, understanding the codebase, and making the necessary changes.\n\n"
    "You have three tools available:\n"
    "- read_file: Read the contents of a file\n"
    "- write_file: Write content to a file (creates or overwrites)\n"
    "- run_command: Execute a shell command in the repository\n\n"
    "Start by exploring the repository structure and understanding the issue, "
    "then make targeted changes to fix the problem. When you are done, "
    "simply respond without making any tool calls."
)


class SWEAgenticTools:
    """Tools for interacting with a codebase in a sandbox.

    Provides read_file, write_file, and run_command tools that operate
    on a Modal sandbox with the target repository checked out.
    """

    def __init__(
        self,
        sandbox: SandboxInterface,
        command_timeout: int = 120,
        workdir: str = "/workspace/repo",
    ) -> None:
        self._sandbox = sandbox
        self._command_timeout = command_timeout
        self._workdir = workdir

    @tool
    async def read_file(
        self,
        path: Annotated[str, "Path to the file to read, relative to the repository root or absolute."],
    ) -> ToolResult:
        """Read the contents of a file in the repository."""
        # Resolve relative paths against workdir
        if not path.startswith("/"):
            path = f"{self._workdir}/{path}"

        result = await self._sandbox.read_file(path, max_bytes=MAX_OUTPUT_CHARS)
        if result.exit_code != 0:
            output = json.dumps({"error": f"Failed to read file: {result.stderr.strip()}"})
        else:
            content = result.stdout[:MAX_OUTPUT_CHARS]
            output = json.dumps({"content": content})
        return simple_tool_result(output)

    @tool
    async def write_file(
        self,
        path: Annotated[str, "Path to the file to write, relative to the repository root or absolute."],
        content: Annotated[str, "The full content to write to the file."],
    ) -> ToolResult:
        """Write content to a file in the repository. Creates the file if it doesn't exist."""
        if not path.startswith("/"):
            path = f"{self._workdir}/{path}"

        result = await self._sandbox.write_file(path, content)
        if result.exit_code != 0:
            output = json.dumps({"error": f"Failed to write file: {result.stderr.strip()}"})
        else:
            output = json.dumps({"status": "ok", "path": path})
        return simple_tool_result(output)

    @tool
    async def run_command(
        self,
        command: Annotated[str, "The shell command to execute in the repository."],
    ) -> ToolResult:
        """Execute a shell command in the repository directory."""
        result = await self._sandbox.run_command(
            command, workdir=self._workdir, timeout=self._command_timeout,
            max_output_bytes=MAX_OUTPUT_CHARS,
        )
        stdout = result.stdout[:MAX_OUTPUT_CHARS]
        stderr = result.stderr[:MAX_OUTPUT_CHARS]
        output = json.dumps({
            "exit_code": result.exit_code,
            "stdout": stdout,
            "stderr": stderr,
        })
        return simple_tool_result(output)


class SWEAgenticReward:
    """Reward function that runs FAIL_TO_PASS tests in the sandbox.

    Returns binary reward: 1.0 if all tests pass, 0.0 otherwise.
    """

    def __init__(
        self,
        sandbox: SandboxInterface,
        fail_to_pass: list[str],
        workdir: str = "/workspace/repo",
        test_timeout: int = 300,
    ) -> None:
        self._sandbox = sandbox
        self._fail_to_pass = fail_to_pass
        self._workdir = workdir
        self._test_timeout = test_timeout

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        """Run tests and return (reward, metrics)."""
        if not self._fail_to_pass:
            return 0.0, {"reward": 0.0, "no_tests": 1.0}

        try:
            passed_count = 0
            total = len(self._fail_to_pass)

            for test in self._fail_to_pass:
                result = await self._sandbox.run_command(
                    f'python -m pytest "{test}" -x 2>&1',
                    workdir=self._workdir,
                    timeout=self._test_timeout,
                )
                if result.exit_code == 0:
                    passed_count += 1

            all_passed = passed_count == total
            reward = 1.0 if all_passed else 0.0

            return reward, {
                "reward": reward,
                "tests_passed": float(passed_count),
                "tests_total": float(total),
                "pass_rate": passed_count / total if total > 0 else 0.0,
            }

        except Exception as e:
            logger.error("SWE agentic grading failed: %s", e)
            return 0.0, {"reward": 0.0, "grading_error": 1.0}


def _build_initial_messages(
    problem_statement: str,
    repo: str,
    instance_id: str,
    renderer: Renderer,
    tools: SWEAgenticTools,
    workdir: str = "/workspace/repo",
) -> list[Message]:
    """Build the initial message list with tool schemas and problem description."""
    tool_schemas = [
        tools.read_file.to_spec(),
        tools.write_file.to_spec(),
        tools.run_command.to_spec(),
    ]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tool_schemas,
        system_prompt=SWE_AGENTIC_SYSTEM_PROMPT,
    )

    user_message = (
        f"## Issue in {repo} (instance: {instance_id})\n\n"
        f"{problem_statement}\n\n"
        f"The repository is checked out at `{workdir}`. "
        f"Please investigate and fix this issue. Start by exploring the "
        f"repository structure to understand the codebase."
    )
    return prefix + [{"role": "user", "content": user_message}]


def _build_repo_setup_script(repo: str, base_commit: str) -> str:
    """Build a bash script that clones and sets up the repository in the sandbox."""
    return f"""\
set -e
git clone --depth=50 https://github.com/{repo}.git /workspace/repo 2>/dev/null
cd /workspace/repo
git fetch --depth=1 origin {base_commit} 2>/dev/null || true
git checkout {base_commit} 2>/dev/null || true
pip install -e . 2>/dev/null || true
echo "REPO_SETUP_DONE"
"""


class SWEAgenticEnvGroupBuilder(EnvGroupBuilder):
    """EnvGroupBuilder that creates multi-turn SWE agentic environments with Modal sandboxes."""

    def __init__(
        self,
        instance_id: str,
        problem_statement: str,
        repo: str,
        base_commit: str,
        fail_to_pass: list[str],
        model_name: str,
        renderer_name: str | None,
        group_size: int,
        max_turns: int = 200,
        sandbox_timeout: int = 600,
        command_timeout: int = 120,
        test_timeout: int = 300,
        max_trajectory_tokens: int = 256 * 1024,
        max_generation_tokens: int | None = None,
        context_overflow_reward: float = 0.0,
        docker_image: str | None = None,
        r2e_test_files: list[tuple[str, str]] | None = None,
    ):
        self.instance_id = instance_id
        self.problem_statement = problem_statement
        self.repo = repo
        self.base_commit = base_commit
        self.fail_to_pass = fail_to_pass
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.group_size = group_size
        self.max_turns = max_turns
        self.sandbox_timeout = sandbox_timeout
        self.command_timeout = command_timeout
        self.test_timeout = test_timeout
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self.context_overflow_reward = context_overflow_reward
        self.docker_image = docker_image
        self.r2e_test_files = r2e_test_files or []
        self._sandboxes: list[SandboxInterface] = []

    async def make_envs(self) -> Sequence[Env]:
        self._sandboxes = []

        # Determine workdir and image based on whether we have a pre-built Docker image
        if self.docker_image:
            # R2E-Gym mode: use pre-built Docker image with repo at /testbed
            image = modal.Image.from_registry(self.docker_image)
            workdir = "/testbed"
        else:
            # Legacy mode: build from scratch
            image = (
                modal.Image.debian_slim()
                .apt_install("git")
                .pip_install("pytest", "pytest-timeout", "setuptools")
            )
            workdir = "/workspace/repo"

        # Create renderer
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        resolved_renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(resolved_renderer_name, tokenizer)

        # Build the repo setup script (only needed in legacy mode)
        setup_script = _build_repo_setup_script(self.repo, self.base_commit) if not self.docker_image else None

        envs: list[Env] = []
        for _ in range(self.group_size):
            sandbox = await ModalSandbox.create(
                app_name="nemotron-cascade-swe-agentic",
                image=image,
                timeout=self.sandbox_timeout,
            )
            self._sandboxes.append(sandbox)

            if setup_script:
                # Clone and set up the repository (legacy mode only)
                setup_result = await sandbox.run_command(
                    setup_script, timeout=self.sandbox_timeout,
                )
                if setup_result.exit_code != 0:
                    logger.warning(
                        "Repo setup failed for %s: %s",
                        self.instance_id, setup_result.stderr[:500],
                    )

            # Write R2E-Gym test files into the sandbox if present
            r2e_test_paths: list[str] = []
            if self.r2e_test_files:
                await sandbox.run_command(
                    f"mkdir -p {workdir}/r2e_tests", timeout=30,
                )
                for fname, code in self.r2e_test_files:
                    await sandbox.write_file(f"{workdir}/r2e_tests/{fname}", code)
                    r2e_test_paths.append(f"r2e_tests/{fname}")

            # Determine which tests to run for reward
            reward_test_files = r2e_test_paths if r2e_test_paths else self.fail_to_pass

            # Create tools and reward function for this sandbox
            tools = SWEAgenticTools(
                sandbox=sandbox,
                command_timeout=self.command_timeout,
                workdir=workdir,
            )
            reward_fn: RewardFn = SWEAgenticReward(
                sandbox=sandbox,
                fail_to_pass=reward_test_files,
                workdir=workdir,
                test_timeout=self.test_timeout,
            )

            # Build initial messages
            initial_messages = _build_initial_messages(
                problem_statement=self.problem_statement,
                repo=self.repo,
                instance_id=self.instance_id,
                renderer=renderer,
                tools=tools,
                workdir=workdir,
            )

            # Use build_agent_tool_env for multi-turn tool-use
            env = build_agent_tool_env(
                renderer=renderer,
                tools=[tools.read_file, tools.write_file, tools.run_command],
                initial_messages=initial_messages,
                reward_fn=reward_fn,
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                context_overflow_reward=self.context_overflow_reward,
            )
            envs.append(env)

        return envs

    async def cleanup(self) -> None:
        """Terminate all Modal sandboxes."""
        for sandbox in self._sandboxes:
            try:
                await sandbox.cleanup()
            except Exception as e:
                logger.warning("SWE agentic sandbox cleanup failed: %s", e)
        self._sandboxes.clear()

    def logging_tags(self) -> list[str]:
        return ["swe_agentic"]


class SWEAgenticDataset(RLDataset):
    """Dataset producing batches of SWEAgenticEnvGroupBuilders."""

    def __init__(
        self,
        builders: list[SWEAgenticEnvGroupBuilder],
        batch_size: int,
    ):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.builders))
        assert start < end
        return self.builders[start:end]

    def __len__(self) -> int:
        return math.ceil(len(self.builders) / self.batch_size)


@chz.chz
class SWEAgenticDatasetBuilder(RLDatasetBuilder):
    """Builder for the SWE agentic RL dataset.

    Loads SWE-bench instances and creates multi-turn agentic environments.
    When use_r2e_gym=True, uses R2E-Gym Docker images with pre-installed deps.
    Otherwise falls back to nvidia/Nemotron-Cascade-2-RL-data.
    """

    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int = 64
    max_turns: int = 200
    sandbox_timeout: int = 600
    command_timeout: int = 120
    test_timeout: int = 300
    max_trajectory_tokens: int = 256 * 1024
    max_generation_tokens: int | None = None
    context_overflow_reward: float = 0.0
    use_r2e_gym: bool = True
    seed: int = 0

    async def __call__(self) -> tuple[SWEAgenticDataset, None]:
        from datasets import load_dataset

        if self.use_r2e_gym:
            logger.info("Loading R2E-Gym data for agentic training...")
            ds = load_dataset("R2E-Gym/R2E-Gym-Subset", split="train")
            ds = cast(Dataset, ds)
            ds = ds.filter(
                lambda x: (
                    x.get("docker_image") is not None
                    and x.get("problem_statement") is not None
                )
            )
        else:
            logger.info("Loading SWE-RL data for agentic training...")
            ds = load_dataset("nvidia/Nemotron-Cascade-2-RL-data", name="SWE-RL", split="train")
            ds = cast(Dataset, ds)
            ds = ds.filter(
                lambda x: (
                    x.get("instance_id") is not None
                    and x.get("problem_statement") is not None
                    and x.get("repo") is not None
                    and x.get("base_commit") is not None
                )
            )

        ds = ds.shuffle(seed=self.seed)
        logger.info("SWE agentic dataset: %d valid instances (r2e_gym=%s)", len(ds), self.use_r2e_gym)

        builders: list[SWEAgenticEnvGroupBuilder] = []
        for row in ds:
            builder = self._make_builder(row)
            if builder is not None:
                builders.append(builder)

        return SWEAgenticDataset(builders=builders, batch_size=self.batch_size), None

    def _make_builder(self, row: dict) -> SWEAgenticEnvGroupBuilder | None:
        try:
            if self.use_r2e_gym:
                # R2E-Gym schema
                docker_image = row.get("docker_image", "")
                problem = row.get("problem_statement", "")
                repo = row.get("repo_name", "")
                base_commit = row.get("commit_hash", "")
                instance_id = row.get("instance_id", f"{repo}:{base_commit[:8]}")

                # Extract test files from execution_result_content
                r2e_test_files: list[tuple[str, str]] = []
                exec_content_raw = row.get("execution_result_content", "")
                if isinstance(exec_content_raw, str) and exec_content_raw:
                    try:
                        exec_content = json.loads(exec_content_raw)
                        test_names = exec_content.get("test_file_names", [])
                        test_codes = exec_content.get("test_file_codes", [])
                        for name, code in zip(test_names, test_codes):
                            r2e_test_files.append((name, code))
                    except json.JSONDecodeError:
                        pass

                fail_to_pass: list[str] = []  # Not used in R2E-Gym mode

                if not docker_image or not r2e_test_files:
                    return None
            else:
                # Legacy nvidia SWE-RL schema
                instance_id = row["instance_id"]
                problem = row.get("problem_statement", "")
                repo = row.get("repo", "")
                base_commit = row.get("base_commit", "")
                docker_image = None
                fail_to_pass = row.get("FAIL_TO_PASS", [])
                if isinstance(fail_to_pass, str):
                    try:
                        fail_to_pass = json.loads(fail_to_pass)
                    except json.JSONDecodeError:
                        fail_to_pass = [fail_to_pass]

                if not repo or not base_commit:
                    return None

            return SWEAgenticEnvGroupBuilder(
                instance_id=instance_id,
                problem_statement=problem,
                repo=repo,
                base_commit=base_commit,
                fail_to_pass=fail_to_pass,
                model_name=self.model_name_for_tokenizer,
                renderer_name=self.renderer_name,
                group_size=self.group_size,
                max_turns=self.max_turns,
                sandbox_timeout=self.sandbox_timeout,
                command_timeout=self.command_timeout,
                test_timeout=self.test_timeout,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                context_overflow_reward=self.context_overflow_reward,
                docker_image=docker_image,
                r2e_test_files=r2e_test_files if self.use_r2e_gym else None,
            )
        except Exception as e:
            logger.warning("Failed to parse SWE agentic row: %s", e)
            return None
