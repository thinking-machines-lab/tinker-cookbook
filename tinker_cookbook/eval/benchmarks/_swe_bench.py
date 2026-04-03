"""SWE-bench Verified benchmark -- multi-turn software engineering.

Dataset: ``princeton-nlp/SWE-bench_Verified`` on HuggingFace (500 problems).
Metric: Fraction of problems where all ``fail_to_pass`` tests pass after the
model's edits.
Pattern: Multi-turn ``MessageEnv`` + sandbox Tool + pytest grading.

Uses the cookbook's ``MessageEnv`` / ``EnvFromMessageEnv`` pattern so that
the renderer handles thinking-token stripping and prompt construction
automatically. The model interacts with the repository via a ``bash`` tool.

Requires a sandbox backend (Modal) for command execution. The sandbox image
must include git, python3, and pip.
"""

from __future__ import annotations

import json
import logging
import shlex
from collections.abc import Sequence
from typing import Annotated, Any, cast

from datasets import Dataset

from tinker_cookbook.eval.benchmarks._common import (
    SandboxMixin,
    get_sandbox_factory,
    limit_dataset,
    load_benchmark_dataset,
    make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import (
    BenchmarkBuilder,
    BenchmarkConfig,
    BenchmarkResult,
)
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.types import Env
from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.tool_use.agent_tool_message_env import AgentToolMessageEnv
from tinker_cookbook.tool_use.tools import simple_tool_result, tool
from tinker_cookbook.tool_use.types import ToolResult

logger = logging.getLogger(__name__)

MAX_TURNS = 100
"""Maximum number of agent turns before forced termination."""

_SYSTEM_PROMPT = """\
You are a helpful assistant that can interact with a computer shell to solve \
programming tasks.

You're a software engineer fixing an issue in a code repository. Your task is \
to make changes to source files to fix the issue described in the problem statement.

For each response:
1. Include reasoning explaining what you're trying to accomplish
2. Use the bash tool to execute commands

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix by running your reproduction script again
5. Test edge cases to ensure your fix is robust

## Important Rules

- The repository is at /workspace/repo — use `cd /workspace/repo && ...` in each command
- MODIFY only source code files, NOT tests or configuration files
- Use non-interactive commands only (no vi, nano, etc.)
- Use sed, awk, or python for file editing
- When you are confident the fix is complete, stop calling tools

## Environment

- Full Linux shell with git, grep, find, python3, sed, etc.
- PAGER=cat (no interactive paging)
- Each command runs in a subshell — cd is not persistent between commands"""


def _parse_test_ids(raw: str | list) -> list[str]:
    """Parse ``fail_to_pass`` / ``pass_to_pass`` which may be a JSON string or list."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        if raw.strip():
            return [raw.strip()]
    return []


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class _SWEBashTool:
    """Bash tool for SWE-bench — executes commands in the repo sandbox."""

    def __init__(self, sandbox: SandboxInterface) -> None:
        self._sandbox = sandbox

    @tool
    async def bash(
        self,
        command: Annotated[str, "The bash command to execute."],
    ) -> ToolResult:
        """Execute a bash command in the repository sandbox."""
        # Wrap command with env vars to prevent interactive pagers
        wrapped = f"PAGER=cat MANPAGER=cat PIP_PROGRESS_BAR=off TQDM_DISABLE=1 {command}"
        result = await self._sandbox.run_command(
            wrapped, workdir="/workspace/repo", timeout=120, max_output_bytes=16000
        )
        stdout = result.stdout
        stderr = result.stderr

        # Truncate long output with a warning (matches mini-swe-agent behavior)
        max_chars = 10000
        if len(stdout) > max_chars:
            head = stdout[:5000]
            tail = stdout[-5000:]
            elided = len(stdout) - max_chars
            stdout = (
                f"{head}\n\n[... {elided} characters elided ...]\n\n{tail}\n"
                "[Output truncated. Use head/tail/grep for smaller output.]"
            )
        if len(stderr) > 4000:
            stderr = stderr[:4000] + "\n[stderr truncated]"

        output = json.dumps({"exit_code": result.exit_code, "stdout": stdout, "stderr": stderr})
        return simple_tool_result(output)


# ---------------------------------------------------------------------------
# Reward function — runs fail_to_pass tests
# ---------------------------------------------------------------------------


class _SWEBenchReward:
    """Reward function that runs fail_to_pass tests via pytest."""

    def __init__(
        self,
        sandbox: SandboxInterface,
        fail_to_pass: list[str],
        instance_id: str,
    ) -> None:
        self._sandbox = sandbox
        self._fail_to_pass = fail_to_pass
        self._instance_id = instance_id

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        """Grade by running fail_to_pass tests. All must pass for reward=1."""
        if not self._fail_to_pass:
            return 0.0, {"no_tests": 1.0}

        # Run all tests in a single pytest invocation for efficiency
        test_args = " ".join(shlex.quote(t) for t in self._fail_to_pass)
        try:
            result = await self._sandbox.run_command(
                f"cd /workspace/repo && python -m pytest {test_args} --tb=short 2>&1",
                timeout=120,
            )
            all_passed = result.exit_code == 0
        except Exception as e:
            logger.warning(f"swe_bench test error: {e}")
            all_passed = False

        num_turns = sum(1 for msg in history if msg.get("role") == "assistant")

        return (
            1.0 if all_passed else 0.0,
            {
                "correct": float(all_passed),
                "num_turns": float(num_turns),
                "num_tests_total": float(len(self._fail_to_pass)),
            },
        )


# ---------------------------------------------------------------------------
# Env factory (creates sandbox + MessageEnv on first observation)
# ---------------------------------------------------------------------------


class _SWEBenchEnvFactory(SandboxMixin, Env):
    """Wrapper that creates sandbox, clones repo, and delegates to EnvFromMessageEnv."""

    def __init__(
        self,
        *,
        repo: str,
        base_commit: str,
        problem_statement: str,
        hints_text: str,
        instance_id: str,
        fail_to_pass: list[str],
        test_patch: str,
        sandbox_factory: Any,
        renderer: Renderer,
        example_id: str,
        system_prompt: str | None = None,
        max_trajectory_tokens: int | None = None,
        max_generation_tokens: int | None = None,
    ):
        self.repo = repo
        self.base_commit = base_commit
        self.problem_statement = problem_statement
        self.hints_text = hints_text
        self.instance_id = instance_id
        self.fail_to_pass = fail_to_pass
        self.test_patch = test_patch
        self.sandbox_factory = sandbox_factory
        self.renderer = renderer
        self.example_id = example_id
        self.system_prompt = system_prompt
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens

        self._inner: EnvFromMessageEnv | None = None

    async def initial_observation(self):
        # Create sandbox
        self.sandbox = await self.sandbox_factory()
        assert self.sandbox is not None

        # Clone and checkout
        try:
            repo_url = f"https://github.com/{shlex.quote(self.repo)}.git"
            safe_commit = shlex.quote(self.base_commit)
            setup_cmds = [
                f"git clone --depth=50 {repo_url} /workspace/repo",
                f"cd /workspace/repo && git fetch --depth=50 origin {safe_commit}",
                f"cd /workspace/repo && git checkout {safe_commit}",
            ]
            for cmd in setup_cmds:
                result = await self.sandbox.run_command(cmd, timeout=300)
                if result.exit_code != 0:
                    raise RuntimeError(
                        f"swe_bench setup failed (exit {result.exit_code}): "
                        f"{cmd[:100]}  stderr={result.stderr[:300]}"
                    )
        except Exception:
            await self.cleanup()
            raise

        # Apply test patch
        if self.test_patch.strip():
            try:
                await self.sandbox.write_file("/workspace/test_patch.diff", self.test_patch)
                result = await self.sandbox.run_command(
                    "cd /workspace/repo && git apply /workspace/test_patch.diff",
                    timeout=60,
                )
                if result.exit_code != 0:
                    await self.sandbox.run_command(
                        "cd /workspace/repo && git apply --3way /workspace/test_patch.diff",
                        timeout=60,
                    )
            except Exception as e:
                logger.warning(f"swe_bench: failed to apply test patch: {e}")

        # Create tool and reward
        bash_tool = _SWEBashTool(self.sandbox)
        reward_fn = _SWEBenchReward(self.sandbox, self.fail_to_pass, self.instance_id)

        # Build initial messages
        system_content = self.system_prompt or _SYSTEM_PROMPT
        hints_section = ""
        if self.hints_text.strip():
            hints_section = f"\n\n## Hints\n{self.hints_text[:2000]}"

        user_prompt = (
            f"## Repository: {self.repo}\n\n"
            f"## Problem Statement\n{self.problem_statement}"
            f"{hints_section}\n\n"
            f"The repository is checked out at `/workspace/repo`. "
            f"Explore the codebase, find and fix the issue."
        )

        # Build initial messages with tool specs in the renderer's native format
        tool_specs = [bash_tool.bash.to_spec()]
        initial_messages = self.renderer.create_conversation_prefix_with_tools(
            tools=tool_specs,
            system_prompt=system_content,
        )
        initial_messages.append({"role": "user", "content": user_prompt})

        # Create inner MessageEnv + adapter
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
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_generation_tokens=self.max_generation_tokens,
            context_overflow_reward=0.0,  # Treat as failure, not penalty
        )

        return await self._inner.initial_observation()

    async def step(self, action, *, extra=None):
        assert self._inner is not None
        result = await self._inner.step(action, extra=extra)
        if result.episode_done:
            result.logs["example_id"] = self.example_id
            result.logs["instance_id"] = self.instance_id
            result.logs["repo"] = self.repo
        return result


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class SWEBenchBenchmarkBuilder(BenchmarkBuilder):
    """SWE-bench Verified: multi-turn software engineering.

    Uses the cookbook's ``MessageEnv`` / ``EnvFromMessageEnv`` pattern.
    The model gets shell access via a ``bash`` tool and the renderer
    handles tool-call parsing and thinking-token stripping.

    Requires a sandbox backend (Modal) with git, python3, pip installed.
    """

    name = "swe_bench"
    requires_sandbox = True
    multi_turn = True
    recommended_timeout = 1800

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = cast(Dataset, load_benchmark_dataset("princeton-nlp/SWE-bench_Verified"))
        ds = limit_dataset(ds, config.max_examples)

        sandbox_factory = get_sandbox_factory(
            config, packages=["git", "python3", "python3-pip", "python3-venv"]
        )

        envs: list[Env] = []
        for row in ds:
            row = dict(row)
            instance_id = row.get("instance_id", "")
            repo = row.get("repo", "unknown")
            problem_statement = row.get("problem_statement", "")
            hints_text = row.get("hints_text", "")
            base_commit = row.get("base_commit", "")
            test_patch = row.get("test_patch", "")

            fail_to_pass_raw = row.get("fail_to_pass", row.get("FAIL_TO_PASS", "[]"))
            fail_to_pass = _parse_test_ids(fail_to_pass_raw)

            if not problem_statement:
                logger.warning(f"swe_bench: skipping {instance_id} — no problem statement")
                continue
            if not base_commit:
                logger.warning(f"swe_bench: skipping {instance_id} — no base_commit")
                continue

            example_id = (
                f"swe_bench_{instance_id}"
                if instance_id
                else make_example_id("swe_bench", problem_statement)
            )

            envs.append(
                _SWEBenchEnvFactory(
                    repo=repo,
                    base_commit=base_commit,
                    problem_statement=problem_statement,
                    hints_text=hints_text,
                    instance_id=instance_id,
                    fail_to_pass=fail_to_pass,
                    test_patch=test_patch,
                    sandbox_factory=sandbox_factory,
                    renderer=renderer,
                    example_id=example_id,
                    system_prompt=config.system_prompt,
                    max_trajectory_tokens=config.max_trajectory_tokens,
                    max_generation_tokens=config.max_generation_tokens,
                )
            )

        return envs

    def aggregate(self, rewards: list[float], metrics_list: list[dict]) -> BenchmarkResult:
        """Aggregate with resolve rate and per-instance metrics."""
        num_correct = sum(1 for r in rewards if r > 0)
        resolve_rate = num_correct / len(rewards) if rewards else 0.0

        avg_turns = (
            sum(m.get("num_turns", 0) for m in metrics_list) / len(metrics_list)
            if metrics_list
            else 0.0
        )

        return BenchmarkResult(
            name=self.name,
            score=resolve_rate,
            num_examples=len(rewards),
            num_correct=num_correct,
            metrics={
                "swe_bench/resolve_rate": resolve_rate,
                "swe_bench/avg_turns": avg_turns,
            },
        )


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(SWEBenchBenchmarkBuilder())
