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

MAX_TURNS = 40
"""Maximum number of agent turns before forced termination."""

_SYSTEM_PROMPT = """\
You are an expert software engineer. You have shell access to a repository \
checked out at the relevant commit. Your task is to diagnose and fix the \
issue described below.

Use the bash tool to explore the codebase, understand the bug, edit files, \
and verify your fix. When you are confident the fix is complete, stop calling tools.

Important:
- The repository is at /workspace/repo
- Your working directory is /workspace/repo
- Use standard tools: grep, find, cat, sed, python, git diff, etc.
- Test your changes before stopping"""


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
        result = await self._sandbox.run_command(
            command, workdir="/workspace/repo", timeout=120, max_output_bytes=16000
        )
        output = json.dumps(
            {
                "exit_code": result.exit_code,
                "stdout": result.stdout[:8000],
                "stderr": result.stderr[:4000],
            }
        )
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

        passed_tests: list[str] = []
        failed_tests: list[str] = []

        for test_id in self._fail_to_pass:
            try:
                result = await self._sandbox.run_command(
                    f"cd /workspace/repo && python -m pytest {shlex.quote(test_id)} -x --tb=short 2>&1",
                    timeout=120,
                )
                if result.exit_code == 0:
                    passed_tests.append(test_id)
                else:
                    failed_tests.append(test_id)
            except Exception as e:
                failed_tests.append(test_id)
                logger.warning(f"swe_bench test error for {test_id}: {e}")

        all_passed = len(failed_tests) == 0
        num_turns = sum(1 for msg in history if msg.get("role") == "assistant")

        return (
            1.0 if all_passed else 0.0,
            {
                "correct": float(all_passed),
                "num_turns": float(num_turns),
                "num_tests_passed": float(len(passed_tests)),
                "num_tests_total": float(len(self._fail_to_pass)),
            },
        )


# ---------------------------------------------------------------------------
# Env factory (creates sandbox + MessageEnv on first observation)
# ---------------------------------------------------------------------------


class _SWEBenchEnvFactory(Env):
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
                result = await self._sandbox.run_command(cmd, timeout=300)
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
                await self._sandbox.write_file("/workspace/test_patch.diff", self.test_patch)
                result = await self._sandbox.run_command(
                    "cd /workspace/repo && git apply /workspace/test_patch.diff",
                    timeout=60,
                )
                if result.exit_code != 0:
                    await self._sandbox.run_command(
                        "cd /workspace/repo && git apply --3way /workspace/test_patch.diff",
                        timeout=60,
                    )
            except Exception as e:
                logger.warning(f"swe_bench: failed to apply test patch: {e}")

        # Create tool and reward
        bash_tool = _SWEBashTool(self._sandbox)
        reward_fn = _SWEBenchReward(self._sandbox, self.fail_to_pass, self.instance_id)

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

        initial_messages: list[Message] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ]

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

        sandbox_factory = get_sandbox_factory(config)

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
