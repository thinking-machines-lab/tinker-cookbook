"""SWE-bench Verified benchmark -- multi-turn software engineering.

Dataset: ``princeton-nlp/SWE-bench_Verified`` on HuggingFace (500 problems).
Metric: Fraction of problems where all ``fail_to_pass`` tests pass after the
model's edits.
Pattern: Multi-turn generate + sandbox execution + test grading.

The model gets shell access to the repository at the correct base commit and
iterates (explore, edit, test) until it declares TASK_COMPLETE or hits the
turn limit.  Grading runs the ``fail_to_pass`` tests via pytest; all must
pass for a score of 1.0.

Requires a sandbox backend (Modal) for command execution.
"""

from __future__ import annotations

import json
import logging
import shlex
from collections.abc import Sequence
from typing import cast

import tinker
from datasets import Dataset

from tinker_cookbook.eval.benchmarks._common import (
    SandboxMixin,
    extract_command,
    get_sandbox_factory,
    is_task_complete,
    limit_dataset,
    load_benchmark_dataset,
    make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import (
    BenchmarkBuilder,
    BenchmarkConfig,
    BenchmarkResult,
)
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult

logger = logging.getLogger(__name__)

MAX_TURNS = 40
"""Maximum number of agent turns before forced termination."""

_SYSTEM_PROMPT = """\
You are an expert software engineer. You have shell access to a repository \
checked out at the relevant commit. Your task is to diagnose and fix the \
issue described below.

To execute a command, write it in a ```bash code block:
```bash
your command here
```

You will see the command's stdout and stderr. You can run multiple commands \
across multiple turns — explore the codebase, understand the bug, edit files, \
and verify your fix.

When you are confident the fix is complete, write TASK_COMPLETE.

Important:
- The repository is at /workspace/repo
- Your working directory is /workspace/repo
- Use standard tools: grep, find, cat, sed, python, git diff, etc.
- Test your changes before declaring completion"""


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
        # Might be a single test id
        if raw.strip():
            return [raw.strip()]
    return []


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class SWEBenchEnv(SandboxMixin, Env):
    """Multi-turn env for one SWE-bench problem.

    On each turn:
    1. Parse the model's response for a bash command
    2. Execute the command in the sandbox
    3. Return stdout/stderr as the next observation
    4. On completion or max turns, run ``fail_to_pass`` tests and grade
    """

    def __init__(
        self,
        *,
        repo: str,
        base_commit: str,
        problem_statement: str,
        hints_text: str,
        instance_id: str,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
        test_patch: str,
        sandbox_factory,
        renderer: Renderer,
        example_id: str = "",
    ):
        self.repo = repo
        self.base_commit = base_commit
        self.problem_statement = problem_statement
        self.hints_text = hints_text
        self.instance_id = instance_id
        self.fail_to_pass = fail_to_pass
        self.pass_to_pass = pass_to_pass
        self.test_patch = test_patch
        self.sandbox_factory = sandbox_factory
        self.renderer = renderer
        self.example_id = example_id

        # Runtime state
        self.messages: list[Message] = []
        self.turn_count = 0
        self.commands_executed: list[str] = []

    async def initial_observation(self):
        # Create sandbox
        self.sandbox = await self.sandbox_factory()

        # Clone the repository and checkout the base commit.
        # These are critical — if they fail, cleanup and re-raise.
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
                        f"swe_bench setup command failed (exit {result.exit_code}): "
                        f"{cmd[:100]}  stderr={result.stderr[:300]}"
                    )
        except Exception:
            await self.cleanup()
            raise

        # Apply test patch if present (adds/modifies test files needed for grading).
        # Non-critical — warn and continue if it fails.
        if self.test_patch.strip():
            try:
                await self.sandbox.write_file("/workspace/test_patch.diff", self.test_patch)
                result = await self.sandbox.run_command(
                    "cd /workspace/repo && git apply /workspace/test_patch.diff",
                    timeout=60,
                )
                if result.exit_code != 0:
                    # Try with --3way as fallback
                    await self.sandbox.run_command(
                        "cd /workspace/repo && git apply --3way /workspace/test_patch.diff",
                        timeout=60,
                    )
            except Exception as e:
                logger.warning(f"swe_bench: failed to apply test patch: {e}")

        # Build initial prompt with problem statement
        hints_section = ""
        if self.hints_text.strip():
            hints_section = f"\n\n## Hints\n{self.hints_text[:2000]}"

        user_prompt = (
            f"## Repository: {self.repo}\n\n"
            f"## Problem Statement\n{self.problem_statement}"
            f"{hints_section}\n\n"
            f"The repository is checked out at `/workspace/repo`. "
            f"Explore the codebase, find and fix the issue, then write TASK_COMPLETE."
        )

        self.messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        model_input = self.renderer.build_generation_prompt(self.messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        self.turn_count += 1
        response_text = self.renderer.tokenizer.decode(action)

        self.messages.append({"role": "assistant", "content": response_text})

        # Check if task is complete or turn limit reached
        if is_task_complete(response_text) or self.turn_count >= MAX_TURNS:
            return await self._finalize()

        # Extract and execute command
        command = extract_command(response_text)
        if command is None:
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
                command,
                workdir="/workspace/repo",
                timeout=120,
            )
            output_parts = []
            if result.stdout.strip():
                output_parts.append(f"stdout:\n{result.stdout[:8000]}")
            if result.stderr.strip():
                output_parts.append(f"stderr:\n{result.stderr[:4000]}")
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
        """Run fail_to_pass tests and grade the result."""
        all_passed = False
        test_output = ""
        num_tests_passed = 0
        num_tests_total = len(self.fail_to_pass)

        if self.sandbox and self.fail_to_pass:
            passed_tests = []
            failed_tests = []

            for test_id in self.fail_to_pass:
                try:
                    result = await self.sandbox.run_command(
                        f"cd /workspace/repo && python -m pytest {shlex.quote(test_id)} -x --tb=short 2>&1",
                        timeout=120,
                    )
                    output = (result.stdout + "\n" + result.stderr).strip()
                    if result.exit_code == 0:
                        passed_tests.append(test_id)
                    else:
                        failed_tests.append(test_id)
                        # Capture first failure output for logs
                        if not test_output:
                            test_output = output[:2000]
                except Exception as e:
                    failed_tests.append(test_id)
                    if not test_output:
                        test_output = f"Test execution error for {test_id}: {e}"

            num_tests_passed = len(passed_tests)
            all_passed = len(failed_tests) == 0

            if all_passed:
                test_output = f"All {num_tests_total} fail_to_pass tests now pass."
            elif not test_output:
                test_output = (
                    f"{len(failed_tests)}/{num_tests_total} tests still failing: {failed_tests[:3]}"
                )

        elif not self.fail_to_pass:
            test_output = "No fail_to_pass tests defined for this instance."

        # Cleanup sandbox
        await self.cleanup()

        reward = 1.0 if all_passed else 0.0
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={
                "correct": float(all_passed),
                "num_turns": self.turn_count,
                "num_commands": len(self.commands_executed),
                "num_tests_passed": num_tests_passed,
                "num_tests_total": num_tests_total,
            },
            logs={
                "example_id": self.example_id,
                "instance_id": self.instance_id,
                "repo": self.repo,
                "problem_statement": self.problem_statement[:200],
                "test_output": test_output[:500],
                "num_turns": self.turn_count,
                "commands": "\n".join(self.commands_executed[-5:])[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class SWEBenchBenchmarkBuilder(BenchmarkBuilder):
    """SWE-bench Verified: multi-turn software engineering.

    The model gets shell access to a repository at the correct commit and
    iterates to fix the issue. Graded by running the ``fail_to_pass`` tests —
    all must pass for a score of 1.0.

    Requires a sandbox backend (Modal).
    """

    name = "swe_bench"
    requires_sandbox = True
    multi_turn = True
    recommended_timeout = 1800

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = cast(Dataset, load_benchmark_dataset("princeton-nlp/SWE-bench_Verified"))
        ds = limit_dataset(ds, config.max_examples)

        sandbox_factory = get_sandbox_factory(config)

        envs = []
        for row in ds:
            instance_id = row.get("instance_id", "")
            repo = row.get("repo", "unknown")
            problem_statement = row.get("problem_statement", "")
            hints_text = row.get("hints_text", "")
            base_commit = row.get("base_commit", "")
            test_patch = row.get("test_patch", "")

            # fail_to_pass / pass_to_pass may appear under either casing
            fail_to_pass_raw = row.get("fail_to_pass", row.get("FAIL_TO_PASS", "[]"))
            pass_to_pass_raw = row.get("pass_to_pass", row.get("PASS_TO_PASS", "[]"))

            fail_to_pass = _parse_test_ids(fail_to_pass_raw)
            pass_to_pass = _parse_test_ids(pass_to_pass_raw)

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
                SWEBenchEnv(
                    repo=repo,
                    base_commit=base_commit,
                    problem_statement=problem_statement,
                    hints_text=hints_text,
                    instance_id=instance_id,
                    fail_to_pass=fail_to_pass,
                    pass_to_pass=pass_to_pass,
                    test_patch=test_patch,
                    sandbox_factory=sandbox_factory,
                    renderer=renderer,
                    example_id=example_id,
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
        avg_commands = (
            sum(m.get("num_commands", 0) for m in metrics_list) / len(metrics_list)
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
                "swe_bench/avg_commands": avg_commands,
            },
        )


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(SWEBenchBenchmarkBuilder())
