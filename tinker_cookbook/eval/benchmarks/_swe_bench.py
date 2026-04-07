"""SWE-bench Verified benchmark -- multi-turn software engineering.

Dataset: ``princeton-nlp/SWE-bench_Verified`` on HuggingFace (500 problems).
Metric: Fraction of problems where all ``fail_to_pass`` tests pass after the
model's edits (using the official SWE-bench eval harness).
Pattern: Multi-turn ``MessageEnv`` + sandbox Tool + official eval grading.

Uses the cookbook's ``MessageEnv`` / ``EnvFromMessageEnv`` pattern so that
the renderer handles thinking-token stripping and prompt construction
automatically. The model interacts with the repository via a ``bash`` tool.

Grading uses the official ``swebench`` eval scripts which include:
- Per-repo test runners (Django's ``runtests.py``, sympy's ``bin/test``, etc.)
- Re-install after model edits (for C extensions, setup.py changes)
- Test patch application at grading time (not during agent interaction)
- Per-repo log parsers for test result extraction

Requires a sandbox backend (Modal) for command execution. The sandbox image
has miniconda pre-installed to match the official SWE-bench base environment.
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
    limit_dataset,
    load_benchmark_dataset,
    make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import (
    BenchmarkBuilder,
    BenchmarkConfig,
    BenchmarkResult,
    Metrics,
)
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.types import Env
from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.tool_use.agent_tool_message_env import AgentToolMessageEnv
from tinker_cookbook.tool_use.tools import simple_tool_result, tool
from tinker_cookbook.tool_use.types import ToolResult

logger = logging.getLogger(__name__)

MAX_TURNS = 200
"""Maximum number of agent turns before forced termination."""

# apply_patch helper — GPT-OSS models use this instead of sed/python for edits.
# Reads a patch from stdin in the *** Begin Patch / *** Update File format.
_APPLY_PATCH_SCRIPT = r'''#!/usr/bin/env python3
"""apply_patch: apply a GPT-style patch from stdin.

Compatible with Python 3.5+ (no f-strings) since conda envs may use old Python.
"""
import sys, os

def apply_patch(patch_text, workdir="/workspace/repo"):
    current_file = None
    old_lines = []
    new_lines = []
    results = []

    def flush_hunk():
        if current_file is None or (not old_lines and not new_lines):
            del old_lines[:]
            del new_lines[:]
            return
        fpath = os.path.join(workdir, current_file)
        if not os.path.exists(fpath):
            if new_lines:
                d = os.path.dirname(fpath)
                if d and not os.path.exists(d):
                    os.makedirs(d)
                with open(fpath, "w") as f:
                    f.write("\n".join(new_lines) + "\n")
                results.append("Created " + current_file)
            del old_lines[:]
            del new_lines[:]
            return
        with open(fpath) as f:
            content = f.read()
        old_text = "\n".join(old_lines)
        new_text = "\n".join(new_lines)
        if old_text in content:
            content = content.replace(old_text, new_text, 1)
            with open(fpath, "w") as f:
                f.write(content)
            results.append("Updated " + current_file)
        else:
            results.append("WARNING: Could not find match in " + current_file)
        del old_lines[:]
        del new_lines[:]

    for line in patch_text.split("\n"):
        if line.startswith("*** Begin Patch"):
            continue
        if line.startswith("*** End Patch"):
            flush_hunk()
            continue
        if line.startswith("*** Update File:"):
            flush_hunk()
            current_file = line.split("*** Update File:")[-1].strip()
            continue
        if line.startswith("*** Add File:"):
            flush_hunk()
            current_file = line.split("*** Add File:")[-1].strip()
            continue
        if line.startswith("*** Delete File:"):
            flush_hunk()
            fpath = os.path.join(workdir, line.split("*** Delete File:")[-1].strip())
            if os.path.exists(fpath):
                os.remove(fpath)
                results.append("Deleted " + fpath)
            current_file = None
            continue
        if line.startswith("@@"):
            flush_hunk()
            continue
        if line.startswith("-"):
            old_lines.append(line[1:])
        elif line.startswith("+"):
            new_lines.append(line[1:])
        else:
            old_lines.append(line.lstrip(" "))
            new_lines.append(line.lstrip(" "))

    flush_hunk()
    return results

patch_text = sys.stdin.read()
results = apply_patch(patch_text)
for r in results:
    print(r)
if any("WARNING" in r for r in results):
    sys.exit(1)
'''

_SYSTEM_PROMPT = """\
You are an autonomous software engineer. Your task is to fix a bug in a code \
repository by editing source files.

IMPORTANT: You MUST call the bash tool in EVERY response. If you respond \
without a tool call, the task will end immediately. Keep working until you \
have fixed the issue.

## Workflow

1. **Explore**: Find relevant files with grep and find
2. **Reproduce**: Write and run a script that triggers the bug
3. **Edit**: Fix the source code using sed or a python script
4. **Verify**: Run your reproduction script to confirm the fix
5. **Test**: Run the test suite to check for regressions

## Editing files

Use sed for small edits:
```
cd /workspace/repo && sed -i 's/old_text/new_text/g' path/to/file.py
```

Use a python script for larger edits:
```
cd /workspace/repo && python3 -c "
content = open('path/to/file.py').read()
content = content.replace('old_text', 'new_text')
open('path/to/file.py', 'w').write(content)
"
```

Use cat with heredoc to create new files:
```
cat <<'EOF' > /workspace/repo/path/to/new_file.py
import foo
EOF
```

## Rules

- The repository is at /workspace/repo
- ALWAYS prefix commands with `cd /workspace/repo &&`
- MODIFY only source code files, NOT tests or configuration
- Each command runs in a fresh subshell — cd is not persistent
- PAGER=cat (no interactive paging)
- Do NOT use interactive editors (vi, nano, etc.)"""


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
        # Activate conda env and set env vars to prevent interactive pagers.
        # The setup script creates a 'testbed' conda env with the correct
        # Python version and dependencies for this instance.
        wrapped = (
            "source /opt/miniconda3/bin/activate && conda activate testbed && "
            f"PAGER=cat MANPAGER=cat PIP_PROGRESS_BAR=off TQDM_DISABLE=1 {command}"
        )
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
# Reward function — uses official SWE-bench eval script and grading
# ---------------------------------------------------------------------------


class _SWEBenchReward:
    """Reward function using the official SWE-bench eval harness.

    Runs the per-instance eval script (which includes the correct test runner,
    re-install step, and test patch application) and parses results with the
    official log parsers.
    """

    def __init__(
        self,
        sandbox: SandboxInterface,
        eval_script: str,
        raw_instance: dict,
        instance_id: str,
    ) -> None:
        self._sandbox = sandbox
        self._eval_script = eval_script
        self._raw_instance = raw_instance
        self._instance_id = instance_id

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        """Grade by running the official eval script and parsing results."""
        from swebench.harness.constants import (
            APPLY_PATCH_FAIL,
            RESET_FAILED,
            TESTS_ERROR,
            TESTS_TIMEOUT,
        )
        from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
        from swebench.harness.test_spec.test_spec import make_test_spec

        num_turns = sum(1 for msg in history if msg.get("role") == "assistant")

        try:
            # Write and run the official eval script
            await self._sandbox.write_file("/workspace/eval.sh", self._eval_script)
            result = await self._sandbox.run_command(
                "bash /workspace/eval.sh 2>&1", timeout=300, max_output_bytes=512_000
            )
            test_output = result.stdout + "\n" + result.stderr
            logger.debug(
                f"swe_bench eval {self._instance_id}: exit={result.exit_code} "
                f"stdout={len(result.stdout)}B stderr={len(result.stderr)}B"
            )
        except Exception as e:
            logger.warning(f"swe_bench eval error for {self._instance_id}: {e}")
            return 0.0, {"correct": 0.0, "num_turns": float(num_turns), "eval_error": 1.0}

        # Check for error markers
        for marker in [APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT]:
            if marker in test_output:
                logger.debug(f"swe_bench {self._instance_id}: eval marker {marker}")
                return 0.0, {
                    "correct": 0.0,
                    "num_turns": float(num_turns),
                    "eval_marker": 1.0,
                }

        # Parse test results with the official per-repo log parser
        repo = self._raw_instance.get("repo", "")
        parser = MAP_REPO_TO_PARSER.get(repo)
        if parser is None:
            logger.warning(f"swe_bench: no parser for repo {repo}")
            return 0.0, {"correct": 0.0, "num_turns": float(num_turns), "no_parser": 1.0}

        spec = make_test_spec(self._raw_instance)

        # Extract test output between markers
        start_marker = ">>>>> Start Test Output"
        end_marker = ">>>>> End Test Output"
        if start_marker in test_output and end_marker in test_output:
            test_section = test_output.split(start_marker)[1].split(end_marker)[0]
        else:
            # Fallback: parse entire output
            test_section = test_output

        eval_status = dict(parser(test_section, spec))
        logger.debug(
            f"swe_bench eval {self._instance_id}: "
            f"parsed={len(eval_status)} tests from {len(test_section)}B output"
        )

        # Grade using official logic
        fail_to_pass = _parse_test_ids(
            self._raw_instance.get("fail_to_pass", self._raw_instance.get("FAIL_TO_PASS", "[]"))
        )
        pass_to_pass = _parse_test_ids(
            self._raw_instance.get("pass_to_pass", self._raw_instance.get("PASS_TO_PASS", "[]"))
        )

        # Check fail_to_pass: all must now pass
        f2p_passed = all(eval_status.get(t) in ("PASSED", "XFAIL") for t in fail_to_pass)
        # Check pass_to_pass: all must still pass
        p2p_passed = all(eval_status.get(t) in ("PASSED", "XFAIL") for t in pass_to_pass)
        resolved = f2p_passed and p2p_passed and len(fail_to_pass) > 0

        return (
            1.0 if resolved else 0.0,
            {
                "correct": float(resolved),
                "num_turns": float(num_turns),
                "f2p_total": float(len(fail_to_pass)),
                "f2p_passed": float(sum(1 for t in fail_to_pass if eval_status.get(t) in ("PASSED", "XFAIL"))),
                "p2p_total": float(len(pass_to_pass)),
                "p2p_passed": float(sum(1 for t in pass_to_pass if eval_status.get(t) in ("PASSED", "XFAIL"))),
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
        raw_instance: dict | None = None,
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
        self.raw_instance = raw_instance
        self.system_prompt = system_prompt
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens

        self._inner: EnvFromMessageEnv | None = None

    def _make_setup_and_eval(self) -> tuple[list[str], str]:
        """Generate setup commands and eval script using official SWE-bench TestSpec.

        Uses ``swebench.harness.test_spec.make_test_spec`` to get per-instance
        setup scripts and eval scripts with exact dependency versions and the
        correct per-repo test runner.

        Returns:
            Tuple of (setup_commands, eval_script). The eval script includes
            re-install, test patch application, and the correct test runner.
        """
        from swebench.harness.test_spec.test_spec import make_test_spec

        if self.raw_instance is None:
            raise ValueError("raw_instance not available")

        spec = make_test_spec(self.raw_instance)

        # env_script_list first (creates conda env + installs deps),
        # then repo_script_list (clones repo, checks out commit, installs project).
        cmds: list[str] = []
        for cmd in spec.env_script_list:
            cmds.append(cmd.replace("/testbed", "/workspace/repo"))
        for cmd in spec.repo_script_list:
            cmds.append(cmd.replace("/testbed", "/workspace/repo"))

        # The eval script handles: re-install, test patch application,
        # running the correct per-repo test runner, and result markers.
        eval_script = spec.eval_script.replace("/testbed", "/workspace/repo")

        return cmds, eval_script

    def _make_basic_setup_commands(self) -> list[str]:
        """Fallback setup without swebench package."""
        repo_url = f"https://github.com/{shlex.quote(self.repo)}.git"
        safe_commit = shlex.quote(self.base_commit)
        return [
            f"git clone --depth=50 {repo_url} /workspace/repo",
            f"cd /workspace/repo && git fetch --depth=50 origin {safe_commit}",
            f"cd /workspace/repo && git checkout {safe_commit}",
            "cd /workspace/repo && pip install -e . 2>&1 | tail -5",
        ]

    async def initial_observation(self):
        # Create sandbox
        self.sandbox = await self.sandbox_factory()
        assert self.sandbox is not None

        # Use official SWE-bench test spec for environment setup when available.
        # The eval_script is stored for grading — it handles re-install, test
        # patch application, and running the correct per-repo test runner.
        eval_script: str | None = None
        try:
            setup_cmds, eval_script = self._make_setup_and_eval()
        except Exception as e:
            logger.debug(f"swe_bench: TestSpec unavailable ({e}), using basic setup")
            setup_cmds = self._make_basic_setup_commands()

        # Run all setup commands as a single script so conda env persists
        # across commands (each run_command is a separate shell invocation).
        setup_script = "#!/bin/bash\nset -e\n" + "\n".join(setup_cmds)
        try:
            await self.sandbox.write_file("/workspace/setup.sh", setup_script)
            result = await self.sandbox.run_command(
                "bash /workspace/setup.sh", timeout=600
            )
            if result.exit_code != 0:
                logger.warning(
                    f"swe_bench setup failed (exit {result.exit_code}) for "
                    f"{self.instance_id}: stderr={result.stderr[:500]}"
                )
        except Exception:
            await self.cleanup()
            raise

        # Install apply_patch helper script. GPT-OSS models are trained to
        # use `apply_patch` for file editing — without it, 58% of edit attempts
        # silently fail because the command doesn't exist in bash.
        await self.sandbox.write_file("/usr/local/bin/apply_patch", _APPLY_PATCH_SCRIPT)
        await self.sandbox.run_command("chmod +x /usr/local/bin/apply_patch")

        # NOTE: Test patch is NOT applied here. The official eval script
        # applies it at grading time, after the model is done editing.
        # This prevents the model from accidentally modifying test files.

        # Create tool and reward
        bash_tool = _SWEBashTool(self.sandbox)
        reward_fn = _SWEBenchReward(
            self.sandbox, eval_script or "", self.raw_instance or {}, self.instance_id
        )

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
# Sandbox factory — miniconda image for faithful SWE-bench setup
# ---------------------------------------------------------------------------


def _get_swebench_sandbox_factory(config: BenchmarkConfig) -> object:
    """Create a sandbox factory with miniconda pre-installed.

    The official SWE-bench harness relies on conda for per-instance Python
    version management and dependency isolation.  We bake miniconda into the
    Modal image so the TestSpec commands run verbatim.
    """
    if config.sandbox_factory is not None:
        return config.sandbox_factory

    try:
        import modal

        from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox

        image = (
            modal.Image.debian_slim()
            .apt_install("git", "wget", "build-essential", "libffi-dev", "libssl-dev")
            .run_commands(
                # Install miniconda (matches official SWE-bench base image)
                "wget -q https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-x86_64.sh"
                " -O /root/miniconda.sh",
                "bash /root/miniconda.sh -b -p /opt/miniconda3",
                "rm /root/miniconda.sh",
                "/opt/miniconda3/bin/conda init bash",
                # Make conda available in non-interactive shells
                'echo ". /opt/miniconda3/etc/profile.d/conda.sh" >> /root/.bashrc',
            )
        )

        async def _modal_factory():
            return await ModalSandbox.create(image=image, timeout=1800)

        return _modal_factory
    except ImportError:

        async def _missing_factory():
            raise RuntimeError(
                "SWE-bench requires Modal for sandbox execution.\n"
                "Install with: uv pip install 'tinker-cookbook[eval-swe-bench]'"
            )

        return _missing_factory


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
    experimental = True
    requires_sandbox = True
    multi_turn = True
    recommended_timeout = 1800

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = cast(Dataset, load_benchmark_dataset("princeton-nlp/SWE-bench_Verified"))
        ds = limit_dataset(ds, config.max_examples)

        sandbox_factory = _get_swebench_sandbox_factory(config)

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
                    raw_instance=row,
                    system_prompt=config.system_prompt,
                    max_trajectory_tokens=config.max_trajectory_tokens,
                    max_generation_tokens=config.max_generation_tokens,
                )
            )

        return envs

    def aggregate(self, rewards: list[float], metrics_list: list[Metrics]) -> BenchmarkResult:
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
