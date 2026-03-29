"""
SWE-RL environment for Nemotron-Cascade-2 replication.

Data sources (in priority order):
  1. nvidia/Nemotron-Cascade-RL-SWE (~110K instances) — DEFAULT
     Rich prompts with issue + codebase context + golden patches.
     This is the exact dataset from the Nemotron Cascade 2 paper.
  2. R2E-Gym/R2E-Gym-Subset (4,578 instances) — fallback with Docker execution
  3. nvidia/Nemotron-Cascade-2-RL-data SWE-RL split — legacy

The model generates a code patch, which is scored by either:
  - An LLM judge (Qwen3.5-397B) — the default, matching the paper's use of GPT-OSS-120B.
    When golden_patch is available (Cascade SWE data), the judge compares proposed vs golden.
  - Execution-based testing in a Modal sandbox (R2E-Gym / legacy mode)

Paper hyperparameters (Agentless SWE RL):
  - Batch 128×16=2048, max seq 98K, LR 3e-6
  - GPT-OSS-120B as reward model, ~40-50 steps
  - Mask loss where no rollout in group gets reward > 0.5
"""

import json
import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import chz
import tinker
from datasets import Dataset

from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition, TinkerMessageCompleter
from tinker_cookbook.renderers import Message
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

logger = logging.getLogger(__name__)

# Threshold (bytes) above which we warn about test file sizes before sandbox upload
_LARGE_TEST_FILE_THRESHOLD = 50_000


def _summarize_lengths(ds: Dataset, field: str) -> str:
    """Return a short summary of string lengths for a dataset field."""
    lengths = [len(row.get(field, "") or "") for row in ds.select(range(min(500, len(ds))))]
    if not lengths:
        return "no data"
    lengths.sort()
    return (
        f"min={lengths[0]:,}, median={lengths[len(lengths)//2]:,}, "
        f"max={lengths[-1]:,}, n_sampled={len(lengths)}"
    )


def extract_patch(response: str) -> str | None:
    """Extract a unified diff patch from the model's response."""
    # Try to find diff block in code fence
    match = re.search(r'```(?:diff|patch)?\s*\n(.*?)\n```', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find unified diff directly
    lines = response.split('\n')
    diff_lines = []
    in_diff = False
    for line in lines:
        if line.startswith('---') or line.startswith('+++') or line.startswith('diff --git'):
            in_diff = True
        if in_diff:
            diff_lines.append(line)
            if line.strip() == '' and diff_lines and not any(
                diff_lines[-1].startswith(p) for p in ['+', '-', ' ', '@', 'diff', '---', '+++']
            ):
                break

    if diff_lines:
        return '\n'.join(diff_lines).strip()
    return None


# ---------------------------------------------------------------------------
# LLM Judge for patch quality (matches paper's GPT-OSS-120B reward model)
# ---------------------------------------------------------------------------

SWE_JUDGE_SYSTEM_PROMPT = """\
You are an expert software engineer reviewing a proposed code patch for a GitHub issue.

Rate the quality of the patch on a scale of 0 to 10:
- 0: No valid patch, empty, or completely irrelevant
- 1-2: Has diff syntax but targets wrong files or is clearly incorrect
- 3-4: Addresses the right area but the fix is incomplete or wrong
- 5-6: Reasonable attempt that partially addresses the issue
- 7-8: Good patch that likely fixes the issue with minor concerns
- 9-10: Excellent patch that correctly and completely fixes the issue

Consider:
1. Does it address the described issue?
2. Is the diff syntactically correct and applicable?
3. Are the right files and locations targeted?
4. Would the change likely fix the problem without introducing regressions?

Output ONLY the integer score, nothing else."""

SWE_JUDGE_USER_TEMPLATE = """\
## GitHub Issue
{problem_statement}

## Repository
{repo}

## Proposed Patch
```diff
{patch}
```

Score (0-10):"""

SWE_JUDGE_WITH_GOLDEN_SYSTEM_PROMPT = """\
You are an expert software engineer comparing a proposed code patch against a known correct patch for a GitHub issue.

Rate the quality of the proposed patch on a scale of 0 to 10:
- 0: No valid patch, empty, or completely irrelevant
- 1-2: Has diff syntax but targets entirely wrong files/locations
- 3-4: Targets some right areas but the approach is fundamentally different and wrong
- 5-6: Partially overlaps with the correct fix but misses important parts
- 7-8: Captures the core fix with minor differences (e.g., style, extra changes)
- 9-10: Functionally equivalent to the correct patch (may differ in style/whitespace)

Consider:
1. Does the proposed patch modify the same files and locations as the golden patch?
2. Are the logical changes equivalent or similar?
3. Would the proposed patch achieve the same fix as the golden patch?
4. Does it introduce any regressions not present in the golden patch?

Output ONLY the integer score, nothing else."""

SWE_JUDGE_WITH_GOLDEN_USER_TEMPLATE = """\
## Issue Description
{problem_statement}

## Golden (Correct) Patch
```diff
{golden_patch}
```

## Proposed Patch
```diff
{patch}
```

Score (0-10):"""

# Maximum characters of problem statement to send to the judge
_MAX_JUDGE_PROBLEM_CHARS = 8_000
# Maximum characters for golden patch in judge context
_MAX_JUDGE_GOLDEN_PATCH_CHARS = 8_000


def _parse_swe_judge_score(response_text: str) -> float:
    """Extract a 0-10 integer score from the judge response and normalise to [0, 1]."""
    # Find the LAST integer in the response (thinking models may emit reasoning first)
    matches = re.findall(r'\b(\d{1,2})\b', response_text)
    if matches:
        score = int(matches[-1])
        return min(max(score, 0), 10) / 10.0
    logger.warning(f"Could not parse SWE judge score from: {response_text!r}")
    return 0.0


async def get_swe_llm_judge_reward(
    problem_statement: str,
    repo: str,
    patch: str,
    judge_completer: TinkerMessageCompleter,
    golden_patch: str | None = None,
) -> tuple[float, str]:
    """Query the LLM judge for patch quality. Returns (normalised_reward, raw_response).

    When golden_patch is provided, uses a comparison-based prompt that asks the judge
    to evaluate the proposed patch against the known correct fix. This produces more
    calibrated scores since the judge has a concrete reference.
    """
    truncated_problem = problem_statement[:_MAX_JUDGE_PROBLEM_CHARS]
    if len(problem_statement) > _MAX_JUDGE_PROBLEM_CHARS:
        truncated_problem += "\n[...truncated...]"

    if golden_patch:
        # Golden-patch comparison mode (used with nvidia/Nemotron-Cascade-RL-SWE)
        truncated_golden = golden_patch[:_MAX_JUDGE_GOLDEN_PATCH_CHARS]
        if len(golden_patch) > _MAX_JUDGE_GOLDEN_PATCH_CHARS:
            truncated_golden += "\n[...truncated...]"
        messages: list[Message] = [
            {"role": "system", "content": SWE_JUDGE_WITH_GOLDEN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SWE_JUDGE_WITH_GOLDEN_USER_TEMPLATE.format(
                    problem_statement=truncated_problem,
                    golden_patch=truncated_golden,
                    patch=patch[:8_000],
                ),
            },
        ]
    else:
        # No golden patch — original issue-only mode
        messages = [
            {"role": "system", "content": SWE_JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SWE_JUDGE_USER_TEMPLATE.format(
                    problem_statement=truncated_problem,
                    repo=repo,
                    patch=patch[:8_000],
                ),
            },
        ]

    try:
        judge_response = await judge_completer(messages)
        response_text = judge_response.get("content", "")
        if not isinstance(response_text, str):
            response_text = str(response_text)
        reward = _parse_swe_judge_score(response_text)
        return reward, response_text
    except Exception as e:
        logger.warning(f"SWE LLM judge call failed: {e}")
        return 0.0, f"ERROR: {e}"


async def _write_file_to_sandbox(
    sandbox: "modal.Sandbox",
    path: str,
    content: str,
) -> None:
    """Write a file to a Modal sandbox via stdin, avoiding ARG_MAX limits.

    Instead of embedding file contents in a bash command argument (which hits
    Modal's 65,536-byte CMD limit for large files), we pipe the content through
    stdin using ``cat > path``.
    """
    proc = await sandbox.exec.aio("bash", "-c", f"cat > {path}")
    proc.stdin.write(content.encode())
    proc.stdin.write_eof()
    await proc.wait.aio()


async def run_swe_test_in_modal(
    instance_id: str,
    repo: str,
    base_commit: str,
    patch: str,
    fail_to_pass: list[str],
    timeout: int = 300,
    docker_image: str | None = None,
    r2e_test_files: list[tuple[str, str]] | None = None,
) -> tuple[bool, str]:
    """Run SWE-bench test in Modal sandbox.

    When docker_image is provided (R2E-Gym mode), uses the pre-built Docker image
    which has the repo at /testbed with all dependencies installed. Test files are
    written into the sandbox via stdin (not embedded in CMD) to avoid ARG_MAX limits.

    Returns (passed, details).
    """
    try:
        import modal
    except ImportError:
        return False, "Modal not installed"

    # Track sandbox metrics
    sandbox_metrics: dict[str, float] = {}

    if docker_image:
        # R2E-Gym mode: repo is pre-installed at /testbed with all deps
        test_file_names: list[str] = []
        for fname, code in (r2e_test_files or []):
            file_size = len(code.encode())
            if file_size > _LARGE_TEST_FILE_THRESHOLD:
                logger.warning(
                    "Large test file for %s: %s is %d bytes (threshold: %d). "
                    "Will be written via stdin to avoid ARG_MAX limits.",
                    instance_id, fname, file_size, _LARGE_TEST_FILE_THRESHOLD,
                )
            test_file_names.append(f"r2e_tests/{fname}")

        if not test_file_names:
            logger.warning("No R2E-Gym test files for instance %s", instance_id)
            return False, "No R2E-Gym test files"

        try:
            app = await modal.App.lookup.aio("nemotron-cascade-swe-rl", create_if_missing=True)
            image = modal.Image.from_registry(docker_image)
            sb = await modal.Sandbox.create.aio(
                app=app,
                image=image,
                timeout=timeout,
            )
        except Exception as e:
            logger.error(
                "Sandbox creation failed for instance %s (docker_image=%s): %s",
                instance_id, docker_image, e,
            )
            sandbox_metrics["env/all/sandbox_errors"] = 1.0
            return False, f"Sandbox creation error for {instance_id}: {str(e)[:200]}"

        try:
            # Set up test directory
            setup_proc = await sb.exec.aio("bash", "-c", "mkdir -p /testbed/r2e_tests")
            await setup_proc.wait.aio()

            # Write test files via stdin (avoids ARG_MAX limit)
            for fname, code in (r2e_test_files or []):
                await _write_file_to_sandbox(sb, f"/testbed/r2e_tests/{fname}", code)

            # Write patch via stdin and apply it
            await _write_file_to_sandbox(sb, "/testbed/_proposed.patch", patch)

            test_cmds = " && ".join(
                f'python -m pytest "{t}" -x 2>&1 && PASS=$((PASS + 1))'
                for t in test_file_names
            )

            test_script = f"""\
set -x
cd /testbed
git apply _proposed.patch || {{ echo "PATCH_APPLY_FAILED"; exit 1; }}
PASS=0
TOTAL={len(test_file_names)}
{test_cmds}
echo "RESULT: $PASS/$TOTAL"
test "$PASS" -eq "$TOTAL"
"""
            test_proc = await sb.exec.aio("bash", "-c", test_script)
            stdout = await test_proc.stdout.read.aio()
            stderr = await test_proc.stderr.read.aio()
            exit_code = await test_proc.wait.aio()

            passed = exit_code == 0
            output = (stdout + "\n" + stderr)[-2000:]

            if not passed:
                logger.info(
                    "Sandbox test failed for %s (exit_code=%d):\nstdout: %s\nstderr: %s",
                    instance_id, exit_code, stdout[-500:], stderr[-500:],
                )

            return passed, output
        except Exception as e:
            logger.error("Sandbox execution error for instance %s: %s", instance_id, e)
            return False, f"Sandbox exec error for {instance_id}: {str(e)[:200]}"
        finally:
            try:
                await sb.terminate.aio()
            except Exception:
                pass
    else:
        # Legacy mode: clone from GitHub
        if not repo or not base_commit:
            return False, "Missing repo/commit"

        try:
            app = await modal.App.lookup.aio("nemotron-cascade-swe-rl", create_if_missing=True)
            sb = await modal.Sandbox.create.aio(
                image=modal.Image.debian_slim().pip_install("pytest", "pytest-timeout", "setuptools"),
                app=app,
                timeout=timeout,
            )
        except Exception as e:
            logger.error(
                "Sandbox creation failed for instance %s (legacy mode, repo=%s): %s",
                instance_id, repo, e,
            )
            return False, f"Sandbox creation error for {instance_id}: {str(e)[:200]}"

        try:
            # Write patch via stdin
            await _write_file_to_sandbox(sb, "/tmp/_proposed.patch", patch)

            test_cmds = " && ".join(
                f'python -m pytest "{t}" -x 2>&1 && PASS=$((PASS + 1))'
                for t in fail_to_pass
            )

            test_script = f"""\
set -x
git clone --depth=1 https://github.com/{repo}.git /workspace/repo || exit 1
cd /workspace/repo
git fetch --depth=1 origin {base_commit} && git checkout {base_commit} || true
git apply /tmp/_proposed.patch || {{ echo "PATCH_APPLY_FAILED"; exit 1; }}
pip install -e . 2>/dev/null || true
PASS=0
TOTAL={len(fail_to_pass)}
{test_cmds}
echo "RESULT: $PASS/$TOTAL"
test "$PASS" -eq "$TOTAL"
"""
            test_proc = await sb.exec.aio("bash", "-c", test_script)
            stdout = await test_proc.stdout.read.aio()
            stderr = await test_proc.stderr.read.aio()
            exit_code = await test_proc.wait.aio()

            passed = exit_code == 0
            output = (stdout + "\n" + stderr)[-2000:]

            if not passed:
                logger.info(
                    "Sandbox test failed for %s (exit_code=%d):\nstdout: %s\nstderr: %s",
                    instance_id, exit_code, stdout[-500:], stderr[-500:],
                )

            return passed, output
        except Exception as e:
            logger.error("Sandbox execution error for instance %s: %s", instance_id, e)
            return False, f"Sandbox exec error for {instance_id}: {str(e)[:200]}"
        finally:
            try:
                await sb.terminate.aio()
            except Exception:
                pass


class SWERLEnv(Env):
    """Single-turn SWE-bench environment (agentless: generate patch directly).

    reward_mode controls how patches are scored:
      - "llm_judge": Use an LLM (Qwen3.5-397B) to rate patch quality (0-10 -> [0,1]).
        This matches the paper's approach of using GPT-OSS-120B as a reward model.
        The actual LLM call happens in SWEGroupBuilder.compute_group_rewards().
      - "execution": Run FAIL_TO_PASS tests in a Modal sandbox (binary 0/1).
    """

    def __init__(
        self,
        instance_id: str,
        problem_statement: str,
        repo: str,
        base_commit: str,
        fail_to_pass: list[str],
        renderer: renderers.Renderer,
        reward_mode: Literal["execution", "llm_judge"] = "llm_judge",
        use_modal: bool = True,
        docker_image: str | None = None,
        r2e_test_files: list[tuple[str, str]] | None = None,
        # Cascade SWE data fields (nvidia/Nemotron-Cascade-RL-SWE)
        prebuilt_prompt: str | None = None,
        golden_patch: str | None = None,
    ):
        self.instance_id = instance_id
        self.problem_statement = problem_statement
        self.repo = repo
        self.base_commit = base_commit
        self.fail_to_pass = fail_to_pass
        self.renderer = renderer
        self.reward_mode = reward_mode
        self.use_modal = use_modal
        self.docker_image = docker_image
        # R2E-Gym: list of (filename, code) for test files to write into sandbox
        self.r2e_test_files = r2e_test_files or []
        # Cascade SWE data: pre-constructed prompt with codebase context
        self.prebuilt_prompt = prebuilt_prompt
        # Cascade SWE data: golden patch for comparison-based LLM judge
        self.golden_patch = golden_patch
        # Populated during step() for use by compute_group_rewards()
        self._extracted_patch: str | None = None
        self._response_content: str = ""
        self._stop_reason: str | None = None

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        if self.prebuilt_prompt:
            # Cascade SWE data: prompt already includes issue + codebase context + instructions
            prompt = self.prebuilt_prompt
        else:
            prompt = (
                f"You are a software engineer. Fix the following issue in the repository {self.repo}.\n\n"
                f"## Issue\n{self.problem_statement}\n\n"
                f"## Instructions\n"
                f"Generate a unified diff patch that fixes this issue. "
                f"Output the patch in a ```diff code block."
            )
        messages: list[Message] = [{"role": "user", "content": prompt}]
        return self.renderer.build_generation_prompt(messages), self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)
        self._response_content = content

        stop_reason = (extra or {}).get("stop_reason")
        self._stop_reason = stop_reason

        patch = extract_patch(content)
        self._extracted_patch = patch

        sandbox_error = False
        sandbox_ran = False

        if self.reward_mode == "llm_judge":
            # Reward will be computed in SWEGroupBuilder.compute_group_rewards()
            # Return a placeholder reward of 0; it will be overwritten.
            reward = 0.0
            details = "pending llm_judge"
        elif stop_reason == "length":
            reward = 0.0
            details = "overlong"
        else:
            if not patch:
                reward = 0.0
                details = "no patch found"
            elif self.use_modal and (self.docker_image or (self.repo and self.base_commit)):
                sandbox_ran = True
                passed, details = await run_swe_test_in_modal(
                    self.instance_id, self.repo, self.base_commit,
                    patch, self.fail_to_pass,
                    docker_image=self.docker_image,
                    r2e_test_files=self.r2e_test_files,
                )
                reward = 1.0 if passed else 0.0
                # Detect sandbox-level errors (not just test failures)
                if not passed and ("Sandbox" in details or "Modal error" in details):
                    sandbox_error = True
            else:
                # Fallback: reward for producing a valid-looking patch
                reward = 0.5 if patch else 0.0
                details = "patch produced (no execution)"

        with logtree.scope_header("Problem"):
            logtree.log_text(f"Instance: {self.instance_id}")
            logtree.log_text(f"Repo: {self.repo}")
            logtree.log_text(f"Problem: {self.problem_statement[:500]}...")
        with logtree.scope_header("Response"):
            logtree.log_text(content[:1000])
        with logtree.scope_header("Reward"):
            logtree.table_from_dict(
                {"reward": f"{reward:.1f}", "details": str(details)[:200]},
                caption="SWE-RL reward",
            )

        metrics: dict[str, float] = {
            "correct": reward,
            "has_patch": float(bool(patch)),
        }
        if sandbox_ran:
            metrics["env/all/sandbox_success_rate"] = 0.0 if sandbox_error else 1.0
            metrics["env/all/sandbox_errors"] = 1.0 if sandbox_error else 0.0

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )


@dataclass(frozen=True)
class SWEGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], SWERLEnv]
    num_envs: int
    reward_mode: Literal["execution", "llm_judge"] = "llm_judge"
    # LLM judge config (used only when reward_mode="llm_judge")
    judge_model_name: str = "Qwen/Qwen3.5-397B-A17B"
    judge_renderer_name: str = "qwen3_5"
    judge_max_tokens: int = 512
    judge_temperature: float = 0.0
    judge_base_url: str | None = None

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """Score trajectories. For llm_judge mode, queries the judge model."""
        if self.reward_mode != "llm_judge":
            # Execution mode: rewards already computed in env.step(), return 0 group bonus
            return [(0.0, {}) for _ in trajectory_group]

        # LLM judge mode: create judge completer and score each patch
        judge_tokenizer = get_tokenizer(self.judge_model_name)
        judge_renderer = renderers.get_renderer(self.judge_renderer_name, tokenizer=judge_tokenizer)
        service_client = tinker.ServiceClient(base_url=self.judge_base_url)
        judge_sampling_client = await service_client.create_sampling_client_async(
            base_model=self.judge_model_name,
        )
        judge_completer = TinkerMessageCompleter(
            sampling_client=judge_sampling_client,
            renderer=judge_renderer,
            max_tokens=self.judge_max_tokens,
            temperature=self.judge_temperature,
        )

        results: list[tuple[float, Metrics]] = []
        for traj, env in zip(trajectory_group, env_group):
            assert isinstance(env, SWERLEnv)

            if env._stop_reason == "length":
                results.append((0.0, {"judge_reward": 0.0, "overlong": 1.0, "has_patch": 0.0}))
                continue

            patch = env._extracted_patch
            if not patch:
                results.append((0.0, {"judge_reward": 0.0, "has_patch": 0.0}))
                continue

            reward, judge_text = await get_swe_llm_judge_reward(
                problem_statement=env.problem_statement,
                repo=env.repo,
                patch=patch,
                judge_completer=judge_completer,
                golden_patch=env.golden_patch,
            )

            with logtree.scope_header("LLM Judge"):
                logtree.table_from_dict(
                    {
                        "instance_id": env.instance_id,
                        "judge_raw": judge_text[:200],
                        "judge_reward": f"{reward:.2f}",
                        "patch_len": str(len(patch)),
                    },
                    caption="SWE-RL LLM judge reward",
                )

            results.append((reward, {"judge_reward": reward, "has_patch": 1.0}))

        return results

    def logging_tags(self) -> list[str]:
        return ["swe_rl"]


class SWERLDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        reward_mode: Literal["execution", "llm_judge"] = "llm_judge",
        use_modal: bool = True,
        use_r2e_gym: bool = True,
        use_cascade_swe_data: bool = True,
        seed: int = 0,
    ):
        from datasets import load_dataset

        self.reward_mode = reward_mode
        self.use_r2e_gym = use_r2e_gym
        self.use_cascade_swe_data = use_cascade_swe_data

        if use_cascade_swe_data:
            # nvidia/Nemotron-Cascade-RL-SWE: ~110K instances with rich prompts
            # containing issue + codebase context + golden patches.
            # This is the exact dataset used in the Nemotron Cascade 2 paper.
            logger.info("Loading nvidia/Nemotron-Cascade-RL-SWE from HuggingFace...")
            ds = load_dataset("nvidia/Nemotron-Cascade-RL-SWE", split="train")
            ds = cast(Dataset, ds)
            ds = ds.filter(
                lambda x: (
                    x.get("prompt") is not None
                    and len(x.get("prompt", "")) > 100
                    and x.get("golden_patch") is not None
                )
            )
            logger.info(
                f"Cascade SWE dataset: {len(ds)} instances after filtering "
                f"(prompt lengths: {_summarize_lengths(ds, 'prompt')})"
            )
        elif use_r2e_gym:
            logger.info("Loading R2E-Gym data from HuggingFace...")
            ds = load_dataset("R2E-Gym/R2E-Gym-Subset", split="train")
            ds = cast(Dataset, ds)
            ds = ds.filter(
                lambda x: x.get("docker_image") is not None and x.get("problem_statement") is not None
            )
        else:
            logger.info("Loading SWE-RL data from HuggingFace...")
            ds = load_dataset("nvidia/Nemotron-Cascade-2-RL-data", name="SWE-RL", split="train")
            ds = cast(Dataset, ds)
            ds = ds.filter(
                lambda x: x.get("instance_id") is not None and x.get("problem_statement") is not None
            )

        self.ds = ds.shuffle(seed=seed)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.use_modal = use_modal
        data_source = "cascade_swe" if use_cascade_swe_data else ("r2e_gym" if use_r2e_gym else "legacy")
        logger.info(f"SWE-RL dataset: {len(self.ds)} valid instances (source={data_source})")

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end
        return [
            b for row in self.ds.select(range(batch_start, batch_end))
            if (b := self._make_builder(row)) is not None
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_builder(self, row: dict) -> SWEGroupBuilder | None:
        try:
            if self.use_cascade_swe_data:
                # nvidia/Nemotron-Cascade-RL-SWE schema:
                #   source, instance_id, prompt, golden_patch,
                #   relevant_file_contents, original_prompt
                instance_id = row.get("instance_id", "unknown")
                prebuilt_prompt = row.get("prompt", "")
                golden_patch = row.get("golden_patch", "")
                original_prompt = row.get("original_prompt", "")
                source = row.get("source", "")

                if not prebuilt_prompt or not golden_patch:
                    return None

                # Use original_prompt as problem_statement for judge context
                # (it's shorter and contains the core issue description)
                problem_statement = original_prompt or prebuilt_prompt[:_MAX_JUDGE_PROBLEM_CHARS]

                rm = self.reward_mode
                return SWEGroupBuilder(
                    env_thunk=lambda iid=instance_id, ps=problem_statement, pp=prebuilt_prompt, gp=golden_patch, _rm=rm: SWERLEnv(
                        instance_id=iid,
                        problem_statement=ps,
                        repo="",  # Not needed — prompt has full context
                        base_commit="",
                        fail_to_pass=[],
                        renderer=self.renderer,
                        reward_mode=_rm,
                        use_modal=False,  # No execution for cascade data (LLM judge)
                        prebuilt_prompt=pp,
                        golden_patch=gp,
                    ),
                    num_envs=self.group_size,
                    reward_mode=rm,
                )
            elif self.use_r2e_gym:
                # R2E-Gym schema: docker_image, problem_statement, repo_name, commit_hash, etc.
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

                if not docker_image or not r2e_test_files:
                    return None

                # fail_to_pass is not used in R2E-Gym mode (tests are in r2e_test_files)
                rm = self.reward_mode
                return SWEGroupBuilder(
                    env_thunk=lambda iid=instance_id, p=problem, r=repo, bc=base_commit, di=docker_image, rtf=r2e_test_files, _rm=rm: SWERLEnv(
                        instance_id=iid, problem_statement=p, repo=r,
                        base_commit=bc, fail_to_pass=[],
                        renderer=self.renderer, reward_mode=_rm,
                        use_modal=self.use_modal,
                        docker_image=di, r2e_test_files=rtf,
                    ),
                    num_envs=self.group_size,
                    reward_mode=rm,
                )
            else:
                # Legacy nvidia SWE-RL schema
                instance_id = row["instance_id"]
                problem = row.get("problem_statement", "")
                repo = row.get("repo", "")
                base_commit = row.get("base_commit", "")
                fail_to_pass = row.get("FAIL_TO_PASS", [])
                if isinstance(fail_to_pass, str):
                    try:
                        fail_to_pass = json.loads(fail_to_pass)
                    except json.JSONDecodeError:
                        fail_to_pass = [fail_to_pass]

                rm = self.reward_mode
                return SWEGroupBuilder(
                    env_thunk=lambda iid=instance_id, p=problem, r=repo, bc=base_commit, ftp=fail_to_pass, _rm=rm: SWERLEnv(
                        instance_id=iid, problem_statement=p, repo=r,
                        base_commit=bc, fail_to_pass=ftp,
                        renderer=self.renderer, reward_mode=_rm,
                        use_modal=self.use_modal,
                    ),
                    num_envs=self.group_size,
                    reward_mode=rm,
                )
        except Exception as e:
            logger.warning(f"Failed to parse SWE-RL row: {e}")
            return None


@chz.chz
class SWERLDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int = 4  # Smaller default since each test is expensive
    reward_mode: Literal["execution", "llm_judge"] = "llm_judge"
    use_modal: bool = True
    use_r2e_gym: bool = True  # Use R2E-Gym Docker images (pre-built envs with deps)
    use_cascade_swe_data: bool = True  # Use nvidia/Nemotron-Cascade-RL-SWE (~110K instances with codebase context)
    seed: int = 0

    async def __call__(self) -> tuple[SWERLDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return SWERLDataset(
            batch_size=self.batch_size, group_size=self.group_size,
            renderer=renderer, reward_mode=self.reward_mode,
            use_modal=self.use_modal,
            use_r2e_gym=self.use_r2e_gym,
            use_cascade_swe_data=self.use_cascade_swe_data,
            seed=self.seed,
        ), None
