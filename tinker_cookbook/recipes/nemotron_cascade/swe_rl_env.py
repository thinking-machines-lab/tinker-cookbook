"""
SWE-RL environment for Nemotron-Cascade-2 replication.

Uses the SWE-RL subset (622 valid SWE-bench instances).
The model generates a code patch, which is scored by either:
  - An LLM judge (Qwen3.5-397B) — the default, matching the paper's use of GPT-OSS-120B
  - Execution-based testing in a Modal sandbox (legacy mode)

Paper hyperparameters (Agentless SWE RL):
  - Batch 128×16=2048, max seq 98K, LR 3e-6
  - GPT-OSS-120B as reward model, ~40-50 steps
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

# Maximum characters of problem statement to send to the judge
_MAX_JUDGE_PROBLEM_CHARS = 8_000


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
) -> tuple[float, str]:
    """Query the LLM judge for patch quality. Returns (normalised_reward, raw_response)."""
    truncated_problem = problem_statement[:_MAX_JUDGE_PROBLEM_CHARS]
    if len(problem_statement) > _MAX_JUDGE_PROBLEM_CHARS:
        truncated_problem += "\n[...truncated...]"

    messages: list[Message] = [
        {"role": "system", "content": SWE_JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": SWE_JUDGE_USER_TEMPLATE.format(
                problem_statement=truncated_problem,
                repo=repo,
                patch=patch[:8_000],  # Cap patch length for judge context
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
    which has the repo at /testbed with all dependencies installed. Test files from
    r2e_test_files are written into the sandbox before running.

    Returns (passed, details).
    """
    try:
        import modal
    except ImportError:
        return False, "Modal not installed"

    if docker_image:
        # R2E-Gym mode: repo is pre-installed at /testbed with all deps
        import base64
        patch_b64 = base64.b64encode(patch.encode()).decode()

        # Encode test files as base64 and write them into r2e_tests/
        test_file_setup = "mkdir -p /testbed/r2e_tests\n"
        test_file_names = []
        for fname, code in (r2e_test_files or []):
            code_b64 = base64.b64encode(code.encode()).decode()
            test_file_setup += f"echo '{code_b64}' | base64 -d > /testbed/r2e_tests/{fname}\n"
            test_file_names.append(f"r2e_tests/{fname}")

        if not test_file_names:
            return False, "No R2E-Gym test files"

        test_cmds = " && ".join(
            f'python -m pytest "{t}" -x --timeout=60 2>/dev/null && PASS=$((PASS + 1))'
            for t in test_file_names
        )

        test_script = f"""
set -x
cd /testbed
{test_file_setup}
echo '{patch_b64}' | base64 -d | git apply - 2>/dev/null || echo "PATCH_APPLY_FAILED"
PASS=0
TOTAL={len(test_file_names)}
{test_cmds}
echo "RESULT: $PASS/$TOTAL"
test "$PASS" -eq "$TOTAL"
"""

        try:
            app = await modal.App.lookup.aio("nemotron-cascade-swe-rl", create_if_missing=True)
            image = modal.Image.from_registry(docker_image)
            sb = await modal.Sandbox.create.aio(
                "bash", "-c", test_script,
                image=image,
                app=app,
                timeout=timeout,
            )
            await sb.wait.aio()
            stdout = await sb.stdout.read.aio()
            stderr = await sb.stderr.read.aio()
            passed = sb.returncode == 0
            output = (stdout + "\n" + stderr)[-500:]
            return passed, output
        except Exception as e:
            return False, f"Modal error: {str(e)[:200]}"
    else:
        # Legacy mode: clone from GitHub
        if not repo or not base_commit:
            return False, "Missing repo/commit"

        import base64
        patch_b64 = base64.b64encode(patch.encode()).decode()

        test_cmds = " && ".join(
            f'python -m pytest "{t}" -x --timeout=60 2>/dev/null && PASS=$((PASS + 1))'
            for t in fail_to_pass
        )

        test_script = f"""
set -x
git clone --depth=1 https://github.com/{repo}.git /workspace/repo 2>/dev/null || exit 1
cd /workspace/repo
git fetch --depth=1 origin {base_commit} 2>/dev/null && git checkout {base_commit} 2>/dev/null || true
echo '{patch_b64}' | base64 -d | git apply - 2>/dev/null || echo "PATCH_APPLY_FAILED"
pip install -e . 2>/dev/null || true
PASS=0
TOTAL={len(fail_to_pass)}
{test_cmds}
echo "RESULT: $PASS/$TOTAL"
test "$PASS" -eq "$TOTAL"
"""

        try:
            app = await modal.App.lookup.aio("nemotron-cascade-swe-rl", create_if_missing=True)
            sb = await modal.Sandbox.create.aio(
                "bash", "-c", test_script,
                image=modal.Image.debian_slim().pip_install("pytest", "pytest-timeout", "setuptools"),
                app=app,
                timeout=timeout,
            )
            await sb.wait.aio()
            stdout = await sb.stdout.read.aio()
            stderr = await sb.stderr.read.aio()
            passed = sb.returncode == 0
            output = (stdout + "\n" + stderr)[-500:]
            return passed, output
        except Exception as e:
            return False, f"Modal error: {str(e)[:200]}"


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
        # Populated during step() for use by compute_group_rewards()
        self._extracted_patch: str | None = None
        self._response_content: str = ""
        self._stop_reason: str | None = None

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
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
                passed, details = await run_swe_test_in_modal(
                    self.instance_id, self.repo, self.base_commit,
                    patch, self.fail_to_pass,
                    docker_image=self.docker_image,
                    r2e_test_files=self.r2e_test_files,
                )
                reward = 1.0 if passed else 0.0
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

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={"correct": reward, "has_patch": float(bool(patch))},
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
        seed: int = 0,
    ):
        from datasets import load_dataset

        self.reward_mode = reward_mode
        self.use_r2e_gym = use_r2e_gym

        if use_r2e_gym:
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
        logger.info(f"SWE-RL dataset: {len(self.ds)} valid instances (r2e_gym={use_r2e_gym})")

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
            if self.use_r2e_gym:
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
    seed: int = 0

    async def __call__(self) -> tuple[SWERLDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return SWERLDataset(
            batch_size=self.batch_size, group_size=self.group_size,
            renderer=renderer, reward_mode=self.reward_mode,
            use_modal=self.use_modal,
            use_r2e_gym=self.use_r2e_gym, seed=self.seed,
        ), None
