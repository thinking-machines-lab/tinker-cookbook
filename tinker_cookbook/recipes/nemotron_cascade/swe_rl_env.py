"""
SWE-RL environment for Nemotron-Cascade-2 replication.

Uses the SWE-RL subset (622 valid SWE-bench instances).
The model generates a code patch, which is applied to the repo and
validated by running the FAIL_TO_PASS tests in a Modal sandbox.

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
from typing import cast

import chz
import tinker
from datasets import Dataset

from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
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


async def run_swe_test_in_modal(
    instance_id: str,
    repo: str,
    base_commit: str,
    patch: str,
    fail_to_pass: list[str],
    timeout: int = 300,
) -> tuple[bool, str]:
    """Run SWE-bench test in Modal sandbox.

    Returns (passed, details).
    """
    try:
        import modal
    except ImportError:
        return False, "Modal not installed"

    # Create a sandbox that clones the repo and runs tests
    test_script = f"""
#!/bin/bash
set -e

# Clone repo
git clone https://github.com/{repo}.git /workspace/repo 2>/dev/null
cd /workspace/repo
git checkout {base_commit} 2>/dev/null

# Apply patch
echo '{patch.replace("'", "'\"'\"'")}' | git apply - 2>/dev/null || echo "PATCH_FAILED"

# Install deps (best effort)
pip install -e . 2>/dev/null || true

# Run tests
PASS=0
TOTAL={len(fail_to_pass)}
"""
    for test in fail_to_pass:
        test_script += f"""
if python -m pytest {test} -x --timeout=60 2>/dev/null; then
    PASS=$((PASS + 1))
fi
"""
    test_script += """
echo "RESULT: $PASS/$TOTAL passed"
if [ "$PASS" -eq "$TOTAL" ]; then
    exit 0
else
    exit 1
fi
"""

    try:
        sandbox = await modal.Sandbox.create_async(
            "bash", "-c", test_script,
            image=modal.Image.debian_slim().pip_install("pytest", "pytest-timeout"),
            timeout=timeout,
        )
        result = await sandbox.wait_async()
        stdout = (await sandbox.stdout.read_async()).decode() if hasattr(sandbox.stdout, 'read_async') else ""
        passed = result.returncode == 0
        return passed, stdout[-500:] if stdout else f"exit_code={result.returncode}"
    except Exception as e:
        return False, f"Modal error: {str(e)[:200]}"


class SWERLEnv(Env):
    """Single-turn SWE-bench environment (agentless: generate patch directly)."""

    def __init__(
        self,
        instance_id: str,
        problem_statement: str,
        repo: str,
        base_commit: str,
        fail_to_pass: list[str],
        renderer: renderers.Renderer,
        use_modal: bool = True,
    ):
        self.instance_id = instance_id
        self.problem_statement = problem_statement
        self.repo = repo
        self.base_commit = base_commit
        self.fail_to_pass = fail_to_pass
        self.renderer = renderer
        self.use_modal = use_modal

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

        stop_reason = (extra or {}).get("stop_reason")
        if stop_reason == "length":
            reward = 0.0
            details = "overlong"
        else:
            patch = extract_patch(content)
            if not patch:
                reward = 0.0
                details = "no patch found"
            elif self.use_modal and self.repo and self.base_commit:
                passed, details = await run_swe_test_in_modal(
                    self.instance_id, self.repo, self.base_commit,
                    patch, self.fail_to_pass,
                )
                reward = 1.0 if passed else 0.0
            else:
                # Fallback: reward for producing a valid-looking patch
                reward = 0.5 if patch else 0.0
                details = "patch produced (no execution)"

        with logtree.scope_header("Problem"):
            logtree.log(f"Instance: {self.instance_id}")
            logtree.log(f"Repo: {self.repo}")
            logtree.log(f"Problem: {self.problem_statement[:500]}...")
        with logtree.scope_header("Response"):
            logtree.log(content[:1000])
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
            metrics={"correct": reward, "has_patch": float(bool(extract_patch(content)))},
        )


@dataclass(frozen=True)
class SWEGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], SWERLEnv]
    num_envs: int

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    def logging_tags(self) -> list[str]:
        return ["swe_rl"]


class SWERLDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        use_modal: bool = True,
        seed: int = 0,
    ):
        logger.info("Loading SWE-RL data from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset("nvidia/Nemotron-Cascade-2-RL-data", name="SWE-RL", split="train")
        ds = cast(Dataset, ds)
        # Filter to examples with actual SWE-bench data
        ds = ds.filter(lambda x: x.get("instance_id") is not None and x.get("problem_statement") is not None)
        self.ds = ds.shuffle(seed=seed)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.use_modal = use_modal
        logger.info(f"SWE-RL dataset: {len(self.ds)} valid instances")

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

            return SWEGroupBuilder(
                env_thunk=lambda iid=instance_id, p=problem, r=repo, bc=base_commit, ftp=fail_to_pass: SWERLEnv(
                    instance_id=iid, problem_statement=p, repo=r,
                    base_commit=bc, fail_to_pass=ftp,
                    renderer=self.renderer, use_modal=self.use_modal,
                ),
                num_envs=self.group_size,
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
    use_modal: bool = True
    seed: int = 0

    async def __call__(self) -> tuple[SWERLDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return SWERLDataset(
            batch_size=self.batch_size, group_size=self.group_size,
            renderer=renderer, use_modal=self.use_modal, seed=self.seed,
        ), None
