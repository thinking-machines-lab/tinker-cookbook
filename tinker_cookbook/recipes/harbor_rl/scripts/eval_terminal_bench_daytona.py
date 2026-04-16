"""
Evaluate Terminal-Bench tasks using Daytona instead of Modal.

Prerequisites:
  1. uv pip install daytona-sdk
  2. export DAYTONA_API_KEY=<your key>
  3. Download tasks: uvx harbor datasets download terminal-bench@2.0

Usage:
  cd /path/to/tinker-cookbook
  uv run python tinker_cookbook/recipes/harbor_rl/scripts/eval_terminal_bench_daytona.py

Key difference from Modal/CSB: Daytona builds images REMOTELY from Dockerfiles.
No local Docker Desktop required.

This script is fully self-contained — no modal dependency anywhere in the import chain.
"""

import asyncio
import json
import logging
import os
import random
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import chz
import tinker

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.display import format_trajectory
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborBashTool, HarborReward
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.sandbox.daytona_sandbox import DaytonaSandbox
from tinker_cookbook.tool_use import build_agent_tool_env

logger = logging.getLogger(__name__)

# ── Types copied from harbor_env.py to avoid modal import ──────────────

HARBOR_CACHE_DIR = Path.home() / ".cache" / "harbor" / "tasks"
HARBOR_SYSTEM_PROMPT = (
    "You are a skilled software engineer working in a sandboxed environment. "
    "You have access to a bash tool to execute commands. "
    "Complete the task described by the user."
)


@dataclass(frozen=True)
class HarborTask:
    task_name: str
    instruction: str
    task_dir: Path
    config: dict[str, Any] = field(default_factory=dict)


def load_harbor_tasks(dataset: str) -> list[HarborTask]:
    tasks_dir = HARBOR_CACHE_DIR / dataset
    tasks: list[HarborTask] = []
    for uuid_dir in sorted(tasks_dir.iterdir()):
        (task_dir,) = [d for d in uuid_dir.iterdir() if d.is_dir()]
        tasks.append(
            HarborTask(
                task_name=task_dir.name,
                instruction=(task_dir / "instruction.md").read_text(),
                task_dir=task_dir,
                config=tomllib.loads((task_dir / "task.toml").read_text()),
            )
        )
    tasks.sort(key=lambda t: t.task_name)
    return tasks


def _initial_messages(task: HarborTask, renderer, bash_tool: HarborBashTool) -> list[Message]:
    tool_schemas = [bash_tool.bash.to_spec()]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tool_schemas,
        system_prompt=HARBOR_SYSTEM_PROMPT,
    )
    return prefix + [{"role": "user", "content": task.instruction}]


# ── Config and result types ────────────────────────────────────────────

@chz.chz
class EvalConfig:
    """Configuration for evaluation (standalone, no modal dependency)."""
    model_name: str = "moonshotai/Kimi-K2-Thinking"
    output_path: str = "tinker_cookbook/recipes/harbor_rl/scripts/results"
    max_turns: int = 10
    max_tokens: int = 2048
    temperature: float = 0.0
    sandbox_timeout: int = 3600
    command_timeout: int = 120
    grader_timeout: int = 60
    max_tasks: int | None = None
    checkpoint_url: str | None = None
    base_url: str | None = None
    renderer_name: str | None = None


@dataclass
class TaskResult:
    task_name: str
    reward: float
    reward_details: dict[str, float]
    turns_used: int
    time_seconds: float
    error: str | None = None
    trajectory_str: str | None = None


# ── Eval logic ─────────────────────────────────────────────────────────

async def evaluate_task_daytona(
    task: HarborTask,
    policy: TinkerTokenCompleter,
    renderer,
    config: EvalConfig,
    results_dir: Path,
    lock: asyncio.Lock,
    tokenizer=None,
) -> TaskResult:
    """Evaluate a single task using Daytona."""
    start = time.monotonic()

    env_dir = task.task_dir / "environment"
    dockerfile_path = str(env_dir / "Dockerfile")

    try:
        sandbox = await DaytonaSandbox.create(
            dockerfile_path=dockerfile_path,
            timeout=config.sandbox_timeout,
            cpu=int(os.environ.get("DAYTONA_CPU", "4")),
            memory=int(os.environ.get("DAYTONA_MEMORY", "8")),
        )
    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error("Sandbox creation failed for %s: %s", task.task_name, e)
        result = TaskResult(
            task_name=task.task_name,
            reward=0.0,
            reward_details={},
            turns_used=0,
            time_seconds=round(elapsed, 1),
            error=f"Sandbox creation failed: {e}",
        )
        status = "ERROR"
        summary_line = (
            f"{result.task_name:<40} {result.reward:>7.1f} {result.turns_used:>6} "
            f"{result.time_seconds:>8.1f} {status:>7}\n"
        )
        async with lock:
            with open(results_dir / "asummary.txt", "a") as f:
                f.write(summary_line)
            with open(results_dir / "aerr.txt", "a") as f:
                f.write(f"{'=' * 60}\nTask: {result.task_name}\n{'=' * 60}\n{result.error}\n\n")
        return result

    try:
        bash_tool = HarborBashTool(sandbox, command_timeout=config.command_timeout)
        reward_fn = HarborReward(
            tests_dir=task.task_dir / "tests",
            sandbox=sandbox,
            grader_timeout=config.grader_timeout,
        )

        env = build_agent_tool_env(
            renderer=renderer,
            tools=[bash_tool.bash],
            initial_messages=_initial_messages(task, renderer, bash_tool),
            reward_fn=reward_fn,
            max_turns=config.max_turns,
        )

        trajectory = await do_single_rollout(policy, env)
        reward = sum(t.reward for t in trajectory.transitions)
        reward_details = trajectory.transitions[-1].metrics if trajectory.transitions else {}
        turns_used = len(trajectory.transitions)
        elapsed = time.monotonic() - start

        trajectory_str = (
            format_trajectory(trajectory, tokenizer, only_last_transition=True)
            if tokenizer
            else None
        )

        result = TaskResult(
            task_name=task.task_name,
            reward=reward,
            reward_details=reward_details,
            turns_used=turns_used,
            time_seconds=round(elapsed, 1),
            trajectory_str=trajectory_str,
        )
    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error("Task %s failed: %s", task.task_name, e)
        result = TaskResult(
            task_name=task.task_name,
            reward=0.0,
            reward_details={},
            turns_used=0,
            time_seconds=round(elapsed, 1),
            error=str(e),
        )
    finally:
        try:
            await sandbox.cleanup()
        except Exception as e:
            logger.warning("Sandbox cleanup failed for %s: %s", task.task_name, e)

    # Write results
    status = "ERROR" if result.error else ("PASS" if result.reward > 0 else "FAIL")
    summary_line = (
        f"{result.task_name:<40} {result.reward:>7.1f} {result.turns_used:>6} "
        f"{result.time_seconds:>8.1f} {status:>7}\n"
    )

    async with lock:
        with open(results_dir / "asummary.txt", "a") as f:
            f.write(summary_line)
        if result.error:
            with open(results_dir / "aerr.txt", "a") as f:
                f.write(f"{'=' * 60}\n")
                f.write(f"Task: {result.task_name}\n")
                f.write(f"{'=' * 60}\n")
                f.write(f"{result.error}\n\n")
        if result.trajectory_str:
            (results_dir / f"{result.task_name}.txt").write_text(result.trajectory_str)

    return result


async def run_eval_daytona(config: EvalConfig, tasks: list[HarborTask]) -> list[TaskResult]:
    """Run evaluation using Daytona sandboxes."""
    results_dir = Path(config.output_path) / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {results_dir}")

    (results_dir / "config.json").write_text(json.dumps({
        "sandbox_backend": "daytona",
        "model_name": config.model_name,
        "max_turns": config.max_turns,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }, indent=2))

    lock = asyncio.Lock()

    service_client = tinker.ServiceClient(base_url=config.base_url)
    if config.checkpoint_url:
        sampling_client = service_client.create_sampling_client(
            model_path=config.checkpoint_url,
            base_model=config.model_name,
        )
    else:
        sampling_client = service_client.create_sampling_client(base_model=config.model_name)

    tokenizer = tokenizer_utils.get_tokenizer(config.model_name)
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )
    renderer = get_renderer(renderer_name, tokenizer)

    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    if config.max_tasks is not None:
        tasks = random.sample(tasks, min(config.max_tasks, len(tasks)))

    max_concurrent = int(os.environ.get("DAYTONA_CONCURRENCY", "4"))
    print(f"Evaluating {len(tasks)} tasks with Daytona (max {max_concurrent} concurrent)...")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_limit(task):
        async with semaphore:
            print(f"  Starting: {task.task_name}")
            result = await evaluate_task_daytona(
                task, policy, renderer, config, results_dir, lock, tokenizer
            )
            status = "ERROR" if result.error else ("PASS" if result.reward > 0 else "FAIL")
            print(f"  Finished: {task.task_name} -> {status} ({result.time_seconds}s)")
            return result

    task_results = list(
        await asyncio.gather(*[run_with_limit(task) for task in tasks])
    )

    # Print summary
    passed = sum(1 for r in task_results if r.reward > 0)
    failed = sum(1 for r in task_results if r.reward == 0 and not r.error)
    errors = sum(1 for r in task_results if r.error)
    print(f"\nResults: {passed} PASS / {failed} FAIL / {errors} ERROR out of {len(task_results)}")

    return task_results


if __name__ == "__main__":
    if not os.environ.get("DAYTONA_API_KEY"):
        print("ERROR: DAYTONA_API_KEY environment variable not set.")
        print("  export DAYTONA_API_KEY=<your Daytona API key>")
        raise SystemExit(1)

    config = EvalConfig(
        max_turns=200,
        temperature=0.1,
        max_tokens=8192,
    )

    tasks = load_harbor_tasks("terminal-bench-2.0")
    print(f"Loaded {len(tasks)} tasks")
    print(f"Sandbox backend: Daytona")
    print(f"  CPU: {os.environ.get('DAYTONA_CPU', '4')}")
    print(f"  Memory: {os.environ.get('DAYTONA_MEMORY', '8')}GB")
    print(f"  Concurrency: {os.environ.get('DAYTONA_CONCURRENCY', '4')}")
    print(f"  No local Docker required (Daytona builds remotely)")

    asyncio.run(run_eval_daytona(config, tasks))
