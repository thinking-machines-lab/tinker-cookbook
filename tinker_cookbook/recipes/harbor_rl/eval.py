"""
Standalone evaluation for Harbor tasks.

Download harbor datasets:
  uvx harbor datasets download swebench-verified@1.0 -o ~/.cache/harbor/tasks/swebench-verified-1.0
  uvx harbor datasets download terminal-bench-2.0 -o ~/.cache/harbor/tasks/terminal-bench-2.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import chz
import tinker

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.display import format_trajectory
from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HarborTask,
    SandboxFactory,
    _initial_messages,
    default_sandbox_factory,
)
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborBashTool, HarborReward
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tool_use import build_agent_tool_env
from tinker_cookbook.utils.ml_log import dump_config

logger = logging.getLogger(__name__)


@chz.chz
class EvalConfig:
    """Configuration for Harbor evaluation."""

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


async def evaluate_task(
    task: HarborTask,
    policy: TinkerTokenCompleter,
    renderer: Renderer,
    sandbox_factory: SandboxFactory,
    config: EvalConfig,
    results_dir: Path,
    lock: asyncio.Lock,
    tokenizer: tokenizer_utils.Tokenizer | None = None,
) -> TaskResult:
    """Evaluate a single task: create sandbox, run agent loop, grade, cleanup.

    Writes results to files in results_dir as soon as the task completes.
    """
    start = time.monotonic()
    env_dir = task.task_dir / "environment"

    # NB: sandbox_factory is inside the try/except. If sandbox creation
    # raises (image build OOM, fishbowl timeout, etc.) the exception would
    # otherwise escape to asyncio.gather and cancel every in-flight task.
    sandbox = None
    try:
        sandbox = await sandbox_factory(env_dir, config.sandbox_timeout)
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
        if sandbox is not None:
            try:
                await sandbox.cleanup()
            except Exception as e:
                logger.warning("Sandbox cleanup failed for %s: %s", task.task_name, e)

    # Write results to files immediately
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


async def run_eval(
    config: EvalConfig,
    tasks: list[HarborTask],
    sandbox_factory: SandboxFactory = default_sandbox_factory,
) -> list[TaskResult]:
    """Run evaluation on a list of Harbor tasks.

    Results are written to files in <output_path>/<timestamp>/ as each task completes.

    Args:
        config: Evaluation configuration.
        tasks: List of HarborTask to evaluate.
        sandbox_factory: Factory for creating sandboxes (defaults to Modal).

    Returns:
        List of per-task results.
    """
    results_dir = Path(config.output_path) / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {results_dir}")

    config_dict = dump_config(config)
    (results_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

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

    logger.info("Starting evaluation of %d tasks", len(tasks))

    task_results = list(
        await asyncio.gather(
            *[
                evaluate_task(
                    task,
                    policy,
                    renderer,
                    sandbox_factory,
                    config,
                    results_dir,
                    lock,
                    tokenizer,
                )
                for task in tasks
            ]
        )
    )

    return task_results
