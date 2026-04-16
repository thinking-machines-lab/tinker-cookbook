"""
Evaluate Terminal-Bench tasks using CodeSandbox instead of Modal.

Prerequisites:
  1. export CSB_API_KEY=<your key>
  2. Docker must be running (for building task images)
  3. Download tasks: uvx harbor datasets download terminal-bench@2.0

Usage:
  cd /path/to/tinker-cookbook
  uv run python tinker_cookbook/recipes/harbor_rl/scripts/eval_terminal_bench_csb.py
"""

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import modal
import tinker

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.display import format_trajectory
from tinker_cookbook.recipes.harbor_rl.eval import EvalConfig, TaskResult
from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HarborTask,
    _initial_messages,
    load_harbor_tasks,
)
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborBashTool, HarborReward
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.sandbox.codesandbox_sandbox import CodeSandboxSandbox
from tinker_cookbook.tool_use import build_agent_tool_env
from tinker_cookbook.utils.ml_log import dump_config

logger = logging.getLogger(__name__)


async def evaluate_task_csb(
    task: HarborTask,
    policy: TinkerTokenCompleter,
    renderer,
    config: EvalConfig,
    results_dir: Path,
    lock: asyncio.Lock,
    tokenizer=None,
) -> TaskResult:
    """Evaluate a single task using CodeSandbox instead of Modal.

    This is a modified version of eval.evaluate_task that creates
    CodeSandbox sandboxes directly from the task's Dockerfile path,
    bypassing modal.Image entirely.
    """
    start = time.monotonic()

    env_dir = task.task_dir / "environment"
    dockerfile_path = str(env_dir / "Dockerfile")
    context_dir = str(env_dir)

    # Create a CSB sandbox directly from the Dockerfile
    # Use template alias for caching (avoids rebuilding on subsequent runs)
    task_name_slug = task.task_name.lower().replace("_", "-").replace(" ", "-")
    sandbox = await CodeSandboxSandbox.create(
        dockerfile_path=dockerfile_path,
        context_dir=context_dir,
        timeout=config.sandbox_timeout,
        tier=os.environ.get("CSB_TIER", "Micro"),
        template_alias=f"harbor@{task_name_slug}",
    )

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


async def run_eval_csb(config: EvalConfig, tasks: list[HarborTask]) -> list[TaskResult]:
    """Run evaluation using CodeSandbox sandboxes."""
    results_dir = Path(config.output_path) / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {results_dir}")

    config_dict = dump_config(config)
    config_dict["sandbox_backend"] = "codesandbox"
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

    max_concurrent = int(os.environ.get("CSB_CONCURRENCY", "4"))
    logger.info("Starting CodeSandbox evaluation of %d tasks (concurrency=%d)",
                len(tasks), max_concurrent)
    print(f"Evaluating {len(tasks)} tasks with CodeSandbox (max {max_concurrent} concurrent)...")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_limit(task):
        async with semaphore:
            print(f"  Starting: {task.task_name}")
            result = await evaluate_task_csb(
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
    if not os.environ.get("CSB_API_KEY"):
        print("ERROR: CSB_API_KEY environment variable not set.")
        print("  export CSB_API_KEY=<your CodeSandbox API key>")
        raise SystemExit(1)

    config = EvalConfig(
        max_turns=200,
        temperature=0.1,
        max_tokens=8192,
    )

    tasks = load_harbor_tasks("terminal-bench-2.0")
    print(f"Loaded {len(tasks)} tasks")
    print(f"Sandbox backend: CodeSandbox (tier={os.environ.get('CSB_TIER', 'Micro')})")

    asyncio.run(run_eval_csb(config, tasks))
