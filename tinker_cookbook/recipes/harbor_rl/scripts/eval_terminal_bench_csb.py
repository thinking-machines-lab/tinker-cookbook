"""Evaluate Terminal-Bench tasks using CodeSandbox instead of Modal.

Prerequisites:
  1. export CSB_API_KEY=<your key>
  2. Docker must be running (for building task images)
  3. Download tasks: uvx harbor datasets download terminal-bench@2.0
  4. Create symlinks if needed (see Cookbook 06 in sandbox-tests repo)

Usage:
  cd /path/to/tinker-cookbook
  uv run python tinker_cookbook/recipes/harbor_rl/scripts/eval_terminal_bench_csb.py
"""

import asyncio
import os
from pathlib import Path

from tinker_cookbook.recipes.harbor_rl.eval import EvalConfig, run_eval
from tinker_cookbook.recipes.harbor_rl.harbor_env import load_harbor_tasks
from tinker_cookbook.sandbox.codesandbox_sandbox import CodeSandboxSandbox
from tinker_cookbook.sandbox.sandbox_interface import SandboxInterface


async def csb_sandbox_factory(env_dir: Path, timeout: int) -> SandboxInterface:
    """Create a CodeSandbox sandbox from a task's environment directory."""
    dockerfile_path = str(env_dir / "Dockerfile")
    context_dir = str(env_dir)
    task_name = env_dir.parent.name.lower().replace("_", "-").replace(" ", "-")

    return await CodeSandboxSandbox.create(
        dockerfile_path=dockerfile_path,
        context_dir=context_dir,
        timeout=timeout,
        cpu=int(os.environ.get("CSB_CPU", "2")),
        memory_mb=int(os.environ.get("CSB_MEMORY_MB", "2048")),
        disk_mb=int(os.environ.get("CSB_DISK_MB", "10240")),
        snapshot_alias=f"harbor@{task_name}",
    )


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
    print(f"Sandbox backend: CodeSandbox "
          f"(cpu={os.environ.get('CSB_CPU', '2')}, "
          f"mem={os.environ.get('CSB_MEMORY_MB', '2048')}MB)")

    asyncio.run(run_eval(config, tasks, sandbox_factory=csb_sandbox_factory))
