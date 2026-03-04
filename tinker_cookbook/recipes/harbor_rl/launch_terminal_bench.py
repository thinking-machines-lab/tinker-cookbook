"""
Load Terminal-Bench tasks from the Harbor cache and launch RL training.

uv run python tinker_cookbook/recipes/harbor_rl/launch_terminal_bench.py

"""

import asyncio
import tomllib
from pathlib import Path

import chz

from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask, default_sandbox_factory
from tinker_cookbook.recipes.harbor_rl.train import CLIConfig, cli_main


def load_terminal_bench_tasks(
    tasks_dir: Path = Path.home() / ".cache" / "harbor" / "tasks",
) -> list[HarborTask]:
    """Load all Harbor tasks from a downloaded task directory.

    Each subdirectory under *tasks_dir* is a shortuuid hash containing a single
    child directory named after the task.  That child holds ``instruction.md``,
    ``task.toml``, ``environment/``, ``tests/``, and ``solution/``.
    """
    tasks: list[HarborTask] = []
    for uuid_dir in sorted(tasks_dir.iterdir()):
        # Each uuid dir contains exactly one task directory
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


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    tasks = load_terminal_bench_tasks()[:1]
    asyncio.run(cli_main(cli_config, tasks, sandbox_factory=default_sandbox_factory))
