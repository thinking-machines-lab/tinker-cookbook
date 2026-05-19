"""
Load Terminal-Bench tasks from the Harbor cache and launch RL training.

uv run python tinker_cookbook/recipes/harbor_rl/scripts/train_terminal_bench.py

"""

import asyncio

import chz
import modal

from tinker_cookbook.recipes.harbor_rl.harbor_env import default_sandbox_factory, load_harbor_tasks
from tinker_cookbook.recipes.harbor_rl.train import CLIConfig, cli_main

if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    tasks = load_harbor_tasks("terminal-bench-2.0")
    with modal.enable_output():
        asyncio.run(cli_main(cli_config, tasks, sandbox_factory=default_sandbox_factory))
