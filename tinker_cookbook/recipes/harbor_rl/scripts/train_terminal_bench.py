"""
Load Terminal-Bench tasks from the Harbor cache and launch RL training.
"""

import asyncio

import chz

from tinker_cookbook.recipes.harbor_rl.harbor_env import default_sandbox_factory, load_harbor_tasks
from tinker_cookbook.recipes.harbor_rl.train import CLIConfig, cli_main

TERMINAL_BENCH_DATASET = "terminal-bench-2.0/terminal-bench"


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    tasks = load_harbor_tasks(TERMINAL_BENCH_DATASET)
    asyncio.run(cli_main(cli_config, tasks, sandbox_factory=default_sandbox_factory))
