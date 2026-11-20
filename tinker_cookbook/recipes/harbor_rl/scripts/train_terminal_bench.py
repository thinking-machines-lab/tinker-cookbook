"""
Load Terminal-Bench tasks from the Harbor cache and launch RL training.
"""

import asyncio

import chz

from tinker_cookbook.recipes.harbor_rl.harbor_env import load_harbor_tasks, sandbox_factory_for
from tinker_cookbook.recipes.harbor_rl.train import CLIConfig, cli_main

TERMINAL_BENCH_DATASET = "terminal-bench-2.0/terminal-bench"


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    tasks = load_harbor_tasks(TERMINAL_BENCH_DATASET)
    sandbox_factory = sandbox_factory_for(cli_config.sandbox_backend)
    asyncio.run(cli_main(cli_config, tasks, sandbox_factory=sandbox_factory))
