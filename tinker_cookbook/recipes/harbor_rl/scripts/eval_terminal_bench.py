"""
Load Terminal-Bench tasks from the Harbor cache and run evaluation.

uv run python tinker_cookbook/recipes/harbor_rl/scripts/eval_terminal_bench.py

"""

import asyncio

from tinker_cookbook.recipes.harbor_rl.eval import EvalConfig, run_eval
from tinker_cookbook.recipes.harbor_rl.harbor_env import default_sandbox_factory, load_harbor_tasks

if __name__ == "__main__":
    config = EvalConfig(
        max_turns=200,
        temperature=0.1,
        max_tokens=8192,
    )
    tasks = load_harbor_tasks("terminal-bench-2.0")
    asyncio.run(run_eval(config, tasks, sandbox_factory=default_sandbox_factory))
