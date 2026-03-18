"""Reusable sweep infrastructure for parameter grid searches.

Usage::

    from tinker_cookbook import sweep

    results = sweep.run(
        cli_main,
        CLIConfig(model_name="Qwen/Qwen3.5-4B"),
        learning_rate=[1e-4, 3e-4, 1e-3],
        lora_rank=[32, 128],
    )
"""

from tinker_cookbook.sweep.grid import default_run_name, grid
from tinker_cookbook.sweep.results import collect
from tinker_cookbook.sweep.runner import run

__all__ = ["collect", "default_run_name", "grid", "run"]
