"""Run RL training sequentially with different loss functions.

Compares Tinker's built-in RL loss functions (importance_sampling, ppo, cispo, dro)
on the same task by running each one after the other.

Usage:
    # Quick comparison on arithmetic
    python -m tinker_cookbook.recipes.loss_fn_comparison.sweep

    # Comparison on GSM8K (slower, more realistic)
    python -m tinker_cookbook.recipes.loss_fn_comparison.sweep env=gsm8k model_name="Qwen/Qwen3-8B" max_tokens=512

    # Run only a subset of loss functions
    python -m tinker_cookbook.recipes.loss_fn_comparison.sweep loss_fns=ppo,cispo
"""

import asyncio
import logging
from typing import Any

import chz

from tinker_cookbook.recipes.math_rl.train import CLIConfig, cli_main

logger = logging.getLogger(__name__)

# Loss functions to compare, with their recommended configs.
LOSS_FN_CONFIGS: dict[str, dict[str, Any] | None] = {
    "importance_sampling": None,
    "ppo": None,
    "cispo": None,
    "dro": None,
}


@chz.chz
class SweepConfig:
    """Configuration for the loss function comparison sweep."""

    # Model and environment (passed through to CLIConfig)
    model_name: str = "meta-llama/Llama-3.2-1B"
    env: str = "arithmetic"
    group_size: int = 4
    groups_per_batch: int = 100
    learning_rate: float = 1e-4
    max_tokens: int = 5
    lora_rank: int = 32
    max_steps: int = 50

    # Which loss functions to include (comma-separated, or "all")
    loss_fns: str = "all"

    # Sweep infrastructure
    wandb_project: str | None = None
    seed: int = 0


def _build_cli_config(
    sweep: SweepConfig, loss_fn: str, loss_fn_config: dict[str, Any] | None
) -> CLIConfig:
    return CLIConfig(
        model_name=sweep.model_name,
        env=sweep.env,
        group_size=sweep.group_size,
        groups_per_batch=sweep.groups_per_batch,
        learning_rate=sweep.learning_rate,
        max_tokens=sweep.max_tokens,
        lora_rank=sweep.lora_rank,
        max_steps=sweep.max_steps,
        loss_fn=loss_fn,
        loss_fn_config=loss_fn_config,
        wandb_project=sweep.wandb_project,
        wandb_name=f"loss-cmp-{loss_fn}" if sweep.wandb_project else None,
        seed=sweep.seed,
        behavior_if_log_dir_exists="overwrite",
    )


async def run_sweep(sweep: SweepConfig) -> None:
    # Parse which loss functions to run
    if sweep.loss_fns == "all":
        selected = list(LOSS_FN_CONFIGS.keys())
    else:
        selected = [s.strip() for s in sweep.loss_fns.split(",")]
        for name in selected:
            if name not in LOSS_FN_CONFIGS:
                raise ValueError(
                    f"Unknown loss function: {name!r}. "
                    f"Choose from: {list(LOSS_FN_CONFIGS.keys())}"
                )

    for i, loss_fn in enumerate(selected):
        logger.info(f"[{i + 1}/{len(selected)}] Training with loss_fn={loss_fn}")
        cli_config = _build_cli_config(sweep, loss_fn, LOSS_FN_CONFIGS[loss_fn])
        await cli_main(cli_config)
        logger.info(f"[{i + 1}/{len(selected)}] Finished loss_fn={loss_fn}")

    logger.info(
        f"All {len(selected)} runs complete. "
        f"Use analyze.py to compare results, or check metrics.jsonl in each log directory."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sweep_config = chz.entrypoint(SweepConfig)
    asyncio.run(run_sweep(sweep_config))
