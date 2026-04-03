"""Launch parallel RL training runs with different loss functions.

Compares Tinker's built-in RL loss functions (importance_sampling, ppo, cispo, dro)
on the same task. Uses xmux to launch all runs in parallel tmux panes.

Usage:
    # Quick comparison on arithmetic (< 5 min)
    python -m tinker_cookbook.recipes.loss_fn_comparison.sweep

    # Comparison on GSM8K (slower, more realistic)
    python -m tinker_cookbook.recipes.loss_fn_comparison.sweep env=gsm8k model_name="Qwen/Qwen3-8B" max_tokens=512

    # Dry run (show commands without executing)
    python -m tinker_cookbook.recipes.loss_fn_comparison.sweep --dry-run
"""

import asyncio
import sys
from typing import Any

import chz

from tinker_cookbook.recipes.math_rl.train import CLIConfig, cli_main
from tinker_cookbook.xmux import JobSpec, SwarmConfig, launch_swarm

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


def _build_cli_config(sweep: SweepConfig, loss_fn: str, loss_fn_config: dict[str, Any] | None) -> CLIConfig:
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


def _main_fn(cli_config: CLIConfig) -> None:
    asyncio.run(cli_main(cli_config))


def run_sweep(sweep: SweepConfig, dry_run: bool = False) -> None:
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

    job_specs = []
    for loss_fn in selected:
        cli_config = _build_cli_config(sweep, loss_fn, LOSS_FN_CONFIGS[loss_fn])
        job_specs.append(
            JobSpec(
                main_fn=_main_fn,
                log_relpath=f"loss_fn_comparison/{sweep.env}/{loss_fn}",
                entrypoint_config=cli_config,
            )
        )

    config = SwarmConfig(
        sweep_name=f"loss-fn-cmp-{sweep.env}",
        dry_run=dry_run,
    )
    launch_swarm(job_specs, config)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--dry-run"]
    # Re-inject cleaned argv so chz.entrypoint sees only config args
    sys.argv = [sys.argv[0]] + argv
    sweep_config = chz.entrypoint(SweepConfig)
    run_sweep(sweep_config, dry_run=dry_run)
