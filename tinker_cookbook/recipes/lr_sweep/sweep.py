"""
LR Sweep Recipe — find optimal learning rates for any supported model.

Usage::

    uv run python -m tinker_cookbook.recipes.lr_sweep.sweep
    uv run python -m tinker_cookbook.recipes.lr_sweep.sweep base.model_name=openai/gpt-oss-20b
    uv run python -m tinker_cookbook.recipes.lr_sweep.sweep \\
        base.model_name=Qwen/Qwen3.5-4B \\
        'learning_rates=[1e-4, 3e-4, 1e-3]'
"""

import json
import os

import chz

from tinker_cookbook import sweep
from tinker_cookbook.recipes.chat_sl.train import CLIConfig, cli_main


@chz.chz
class LRSweepConfig:
    """Configuration for an LR sweep experiment.

    Composes with the existing ``CLIConfig`` from ``chat_sl`` — no duplication
    of training parameters.
    """

    base: CLIConfig = chz.field(
        default_factory=lambda: CLIConfig(
            model_name="Qwen/Qwen3.5-4B",
            dataset="tulu3",
            batch_size=128,
            max_length=8192,
        )
    )

    # Sweep axes
    learning_rates: list[float] = chz.field(
        default_factory=lambda: [1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 1e-3]
    )
    lora_ranks: list[int] = chz.field(default_factory=lambda: [32, 128])

    # Budget
    training_budget_examples: int = 100_000


def run_lr_sweep(config: LRSweepConfig) -> None:
    """Run an LR sweep and print the best learning rate per rank."""
    max_steps = config.training_budget_examples // config.base.batch_size
    base = chz.replace(
        config.base,
        max_steps=max_steps,
        behavior_if_log_dir_exists="delete",
    )

    results = sweep.run(
        cli_main,
        base,
        learning_rate=config.learning_rates,
        lora_rank=config.lora_ranks,
    )

    if results.empty:
        print("No completed runs found.")
        return

    # Print results table
    display_cols = [
        c for c in ["learning_rate", "lora_rank", "train_mean_nll"] if c in results.columns
    ]
    print("\n" + results[display_cols + ["log_path"]].to_string(index=False))

    # Find and print best LR per rank
    print("\n--- Best learning rate per rank ---")
    recommendations: dict[str, dict[str, float | int]] = {}
    for rank_key, group in results.groupby("lora_rank"):
        rank = int(rank_key)  # type: ignore[arg-type]
        best = group.loc[group["train_mean_nll"].idxmin()]
        lr = float(best["learning_rate"])
        loss = float(best["train_mean_nll"])
        print(f"  rank={rank}: lr={lr:.2e} (loss={loss:.4f})")
        recommendations[f"rank_{rank}"] = {
            "learning_rate": lr,
            "lora_rank": rank,
            "loss": loss,
        }

    # Write recommendations JSON
    sweep_dir = os.path.dirname(results["log_path"].iloc[0])
    rec_path = os.path.join(sweep_dir, "lr_recommendations.json")
    with open(rec_path, "w") as f:
        json.dump(
            {"model_name": base.model_name, "recommendations": recommendations},
            f,
            indent=2,
        )
    print(f"\nRecommendations written to {rec_path}")


if __name__ == "__main__":
    chz.nested_entrypoint(run_lr_sweep)
