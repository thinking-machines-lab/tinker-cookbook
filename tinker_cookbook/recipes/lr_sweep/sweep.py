"""
LR Sweep Recipe — find optimal learning rates for any recipe.

Works with any recipe that follows the ``CLIConfig`` + ``cli_main`` pattern.
Use short aliases for common recipes, or pass a full module path for any recipe.

Short aliases::

    python -m tinker_cookbook.recipes.lr_sweep.sweep recipe=sft
    python -m tinker_cookbook.recipes.lr_sweep.sweep recipe=math_rl
    python -m tinker_cookbook.recipes.lr_sweep.sweep recipe=code_rl
    python -m tinker_cookbook.recipes.lr_sweep.sweep recipe=dpo

Full module path (works for any recipe with CLIConfig + cli_main)::

    python -m tinker_cookbook.recipes.lr_sweep.sweep \\
        recipe=tinker_cookbook.recipes.harbor_rl.train \\
        base.model_name=Qwen/Qwen3-8B

Quick smoke test (5 steps)::

    python -m tinker_cookbook.recipes.lr_sweep.sweep \\
        recipe=sft training_budget_examples=640 \\
        'learning_rates=[1e-4, 3e-4, 1e-3]' 'lora_ranks=[32]'
"""

import asyncio
import importlib
import inspect
import json
import os
from collections.abc import Callable
from typing import Any

import chz

from tinker_cookbook import sweep

# ---------------------------------------------------------------------------
# Recipe resolution — short aliases + dynamic module import
# ---------------------------------------------------------------------------

# Short aliases for common recipes. Maps alias -> module path.
_ALIASES: dict[str, str] = {
    "sft": "tinker_cookbook.recipes.chat_sl.train",
    "math_rl": "tinker_cookbook.recipes.math_rl.train",
    "code_rl": "tinker_cookbook.recipes.code_rl.train",
    "harbor_rl": "tinker_cookbook.recipes.harbor_rl.train",
    "rubric": "tinker_cookbook.recipes.rubric.train",
    "search_tool": "tinker_cookbook.recipes.search_tool.train",
    "dpo": "tinker_cookbook.recipes.preference.dpo.train",
    "shorter": "tinker_cookbook.recipes.preference.shorter.train",
    "distillation": "tinker_cookbook.recipes.prompt_distillation.train",
}


def get_recipe(name: str) -> tuple[type, Callable[..., None]]:
    """Resolve a recipe by alias or module path.

    Returns ``(CLIConfig class, main function)``. The main function is
    guaranteed to be synchronous (async ``cli_main`` is wrapped automatically).
    """
    module_path = _ALIASES.get(name, name)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        available = ", ".join(sorted(_ALIASES.keys()))
        raise ValueError(
            f"Could not import recipe '{name}' (resolved to '{module_path}'). "
            f"Available aliases: {available}. "
            f"Or pass a full module path like 'tinker_cookbook.recipes.my_recipe.train'."
        ) from e

    # Find CLIConfig (or ExperimentConfig as fallback)
    config_cls = getattr(module, "CLIConfig", None) or getattr(module, "ExperimentConfig", None)
    if config_cls is None:
        raise ValueError(
            f"Recipe module '{module_path}' does not have a CLIConfig or ExperimentConfig class."
        )

    # Find cli_main (or run_experiment as fallback)
    main_fn = getattr(module, "cli_main", None) or getattr(module, "run_experiment", None)
    if main_fn is None:
        raise ValueError(
            f"Recipe module '{module_path}' does not have a cli_main or run_experiment function."
        )

    # Wrap async functions so sweep.run can call them synchronously
    if inspect.iscoroutinefunction(main_fn):
        original_fn = main_fn

        def sync_wrapper(config: Any) -> None:
            asyncio.run(original_fn(config))

        return config_cls, sync_wrapper

    return config_cls, main_fn


# ---------------------------------------------------------------------------
# Sweep config and entry point
# ---------------------------------------------------------------------------


@chz.chz
class LRSweepConfig:
    """Configuration for an LR sweep experiment.

    The ``recipe`` field selects which training recipe to sweep — use a short
    alias (``sft``, ``math_rl``, ``dpo``, etc.) or a full module path.
    The ``base`` field is the selected recipe's CLIConfig — all training
    parameters are inherited, no duplication.
    """

    recipe: str = "sft"

    # Base config for the selected recipe. Type is Any because the actual type
    # depends on which recipe is selected. chz handles this via CLI overrides.
    base: Any = None

    # Sweep axes
    learning_rates: list[float] = chz.field(
        default_factory=lambda: [1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 1e-3]
    )
    lora_ranks: list[int] = chz.field(default_factory=lambda: [32, 128])

    # Budget (controls max_steps = training_budget_examples // batch_size)
    training_budget_examples: int = 100_000

    # Metric to optimize (depends on recipe)
    metric: str = "train_mean_nll"


def _get_batch_size(config: Any) -> int:
    """Extract batch size from a recipe config, handling different field names."""
    if hasattr(config, "batch_size"):
        return config.batch_size
    if hasattr(config, "groups_per_batch"):
        return config.groups_per_batch
    return 128


def run_lr_sweep(config: LRSweepConfig) -> None:
    """Run an LR sweep and print the best learning rate per rank."""
    config_cls, main_fn = get_recipe(config.recipe)

    # Build base config: use provided base or create default
    if config.base is None:
        base = config_cls()
    else:
        base = config.base

    # Set max_steps from budget
    batch_size = _get_batch_size(base)
    max_steps = config.training_budget_examples // batch_size

    replace_kwargs: dict[str, Any] = {"max_steps": max_steps}
    if hasattr(base, "behavior_if_log_dir_exists"):
        replace_kwargs["behavior_if_log_dir_exists"] = "delete"
    base = chz.replace(base, **replace_kwargs)

    results = sweep.run(
        main_fn,
        base,
        learning_rate=config.learning_rates,
        lora_rank=config.lora_ranks,
    )

    if results.empty:
        print("No completed runs found.")
        return

    # Print results table
    metric = config.metric
    display_cols = [c for c in ["learning_rate", "lora_rank", metric] if c in results.columns]
    print("\n" + results[display_cols + ["log_path"]].to_string(index=False))

    if metric not in results.columns:
        print(
            f"\nWarning: metric '{metric}' not found in results. "
            f"Available: {[c for c in results.columns if c != 'log_path']}"
        )
        return

    # Find and print best LR per rank
    print(f"\n--- Best learning rate per rank (by {metric}) ---")
    recommendations: dict[str, dict[str, float | int]] = {}
    for rank_key, group in results.groupby("lora_rank"):
        rank = int(rank_key)  # type: ignore[arg-type]
        best = group.loc[group[metric].idxmin()]
        lr = float(best["learning_rate"])
        loss = float(best[metric])
        print(f"  rank={rank}: lr={lr:.2e} ({metric}={loss:.4f})")
        recommendations[f"rank_{rank}"] = {
            "learning_rate": lr,
            "lora_rank": rank,
            metric: loss,
        }

    # Write recommendations JSON
    sweep_dir = os.path.dirname(results["log_path"].iloc[0])
    rec_path = os.path.join(sweep_dir, "lr_recommendations.json")
    model_name = getattr(base, "model_name", "unknown")
    with open(rec_path, "w") as f:
        json.dump(
            {
                "recipe": config.recipe,
                "model_name": model_name,
                "metric": metric,
                "recommendations": recommendations,
            },
            f,
            indent=2,
        )
    print(f"\nRecommendations written to {rec_path}")


if __name__ == "__main__":
    chz.nested_entrypoint(run_lr_sweep)
