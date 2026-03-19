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
import sys
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
# Sweep config — built dynamically based on the selected recipe
# ---------------------------------------------------------------------------


def _make_sweep_config_cls(recipe_config_cls: type) -> type:
    """Dynamically create an LRSweepConfig with a concrete ``base`` field type.

    This is necessary because ``chz`` needs a concrete type for ``base`` at
    parse time, but we don't know which recipe is selected until we read ``recipe``
    from the CLI arguments.
    """

    @chz.chz
    class LRSweepConfig:
        """Configuration for an LR sweep experiment."""

        base: recipe_config_cls = chz.field(default_factory=recipe_config_cls)  # type: ignore[valid-type]

        # Sweep axes
        learning_rates: list[float] = chz.field(
            default_factory=lambda: [1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 1e-3]
        )
        lora_ranks: list[int] = chz.field(default_factory=lambda: [32, 128])

        # Budget (controls max_steps = training_budget_examples // batch_size)
        training_budget_examples: int = 100_000

        # Execution
        max_parallel: int = 1

        # Metric to optimize (depends on recipe)
        metric: str = "train_mean_nll"

    return LRSweepConfig


def _get_batch_size(config: Any) -> int:
    """Extract batch size from a recipe config, handling different field names."""
    if hasattr(config, "batch_size"):
        return config.batch_size
    if hasattr(config, "groups_per_batch"):
        return config.groups_per_batch
    return 128


def _extract_recipe_from_argv(argv: list[str]) -> str:
    """Extract the recipe= argument from argv without consuming it."""
    for arg in argv:
        if arg.startswith("recipe="):
            return arg[len("recipe=") :]
    return "sft"


def run_lr_sweep(config: Any, recipe_name: str) -> None:
    """Run an LR sweep and print the best learning rate per rank."""
    _, main_fn = get_recipe(recipe_name)
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
        max_parallel=config.max_parallel,
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
                "recipe": recipe_name,
                "model_name": model_name,
                "metric": metric,
                "recommendations": recommendations,
            },
            f,
            indent=2,
        )
    print(f"\nRecommendations written to {rec_path}")


if __name__ == "__main__":
    # Two-phase parsing: extract recipe from argv first to determine the
    # concrete type for the `base` field, then let chz parse everything.
    recipe_name = _extract_recipe_from_argv(sys.argv[1:])
    config_cls, _ = get_recipe(recipe_name)
    sweep_config_cls = _make_sweep_config_cls(config_cls)

    # Filter out recipe= from argv since it's not a field on the dynamic config
    filtered_argv = [a for a in sys.argv[1:] if not a.startswith("recipe=")]

    sweep_config = chz.Blueprint(sweep_config_cls).make_from_argv(filtered_argv)
    run_lr_sweep(sweep_config, recipe_name)
