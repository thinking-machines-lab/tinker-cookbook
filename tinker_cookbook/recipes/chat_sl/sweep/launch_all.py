"""
Launch LR + LoRA rank sweeps for all models in the Tinker lineup.

Runs sweeps in parallel both within each model (multiple LR/rank combos)
and across models (multiple model sweeps concurrently).

Usage::

    # Dry run — print what would be launched
    uv run python -m tinker_cookbook.recipes.chat_sl.sweep.launch_all --dry-run

    # Launch everything (default: 4 models concurrently, 3 jobs per model)
    uv run python -m tinker_cookbook.recipes.chat_sl.sweep.launch_all

    # Custom parallelism
    uv run python -m tinker_cookbook.recipes.chat_sl.sweep.launch_all \
        --models-parallel 2 --jobs-per-model 2

    # Launch only a specific size tier
    uv run python -m tinker_cookbook.recipes.chat_sl.sweep.launch_all --tier large

    # Launch a single model
    uv run python -m tinker_cookbook.recipes.chat_sl.sweep.launch_all \
        --only Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Model sweep configurations
# ---------------------------------------------------------------------------

# Consistent LR range across all tiers
LR_ALL = [4e-05, 1e-04, 3e-04, 1e-03, 3e-03]

# LoRA ranks by size tier (larger models → smaller ranks)
RANKS_LARGE = [1, 2, 4]
RANKS_MEDIUM = [1, 4, 16, 64]
RANKS_SMALL = [4, 16, 64, 128]
RANKS_COMPACT = [4, 16, 64, 128]

WANDB_PROJECT = "lr-sweep-2026-03"
DATASET = "tulu3"
BATCH_SIZE = 128
TRAINING_BUDGET = 100_000


@dataclass
class ModelSweepConfig:
    model_name: str
    tier: str  # large, medium, small, compact
    learning_rates: list[float]
    lora_ranks: list[int]
    note: str = ""  # e.g. "MoE", "Dense", etc.


def _cfg(model: str, tier: str, note: str = "") -> ModelSweepConfig:
    rank_map = {
        "large": RANKS_LARGE,
        "medium": RANKS_MEDIUM,
        "small": RANKS_SMALL,
        "compact": RANKS_COMPACT,
    }
    return ModelSweepConfig(
        model_name=model,
        tier=tier,
        learning_rates=LR_ALL,
        lora_ranks=rank_map[tier],
        note=note,
    )


# Models already swept (included for completeness, skipped by default)
ALREADY_DONE: dict[str, ModelSweepConfig] = {
    "deepseek-ai/DeepSeek-V3.1-Base": _cfg(
        "deepseek-ai/DeepSeek-V3.1-Base", "large", "MoE, already swept"
    ),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": _cfg(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "medium", "MoE, already swept"
    ),
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": _cfg(
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "large", "MoE, already swept"
    ),
}

# All new models to sweep, grouped by tier
NEW_MODELS: list[ModelSweepConfig] = [
    # --- Large ---
    _cfg("Qwen/Qwen3.5-397B-A17B", "large", "MoE Hybrid+Vision"),
    _cfg("Qwen/Qwen3-VL-235B-A22B-Instruct", "large", "MoE Vision"),
    _cfg("Qwen/Qwen3-235B-A22B-Instruct-2507", "large", "MoE Instruction"),
    _cfg("deepseek-ai/DeepSeek-V3.1", "large", "MoE Hybrid"),
    _cfg("meta-llama/Llama-3.1-70B", "large", "Dense Base"),
    _cfg("meta-llama/Llama-3.3-70B-Instruct", "large", "Dense Instruction"),
    _cfg("moonshotai/Kimi-K2-Thinking", "large", "MoE Reasoning"),
    _cfg("moonshotai/Kimi-K2.5", "large", "MoE Reasoning+Vision"),
    # --- Medium ---
    _cfg("Qwen/Qwen3.5-35B-A3B", "medium", "MoE Hybrid+Vision"),
    _cfg("Qwen/Qwen3.5-27B", "medium", "Dense Hybrid+Vision"),
    _cfg("Qwen/Qwen3-VL-30B-A3B-Instruct", "medium", "MoE Vision"),
    _cfg("Qwen/Qwen3-30B-A3B-Instruct-2507", "medium", "MoE Instruction"),
    _cfg("Qwen/Qwen3-30B-A3B", "medium", "MoE Hybrid"),
    _cfg("Qwen/Qwen3-30B-A3B-Base", "medium", "MoE Base"),
    _cfg("Qwen/Qwen3-32B", "medium", "Dense Hybrid"),
    _cfg("openai/gpt-oss-120b", "medium", "MoE Reasoning"),
    # --- Small ---
    _cfg("Qwen/Qwen3-8B", "small", "Dense Hybrid"),
    _cfg("Qwen/Qwen3-8B-Base", "small", "Dense Base"),
    _cfg("openai/gpt-oss-20b", "small", "MoE Reasoning"),
    _cfg("meta-llama/Llama-3.1-8B", "small", "Dense Base"),
    _cfg("meta-llama/Llama-3.1-8B-Instruct", "small", "Dense Instruction"),
    _cfg("Qwen/Qwen3.5-4B", "small", "Dense Hybrid+Vision"),
    # --- Compact ---
    _cfg("Qwen/Qwen3-4B-Instruct-2507", "compact", "Dense Instruction"),
    _cfg("meta-llama/Llama-3.2-3B", "compact", "Dense Base"),
    _cfg("meta-llama/Llama-3.2-1B", "compact", "Dense Base"),
]


# ---------------------------------------------------------------------------
# Sweep launcher
# ---------------------------------------------------------------------------


SWEEP_ROOT = "/tmp/tinker-sweeps"


def build_sweep_command(cfg: ModelSweepConfig, jobs_per_model: int) -> list[str]:
    """Build the CLI command for one model's sweep."""
    lr_str = ", ".join(str(lr) for lr in cfg.learning_rates)
    rank_str = ", ".join(str(r) for r in cfg.lora_ranks)

    # Use a model-specific sweep directory to avoid collisions when
    # multiple models launch concurrently with the same timestamp.
    model_slug = cfg.model_name.replace("/", "-")
    sweep_dir = f"{SWEEP_ROOT}/{model_slug}"

    return [
        sys.executable,
        "-m",
        "tinker_cookbook.recipes.chat_sl.sweep",
        "recipe=sft",
        f"base.model_name={cfg.model_name}",
        f"base.dataset={DATASET}",
        f"base.batch_size={BATCH_SIZE}",
        f"base.wandb_project={WANDB_PROJECT}",
        "base.behavior_if_log_dir_exists=delete",
        "metric=test/nll",
        f"training_budget_examples={TRAINING_BUDGET}",
        f"max_parallel={jobs_per_model}",
        f"sweep_dir={sweep_dir}",
        f"learning_rates=[{lr_str}]",
        f"lora_ranks=[{rank_str}]",
    ]


async def run_model_sweep(
    cfg: ModelSweepConfig,
    jobs_per_model: int,
    dry_run: bool,
    semaphore: asyncio.Semaphore,
    log_dir: str = "/tmp/tinker-sweep-logs",
) -> tuple[str, bool]:
    """Run sweep for one model, respecting the concurrency semaphore."""
    cmd = build_sweep_command(cfg, jobs_per_model)
    n_combos = len(cfg.learning_rates) * len(cfg.lora_ranks)
    cmd_str = " \\\n    ".join(cmd)

    if dry_run:
        print(f"\n[DRY RUN] {cfg.model_name} ({cfg.tier}, {n_combos} combos)")
        print(f"  {cmd_str}")
        return cfg.model_name, True

    async with semaphore:
        short_name = cfg.model_name.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"LAUNCHING: {cfg.model_name} ({cfg.tier}, {n_combos} combos, {jobs_per_model} parallel)")
        print(f"{'='*60}")

        # Write output to a log file instead of piping through asyncio.
        # Piping breaks when training data produces very long lines, which
        # fills the pipe buffer and hangs the subprocess.
        import os

        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{short_name}.log")
        log_file = open(log_path, "w")  # noqa: SIM115
        print(f"[{short_name}] Log: {log_path}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
        )

        await proc.wait()
        log_file.close()
        success = proc.returncode == 0

        status = "DONE" if success else "FAILED"
        print(f"[{short_name}] {status} (exit code {proc.returncode})")
        return cfg.model_name, success


async def run_all(
    models: list[ModelSweepConfig],
    models_parallel: int,
    jobs_per_model: int,
    dry_run: bool,
) -> None:
    """Launch all model sweeps with bounded concurrency."""
    semaphore = asyncio.Semaphore(models_parallel)

    total_combos = sum(len(m.learning_rates) * len(m.lora_ranks) for m in models)
    print(f"Sweep plan: {len(models)} models, {total_combos} total runs")
    print(f"Parallelism: {models_parallel} models concurrent, {jobs_per_model} jobs per model")
    print(f"W&B project: {WANDB_PROJECT}")
    print(f"Dataset: {DATASET}, batch_size: {BATCH_SIZE}, budget: {TRAINING_BUDGET}")

    tasks = [run_model_sweep(cfg, jobs_per_model, dry_run, semaphore) for cfg in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for result in results:
        if isinstance(result, Exception):
            print(f"  ERROR: {result}")
        else:
            name, success = result
            print(f"  {'OK' if success else 'FAIL'}: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch LR sweeps for all Tinker models")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--models-parallel",
        type=int,
        default=4,
        help="Max models to sweep concurrently (default: 4)",
    )
    parser.add_argument(
        "--jobs-per-model",
        type=int,
        default=3,
        help="Max parallel jobs within each model's sweep (default: 3)",
    )
    parser.add_argument(
        "--tier",
        choices=["large", "medium", "small", "compact"],
        help="Only launch models in this size tier",
    )
    parser.add_argument("--only", type=str, help="Launch only this model (exact HF name)")
    parser.add_argument(
        "--include-done",
        action="store_true",
        help="Also re-run already-completed models",
    )
    args = parser.parse_args()

    models = list(NEW_MODELS)
    if args.include_done:
        models.extend(ALREADY_DONE.values())

    if args.only:
        models = [m for m in models if m.model_name == args.only]
        if not models:
            all_names = [m.model_name for m in NEW_MODELS] + list(ALREADY_DONE.keys())
            print(f"Model '{args.only}' not found. Available:")
            for name in sorted(all_names):
                print(f"  {name}")
            sys.exit(1)
    elif args.tier:
        models = [m for m in models if m.tier == args.tier]

    if not models:
        print("No models to sweep.")
        sys.exit(0)

    asyncio.run(run_all(models, args.models_parallel, args.jobs_per_model, args.dry_run))


if __name__ == "__main__":
    main()
