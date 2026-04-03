"""Analyze and compare results from loss function comparison runs.

Reads metrics.jsonl from each run's log directory and produces a comparison
table showing convergence speed, final accuracy, and training stability.

Usage:
    # Analyze runs from the default sweep output directory
    python -m tinker_cookbook.recipes.loss_fn_comparison.analyze

    # Analyze runs from a custom directory
    python -m tinker_cookbook.recipes.loss_fn_comparison.analyze --log-dir ~/experiments/loss_fn_comparison/arithmetic
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def read_metrics(metrics_path: Path) -> list[dict]:
    records = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_key_metrics(records: list[dict]) -> dict:
    """Extract key metrics from a training run."""
    steps = []
    rewards = []
    losses = []
    correct_fracs = []

    for r in records:
        step = r.get("step")
        if step is None:
            continue
        steps.append(step)
        if "env/all/reward/total" in r:
            rewards.append(r["env/all/reward/total"])
        if "loss:sum" in r:
            losses.append(r["loss:sum"])
        if "env/all/correct" in r:
            correct_fracs.append(r["env/all/correct"])

    if not steps:
        return {"error": "No metrics found"}

    result: dict = {
        "total_steps": max(steps),
    }

    if rewards:
        result["reward_first"] = rewards[0]
        result["reward_last"] = rewards[-1]
        result["reward_max"] = max(rewards)
        result["reward_mean"] = float(np.mean(rewards))
        # Stability: std of reward over last 25% of training
        tail = rewards[len(rewards) * 3 // 4 :]
        result["reward_tail_std"] = float(np.std(tail)) if len(tail) > 1 else 0.0

    if correct_fracs:
        result["correct_first"] = correct_fracs[0]
        result["correct_last"] = correct_fracs[-1]
        result["correct_max"] = max(correct_fracs)

    if losses:
        result["loss_first"] = losses[0]
        result["loss_last"] = losses[-1]

    return result


def format_table(results: dict[str, dict]) -> str:
    """Format results as a comparison table."""
    lines = []

    # Header
    loss_fns = list(results.keys())
    header = f"{'Metric':<30}" + "".join(f"{name:>20}" for name in loss_fns)
    lines.append(header)
    lines.append("-" * len(header))

    # Rows for each metric
    metric_labels = [
        ("total_steps", "Total steps"),
        ("reward_first", "Reward (first step)"),
        ("reward_last", "Reward (last step)"),
        ("reward_max", "Reward (max)"),
        ("reward_tail_std", "Reward stability (tail std)"),
        ("correct_first", "Correct % (first step)"),
        ("correct_last", "Correct % (last step)"),
        ("correct_max", "Correct % (max)"),
        ("loss_first", "Loss (first step)"),
        ("loss_last", "Loss (last step)"),
    ]

    for key, label in metric_labels:
        values = []
        for name in loss_fns:
            val = results[name].get(key)
            if val is None:
                values.append("N/A")
            elif isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))

        row = f"{label:<30}" + "".join(f"{v:>20}" for v in values)
        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze loss function comparison results")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/tmp/tinker-examples/math_rl",
        help="Parent directory containing per-loss-function run directories",
    )
    parser.add_argument(
        "--loss-fns",
        type=str,
        default="importance_sampling,ppo,cispo,dro",
        help="Comma-separated list of loss function names to compare",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    loss_fn_names = [s.strip() for s in args.loss_fns.split(",")]

    results: dict[str, dict] = {}

    for name in loss_fn_names:
        # Find the most recent run directory matching this loss function
        matching_dirs = sorted(log_dir.glob(f"*-{name}-*"), key=lambda p: p.stat().st_mtime)
        if not matching_dirs:
            # Also check for exact subdirectory name
            exact = log_dir / name
            if exact.is_dir():
                matching_dirs = [exact]

        if not matching_dirs:
            print(f"Warning: No run directory found for loss_fn={name} in {log_dir}")
            continue

        run_dir = matching_dirs[-1]  # Most recent
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            # Check subdirectories (the log_path includes iteration dirs)
            metrics_path = run_dir / "metrics.jsonl"
            if not metrics_path.exists():
                print(f"Warning: No metrics.jsonl found in {run_dir}")
                continue

        print(f"Reading {name}: {metrics_path}")
        records = read_metrics(metrics_path)
        results[name] = extract_key_metrics(records)

    if not results:
        print("No results found. Make sure training runs have completed.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("LOSS FUNCTION COMPARISON")
    print("=" * 80 + "\n")
    print(format_table(results))
    print()

    # Summary
    if len(results) > 1 and all("reward_last" in v for v in results.values()):
        best = max(results.items(), key=lambda kv: kv[1].get("reward_last", 0))
        most_stable = min(results.items(), key=lambda kv: kv[1].get("reward_tail_std", float("inf")))
        print(f"Best final reward:     {best[0]} ({best[1]['reward_last']:.4f})")
        print(f"Most stable (low std): {most_stable[0]} ({most_stable[1]['reward_tail_std']:.4f})")


if __name__ == "__main__":
    main()
