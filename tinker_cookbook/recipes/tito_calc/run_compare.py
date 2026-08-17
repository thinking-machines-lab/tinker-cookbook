"""Parity comparison: cookbook's Llama3Renderer vs TITO via apply_chat_template.

Reuses the cookbook's ``math_rl`` arithmetic recipe end-to-end. Two runs of the
same recipe, same model, same seed, same hyperparams — only the renderer flag
differs:

- Run A: ``renderer_name=llama3`` — the cookbook's per-family Python
  reimplementation of the Llama 3 chat template.
- Run B: ``renderer_name=apply_chat_template`` — the model-agnostic
  ``TitoRenderer`` (defined in ``tito_renderers.py``), which delegates to
  ``tokenizer.apply_chat_template``. **Zero per-family code** — the same
  renderer class would work for any model whose chat template passes the
  §6 prefix-preservation property.

The blog's message: when the chat template is prefix-preserving for tool
messages (§6 holds), the renderer abstraction is optional —
``apply_chat_template`` does the job with no per-family Python. This recipe
demonstrates that on Llama 3.x, both arms train cleanly and the reward
trajectories track each other.

Run::

    python -m tinker_cookbook.recipes.tito_calc.run_compare \\
        [--model meta-llama/Llama-3.2-1B] [--max-steps 15] [--batch 4] [--group 4]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from tinker_cookbook.recipes.math_rl.train import CLIConfig as MathCLIConfig
from tinker_cookbook.recipes.math_rl.train import cli_main


async def _run_one(
    *, model: str, renderer_name: str, max_steps: int, batch: int, group: int, log_root: str
) -> str:
    log_path = os.path.join(log_root, renderer_name)
    cfg = MathCLIConfig(
        model_name=model,
        renderer_name=renderer_name,
        env="arithmetic",
        group_size=group,
        groups_per_batch=batch,
        learning_rate=1e-4,
        max_tokens=8,
        max_steps=max_steps,
        seed=0,
        log_path=log_path,
        behavior_if_log_dir_exists="overwrite",
    )
    print(f"\n========== run: renderer={renderer_name} ==========")
    print(f"log_path: {log_path}")
    await cli_main(cfg)
    return log_path


def _load_metrics(log_path: str) -> list[dict]:
    metrics_file = Path(log_path) / "metrics.jsonl"
    if not metrics_file.exists():
        return []
    rows = []
    for line in metrics_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _print_comparison(label_a: str, log_a: str, label_b: str, log_b: str) -> None:
    rows_a = _load_metrics(log_a)
    rows_b = _load_metrics(log_b)
    print()
    print("=" * 96)
    print(f"COMPARISON: A = {label_a}    B = {label_b}")
    print("=" * 96)
    keys = (
        "env/all/correct",
        "env/all/reward/total",
        "optim/kl_sample_train_v1",
        "optim/kl_sample_train_v2",
        "optim/entropy",
    )
    header = f"{'step':>5} " + " | ".join(f"{k:>26}" for k in keys)
    print(header)
    print("-" * len(header))
    for ra, rb in zip(rows_a, rows_b, strict=False):
        step = ra.get("progress/batch", "?")
        line_a = " | ".join(f"{ra.get(k, float('nan')):>26.6f}" for k in keys)
        line_b = " | ".join(f"{rb.get(k, float('nan')):>26.6f}" for k in keys)
        print(f"{step:>5} A: {line_a}")
        print(f"{step:>5} B: {line_b}")
    print()


async def amain(args: argparse.Namespace) -> int:
    log_a = await _run_one(
        model=args.model,
        renderer_name="llama3",
        max_steps=args.max_steps,
        batch=args.batch,
        group=args.group,
        log_root=args.log_root,
    )
    log_b = await _run_one(
        model=args.model,
        renderer_name="apply_chat_template",
        max_steps=args.max_steps,
        batch=args.batch,
        group=args.group,
        log_root=args.log_root,
    )
    _print_comparison(
        "cookbook Llama3Renderer (simplified)",
        log_a,
        "TitoRenderer (apply_chat_template)",
        log_b,
    )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--log-root", default="/tmp/tinker-examples/tito_compare")
    args = p.parse_args()
    sys.exit(asyncio.run(amain(args)))
