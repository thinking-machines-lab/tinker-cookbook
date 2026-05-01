"""WandB and metrics logging utilities."""

from __future__ import annotations

import json
import logging
from math import comb
from typing import Any

from tinker_cookbook.fireworks.utils.config import WandBConfig

logger = logging.getLogger(__name__)


def setup_wandb(wb: WandBConfig, config: dict[str, Any]) -> bool:
    """Initialize WandB if entity is provided. Returns True if active.

    If ``WANDB_API_KEY`` is not set, falls back to offline mode so runs
    are logged locally without requiring authentication.
    """
    if not wb.entity:
        return False
    try:
        import os
        import wandb

        if not os.environ.get("WANDB_API_KEY"):
            logger.warning(
                "WANDB_API_KEY not set; running wandb in offline mode. "
                "Set the key to sync runs to the dashboard."
            )
            os.environ["WANDB_MODE"] = "offline"

        wandb.init(entity=wb.entity, project=wb.project, name=wb.run_name, config=config)
        if wandb.run is not None:
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("perf/*", step_metric="train/step")
            wandb.define_metric("rollout/*", step_metric="train/step")
            wandb.define_metric("batch/*", step_metric="train/step")
            wandb.define_metric("infra/*", step_metric="train/step")
            logger.info("WandB: %s", wandb.run.url)
        return True
    except ImportError:
        logger.warning("wandb not installed; metrics will only be logged to console")
        return False


def compute_pass_at_k(
    reward_groups: list[list[float]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute pass@k from grouped reward samples.

    Each inner list contains per-sample rewards for one prompt group.
    A sample is "correct" if its reward > 0.5.
    """
    if k_values is None:
        k_values = [1]
    results: dict[str, float] = {}
    for k in k_values:
        pass_rates: list[float] = []
        for group_rewards in reward_groups:
            n = len(group_rewards)
            c = sum(1 for r in group_rewards if r > 0.5)
            if n < k:
                continue
            denom = comb(n, k)
            pass_rates.append(1.0 - comb(n - c, k) / denom if denom > 0 else 0.0)
        if pass_rates:
            results[f"rollout/pass@{k}"] = sum(pass_rates) / len(pass_rates)
    return results


def wandb_log(metrics: dict[str, Any], step: int) -> None:
    """Log metrics to WandB if available."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step, commit=True)
    except ImportError:
        pass


def wandb_finish() -> None:
    """Finish WandB run if active."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


def log_metrics_json(step: int, **kwargs: float) -> None:
    """Print a JSON metrics line to stdout (for structured log parsing)."""
    print(json.dumps({"type": "metrics", "step": step, **kwargs}), flush=True)
