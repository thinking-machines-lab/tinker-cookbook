"""Shared metric computation for reward functions.

All domain-specific ``compute_*_metrics`` helpers delegate here so that
the aggregation logic is defined in exactly one place.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def compute_reward_metrics(
    rewards: Sequence[float],
    reward_name: str,
) -> dict[str, float]:
    """Compute aggregate metrics for a batch of reward values.

    Returns a dict with keys:

    - ``reward/{name}/mean``
    - ``reward/{name}/std``
    - ``reward/{name}/fraction_correct`` (fraction of values > 0.5)

    Args:
        rewards: Sequence of reward values (typically 0.0 or 1.0).
        reward_name: Name prefix for the metric keys.
    """
    if not rewards:
        return {}

    n = len(rewards)
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(variance)
    fraction_correct = sum(1.0 for r in rewards if r > 0.5) / n

    return {
        f"reward/{reward_name}/mean": mean,
        f"reward/{reward_name}/std": std,
        f"reward/{reward_name}/fraction_correct": fraction_correct,
    }
