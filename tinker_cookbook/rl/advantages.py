"""Advantage estimation functions for RL training.

Provides pluggable advantage estimators used by the RL training pipeline.
Each estimator takes trajectory groups and returns per-trajectory advantage
values that are used to weight the policy gradient.

Supported estimators:
- **GRPO**: Group-relative advantages (mean-centered within each group).
- **REINFORCE++**: Baseline-subtracted REINFORCE with optional std normalization.

Note: GAE (Generalized Advantage Estimation) will be added when critic/value
model support is available in the Tinker API.
"""

from __future__ import annotations

import logging
from enum import Enum
import numpy as np
import torch

from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.utils import logtree, trace

logger = logging.getLogger(__name__)


class AdvantageMethod(str, Enum):
    """Supported advantage estimation methods.

    Each value corresponds to one of the ``compute_*_advantages`` functions
    in this module.
    """

    GRPO = "grpo"
    REINFORCE_PP = "reinforce_pp"


def compute_grpo_advantages(
    trajectory_groups_P: list[TrajectoryGroup],
) -> list[torch.Tensor]:
    """Compute GRPO advantages: mean-centered rewards within each group.

    This is the original advantage computation used by GRPO. For each group,
    the advantage of trajectory *i* is ``R_i - mean(R)`` where ``R`` is the
    vector of total rewards for the group.

    Args:
        trajectory_groups_P: Groups of trajectories, where each group's
            rewards are centered independently.

    Returns:
        Per-group advantage tensors of shape ``(G,)``, where ``G`` is the
        number of trajectories in each group.
    """
    advantages_P: list[torch.Tensor] = []
    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards())
        advantages_G = rewards_G - rewards_G.mean()
        advantages_P.append(advantages_G)
    return advantages_P


def compute_reinforce_pp_advantages(
    trajectory_groups_P: list[TrajectoryGroup],
    *,
    normalize: bool = True,
    eps: float = 1e-8,
) -> list[torch.Tensor]:
    """Compute REINFORCE++ advantages: baseline-subtracted with optional normalization.

    For each group, computes ``A_i = R_i - mean(R)``. When ``normalize=True``
    (the default), the advantages are further divided by ``std(R) + eps``,
    producing unit-variance advantages that can stabilize training when reward
    scales vary across groups.

    REINFORCE++ is conceptually simpler than GRPO and sometimes works better
    because the std normalization prevents high-variance groups from
    dominating the gradient.

    Args:
        trajectory_groups_P: Groups of trajectories.
        normalize: If True, divide by ``std(R) + eps`` within each group.
            Defaults to True.
        eps: Small constant added to std to avoid division by zero.

    Returns:
        Per-group advantage tensors of shape ``(G,)``.
    """
    advantages_P: list[torch.Tensor] = []
    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards(), dtype=torch.float32)
        mean = rewards_G.mean()
        advantages_G = rewards_G - mean
        if normalize and len(rewards_G) > 1:
            std = rewards_G.std(correction=0)
            advantages_G = advantages_G / (std + eps)
        advantages_P.append(advantages_G)
    return advantages_P


def _log_advantage_stats(
    advantages_P: list[torch.Tensor],
    method: AdvantageMethod,
    trajectory_groups_P: list[TrajectoryGroup] | None = None,
) -> dict[str, float]:
    """Compute and log advantage statistics for telemetry.

    Args:
        advantages_P: Per-group advantage tensors.
        method: The advantage method used (for logtree display).
        trajectory_groups_P: Original trajectory groups, used to compute
            method-specific baselines (e.g. mean reward for REINFORCE++).

    Returns:
        Dict of advantage metrics suitable for ml_log.
    """
    if not advantages_P:
        return {}

    all_advantages = torch.cat(advantages_P)
    stats: dict[str, float] = {
        "advantages/mean": all_advantages.mean().item(),
        "advantages/std": all_advantages.std().item() if len(all_advantages) > 1 else 0.0,
        "advantages/max": all_advantages.max().item(),
        "advantages/min": all_advantages.min().item(),
    }

    # Add method-specific metrics
    if method == AdvantageMethod.REINFORCE_PP and trajectory_groups_P is not None:
        # Log the per-group baseline (mean reward) used for subtraction
        baselines = []
        for traj_group in trajectory_groups_P:
            rewards_G = torch.tensor(traj_group.get_total_rewards(), dtype=torch.float32)
            baselines.append(rewards_G.mean().item())
        stats["reinforce_pp/baseline_value"] = float(np.mean(baselines))

    # Logtree display
    with logtree.scope_header("Advantage Computation"):
        logtree.table_from_dict({
            "method": method.value,
            "num_groups": str(len(advantages_P)),
            "num_trajectories": str(len(all_advantages)),
            "advantage_mean": f"{stats['advantages/mean']:.4f}",
            "advantage_std": f"{stats['advantages/std']:.4f}",
            "advantage_max": f"{stats['advantages/max']:.4f}",
            "advantage_min": f"{stats['advantages/min']:.4f}",
        })

    return stats


def compute_advantages(
    trajectory_groups_P: list[TrajectoryGroup],
    *,
    method: AdvantageMethod = AdvantageMethod.GRPO,
    normalize: bool = True,
    eps: float = 1e-8,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Compute advantages using the specified method.

    This is the main entry point for advantage estimation. It dispatches to
    the appropriate method-specific function based on ``method``.

    Args:
        trajectory_groups_P: Groups of trajectories.
        method: Which advantage estimator to use. Defaults to GRPO for
            backward compatibility.
        normalize: For REINFORCE++, whether to normalize by std. Ignored
            for other methods.
        eps: Epsilon for numerical stability in normalization.

    Returns:
        Tuple of (per-group advantage tensors, advantage metrics dict).
    """
    # Warn if normalize is explicitly set but method doesn't use it
    if not normalize and method != AdvantageMethod.REINFORCE_PP:
        logger.warning(
            "advantage_normalize is set to False but advantage_method is '%s'. "
            "The normalize setting only affects REINFORCE++.",
            method.value,
        )

    if method == AdvantageMethod.GRPO:
        with trace.scope_span_sync("compute_grpo_advantages"):
            advantages_P = compute_grpo_advantages(trajectory_groups_P)
    elif method == AdvantageMethod.REINFORCE_PP:
        with trace.scope_span_sync("compute_reinforce_pp_advantages"):
            advantages_P = compute_reinforce_pp_advantages(
                trajectory_groups_P, normalize=normalize, eps=eps
            )
    else:
        raise ValueError(f"Unknown advantage method: {method}")

    stats = _log_advantage_stats(advantages_P, method, trajectory_groups_P)
    return advantages_P, stats
