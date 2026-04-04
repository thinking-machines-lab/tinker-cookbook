"""Advantage estimation functions for RL training.

Provides pluggable advantage estimators used by the RL training pipeline.
Each estimator takes trajectory groups and returns per-trajectory advantage
values that are used to weight the policy gradient.

Supported estimators:
- **GRPO**: Group-relative advantages (mean-centered within each group).
- **REINFORCE++**: Baseline-subtracted REINFORCE with optional std normalization.
- **GAE**: Generalized Advantage Estimation from TD residuals (requires value predictions).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Sequence

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
    GAE = "gae"


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
            std = rewards_G.std()
            advantages_G = advantages_G / (std + eps)
        advantages_P.append(advantages_G)
    return advantages_P


def compute_gae_advantages(
    trajectory_groups_P: list[TrajectoryGroup],
    value_predictions_P: list[list[list[float]]],
    *,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> list[torch.Tensor]:
    """Compute GAE (Generalized Advantage Estimation) advantages from value predictions.

    GAE uses TD residuals and an exponentially-weighted sum to trade off
    bias and variance in advantage estimation. This is the standard advantage
    estimator used by PPO.

    For each trajectory with T transitions:
      - delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
      - A_t = sum_{l=0}^{T-t-1} (gamma * lam)^l * delta_{t+l}

    The per-trajectory advantage is the sum of per-timestep advantages
    weighted by the number of action tokens at each step (so the scalar
    assigned to each token reflects the multi-step GAE estimate).

    Since many RL setups in tinker-cookbook are single-turn (one transition
    per trajectory), this reduces to ``A = r + gamma * V(s') - V(s)`` in
    the common case.

    Args:
        trajectory_groups_P: Groups of trajectories.
        value_predictions_P: Per-group, per-trajectory, per-timestep value
            predictions. Shape: ``[n_groups][n_trajs][n_transitions + 1]``
            where the last element is the value of the terminal state
            (should be 0 for episodic tasks).
        gamma: Discount factor. Defaults to 1.0.
        lam: GAE lambda for bias-variance trade-off. Higher values (closer
            to 1) give lower bias but higher variance. Defaults to 0.95.

    Returns:
        Per-group advantage tensors of shape ``(G,)``, where each value is the
        sum of GAE advantages across timesteps for that trajectory.
    """
    advantages_P: list[torch.Tensor] = []

    for traj_group, value_preds_G in zip(trajectory_groups_P, value_predictions_P, strict=True):
        group_advantages: list[float] = []

        for traj, values in zip(traj_group.trajectories_G, value_preds_G, strict=True):
            n_steps = len(traj.transitions)
            assert len(values) == n_steps + 1, (
                f"Expected {n_steps + 1} value predictions (one per state including terminal), "
                f"got {len(values)}"
            )

            # Compute TD residuals: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            rewards = np.array([t.reward for t in traj.transitions], dtype=np.float64)
            vals = np.array(values, dtype=np.float64)
            deltas = rewards + gamma * vals[1:] - vals[:-1]

            # GAE: A_t = sum_{l=0}^{T-t-1} (gamma*lam)^l * delta_{t+l}
            # Computed backwards for efficiency
            gae_advantages = np.zeros(n_steps, dtype=np.float64)
            running_gae = 0.0
            for t in reversed(range(n_steps)):
                running_gae = deltas[t] + gamma * lam * running_gae
                gae_advantages[t] = running_gae

            # Sum per-timestep GAE advantages weighted by action token counts
            total_advantage = 0.0
            total_tokens = 0
            for t, transition in enumerate(traj.transitions):
                n_tokens = len(transition.ac.tokens)
                total_advantage += gae_advantages[t] * n_tokens
                total_tokens += n_tokens

            # Average advantage per token (so the scalar is independent of trajectory length)
            if total_tokens > 0:
                group_advantages.append(total_advantage / total_tokens)
            else:
                group_advantages.append(0.0)

        advantages_P.append(torch.tensor(group_advantages, dtype=torch.float32))

    return advantages_P


def _log_advantage_stats(
    advantages_P: list[torch.Tensor],
    method: AdvantageMethod,
) -> dict[str, float]:
    """Compute and log advantage statistics for telemetry.

    Args:
        advantages_P: Per-group advantage tensors.
        method: The advantage method used (for logtree display).

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
    if method == AdvantageMethod.REINFORCE_PP:
        # Log the per-group baseline (mean reward) used for subtraction
        baselines = []
        for adv_G in advantages_P:
            # The baseline is implicit in the centering; log the group mean
            baselines.append(adv_G.mean().item())
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
    value_predictions_P: list[list[list[float]]] | None = None,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
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
        value_predictions_P: Required for GAE. Per-group, per-trajectory,
            per-timestep value predictions.
        gamma: Discount factor for GAE. Defaults to 1.0.
        gae_lambda: Lambda for GAE bias-variance trade-off. Defaults to 0.95.

    Returns:
        Tuple of (per-group advantage tensors, advantage metrics dict).

    Raises:
        ValueError: If ``method`` is GAE but ``value_predictions_P`` is None.
    """
    if method == AdvantageMethod.GRPO:
        with trace.scope_span_sync("compute_grpo_advantages"):
            advantages_P = compute_grpo_advantages(trajectory_groups_P)
    elif method == AdvantageMethod.REINFORCE_PP:
        with trace.scope_span_sync("compute_reinforce_pp_advantages"):
            advantages_P = compute_reinforce_pp_advantages(
                trajectory_groups_P, normalize=normalize, eps=eps
            )
    elif method == AdvantageMethod.GAE:
        if value_predictions_P is None:
            raise ValueError(
                "GAE advantage estimation requires value_predictions_P. "
                "Provide per-group, per-trajectory, per-timestep value predictions."
            )
        with trace.scope_span_sync("compute_gae_advantages"):
            advantages_P = compute_gae_advantages(
                trajectory_groups_P,
                value_predictions_P,
                gamma=gamma,
                lam=gae_lambda,
            )
    else:
        raise ValueError(f"Unknown advantage method: {method}")

    stats = _log_advantage_stats(advantages_P, method)
    return advantages_P, stats
