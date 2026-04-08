"""
FIPO: Future-KL Influenced Policy Optimization

Implements token-level advantage reweighting via future-KL divergence,
as described in "FIPO: Eliciting Deep Reasoning with Future-KL Influenced
Policy Optimization" (arXiv:2603.19835).

The core idea: instead of applying a trajectory-level advantage uniformly
to all tokens, compute per-token influence weights based on discounted
future KL divergence between the current and old policy. Tokens that lead
to large downstream policy shifts get higher/lower influence weights.

Usage: after assemble_training_data() produces data_D, call
apply_fipo_reweighting() to modify advantages in-place, then use
the standard `ppo` loss function for training.
"""

import asyncio
import logging
from typing import cast

import tinker
import torch

from tinker_cookbook.utils.misc_utils import safezip

logger = logging.getLogger(__name__)


def compute_future_kl(
    current_logprobs: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    mask: torch.Tensor,
    decay_half_life: float = 32.0,
    dual_clip_threshold: float = 10.0,
) -> torch.Tensor:
    """Compute discounted future-KL for each token position.

    For each position t, computes:
        FutureKL_t = Σ_{k≥t} M_k · γ^(k-t) · Δlog p_k

    where Δlog p_k = log π_current(o_k) - log π_old(o_k), γ = 2^(-1/τ),
    and M_k is a participation mask filtering excessively large IS ratios.

    Args:
        current_logprobs: Per-token log-probs from current policy. Shape (T,).
        sampling_logprobs: Per-token log-probs from sampling (old) policy. Shape (T,).
        mask: Action mask (1 for action tokens, 0 for observation). Shape (T,).
        decay_half_life: τ in the decay formula γ = 2^(-1/τ). Default 32.
        dual_clip_threshold: c for participation mask. Default 10.0.

    Returns:
        Future-KL values for each position. Shape (T,).
    """
    T = len(current_logprobs)
    if T == 0:
        return torch.zeros(0, dtype=current_logprobs.dtype)

    gamma = 2.0 ** (-1.0 / decay_half_life)
    dtype = current_logprobs.dtype
    device = current_logprobs.device

    # Δlog p = log π_current - log π_old (signed shift)
    delta_logp = (current_logprobs - sampling_logprobs) * mask.to(dtype)

    # Participation mask: exclude tokens where |Δlog p| exceeds threshold
    log_threshold = torch.log(torch.tensor(dual_clip_threshold, device=device, dtype=dtype))
    participation = (delta_logp.abs() <= log_threshold).to(dtype)
    kl_values = delta_logp * participation

    # Discounted future sum via backward pass (O(T))
    future_kl = torch.empty(T, dtype=dtype, device=device)
    running = torch.tensor(0.0, dtype=dtype, device=device)
    for t in range(T - 1, -1, -1):
        running = kl_values[t] + gamma * running
        future_kl[t] = running

    return future_kl


def compute_influence_weights(
    future_kl: torch.Tensor,
    advantages: torch.Tensor,
    current_logprobs: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    clip_low: float = 1.0,
    clip_high: float = 1.2,
    safety_threshold: float = 4.0,
) -> torch.Tensor:
    """Compute influence weights from future-KL values.

    f_t = clip(exp(FutureKL_t), 1 - ε_low, 1 + ε_high)

    Args:
        future_kl: Per-token future-KL values. Shape (T,).
        advantages: Per-token advantages. Shape (T,).
        current_logprobs: Current policy log-probs. Shape (T,).
        sampling_logprobs: Old policy log-probs. Shape (T,).
        clip_low: Lower clipping bound. Default 1.0.
        clip_high: Upper clipping bound. Default 1.2.
        safety_threshold: IS ratio threshold for safety clamping. Default 4.0.

    Returns:
        Per-token influence weights. Shape (T,).
    """
    influence = torch.clamp(
        torch.exp(future_kl),
        min=1.0 - clip_low,
        max=1.0 + clip_high,
    )

    # Safety: for negative advantage + high IS ratio, clamp influence weights
    ratio = torch.exp(current_logprobs - sampling_logprobs)
    mask_neg_high_is = (advantages < 0) & (ratio > safety_threshold)
    influence = torch.where(
        mask_neg_high_is,
        torch.clamp(influence, min=0.8, max=1.0),
        influence,
    )

    return influence


async def apply_fipo_reweighting(
    data_D: list[tinker.Datum],
    sampling_client: tinker.SamplingClient,
    decay_half_life: float = 32.0,
    dual_clip_threshold: float = 10.0,
    influence_clip_low: float = 1.0,
    influence_clip_high: float = 1.2,
    safety_threshold: float = 4.0,
) -> dict[str, float]:
    """Apply FIPO advantage reweighting to training data in-place.

    Computes current policy logprobs via the sampling client, calculates
    future-KL influence weights, and multiplies advantages by these weights.

    This should be called after assemble_training_data() and before
    the training step (forward_backward with ppo loss).

    Args:
        data_D: Training datums with "logprobs", "advantages", "mask",
            and "target_tokens" in loss_fn_inputs. Modified in-place.
        sampling_client: Sampling client with current training weights.
        decay_half_life: τ for future-KL decay. Default 32.
        dual_clip_threshold: c for participation mask. Default 10.0.
        influence_clip_low: Lower bound for influence clipping. Default 1.0.
        influence_clip_high: Upper bound for influence clipping. Default 1.2.
        safety_threshold: Safety threshold for negative samples. Default 4.0.

    Returns:
        Dict of FIPO metrics (influence weight stats, future-KL stats).
    """
    # Reconstruct full sequences and compute current logprobs
    full_sequence_inputs_D = [
        datum.model_input.append_int(cast(int, datum.loss_fn_inputs["target_tokens"].data[-1]))
        for datum in data_D
    ]
    current_logprobs_D = await asyncio.gather(
        *[
            sampling_client.compute_logprobs_async(seq)
            for seq in full_sequence_inputs_D
        ]
    )

    all_influence_weights: list[torch.Tensor] = []
    all_future_kl_abs: list[torch.Tensor] = []

    for datum, current_logprobs_raw in safezip(data_D, current_logprobs_D):
        sampling_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        advantages = datum.loss_fn_inputs["advantages"].to_torch()
        mask = datum.loss_fn_inputs["mask"].to_torch()

        # current_logprobs_raw includes logprobs for all positions including BOS
        # Skip first element to align with the shifted datum format
        current_logprobs = torch.tensor(current_logprobs_raw[1:], dtype=sampling_logprobs.dtype)

        action_mask = mask > 0
        if action_mask.sum() == 0:
            continue

        # Compute future-KL
        future_kl = compute_future_kl(
            current_logprobs=current_logprobs,
            sampling_logprobs=sampling_logprobs,
            mask=mask,
            decay_half_life=decay_half_life,
            dual_clip_threshold=dual_clip_threshold,
        )

        # Compute influence weights
        influence = compute_influence_weights(
            future_kl=future_kl,
            advantages=advantages,
            current_logprobs=current_logprobs,
            sampling_logprobs=sampling_logprobs,
            clip_low=influence_clip_low,
            clip_high=influence_clip_high,
            safety_threshold=safety_threshold,
        )

        # Reweight advantages in-place
        weighted_advantages = advantages * influence
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(weighted_advantages)

        all_influence_weights.append(influence[action_mask])
        all_future_kl_abs.append(future_kl[action_mask].abs())

    if not all_influence_weights:
        return {"fipo/num_datums": 0}

    cat_influence = torch.cat(all_influence_weights)
    cat_future_kl = torch.cat(all_future_kl_abs)

    return {
        "fipo/num_datums": len(data_D),
        "fipo/influence_weight_mean": cat_influence.mean().item(),
        "fipo/influence_weight_std": cat_influence.std().item(),
        "fipo/influence_weight_min": cat_influence.min().item(),
        "fipo/influence_weight_max": cat_influence.max().item(),
        "fipo/future_kl_abs_mean": cat_future_kl.mean().item(),
    }
