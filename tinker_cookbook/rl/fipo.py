"""
FIPO: Future-KL Influenced Policy Optimization

Implements token-level advantage reweighting via future-KL divergence,
as described in "FIPO: Eliciting Deep Reasoning with Future-KL Influenced
Policy Optimization" (arXiv:2603.19835).

The core idea: instead of applying a trajectory-level advantage uniformly
to all tokens, compute per-token influence weights based on discounted
future KL divergence between the current and old policy. Tokens that lead
to large downstream policy shifts get higher/lower influence weights.
"""

import logging

import tinker
import torch

logger = logging.getLogger(__name__)


def compute_future_kl(
    training_logprobs: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    mask: torch.Tensor,
    decay_half_life: float = 32.0,
    chunk_size: int = 128,
    dual_clip_threshold: float = 10.0,
) -> torch.Tensor:
    """Compute discounted future-KL for each token position.

    For each position t, computes:
        FutureKL_t = Σ_{k≥t} M_k · γ^(k-t) · Δlog p_k

    where Δlog p_k = log π_θ(o_k) - log π_old(o_k), γ = 2^(-1/τ), and M_k
    is a participation mask that filters out tokens with excessively large
    importance ratios (dual-clip filter).

    Args:
        training_logprobs: Per-token log-probs from current policy. Shape (T,).
        sampling_logprobs: Per-token log-probs from old policy. Shape (T,).
        mask: Action mask (1 for action tokens, 0 for observation). Shape (T,).
        decay_half_life: τ in the decay formula γ = 2^(-1/τ). Controls how far
            into the future the KL influence extends. Default 32.
        chunk_size: Block size for chunked matrix computation. Default 128.
        dual_clip_threshold: c threshold for the participation mask. Tokens where
            the IS ratio exceeds c are excluded from future-KL. Default 10.0.

    Returns:
        Future-KL values for each position. Shape (T,).
    """
    T = len(training_logprobs)
    if T == 0:
        return torch.zeros(0, dtype=training_logprobs.dtype)

    gamma = 2.0 ** (-1.0 / decay_half_life)
    device = training_logprobs.device
    dtype = training_logprobs.dtype

    # Δlog p = log π_θ - log π_old (signed shift)
    delta_logp = (training_logprobs - sampling_logprobs) * mask.to(dtype)

    # Participation mask: exclude tokens where IS ratio exceeds dual-clip threshold
    log_threshold = torch.log(torch.tensor(dual_clip_threshold, device=device, dtype=dtype))
    participation = (delta_logp.abs() <= log_threshold).to(dtype)
    kl_values = delta_logp * participation

    # Compute discounted future sum efficiently using backward pass (O(T))
    future_kl = torch.empty(T, dtype=dtype, device=device)
    running = torch.tensor(0.0, dtype=dtype, device=device)
    for t in range(T - 1, -1, -1):
        running = kl_values[t] + gamma * running
        future_kl[t] = running

    return future_kl


def compute_fipo_influence_weights(
    future_kl: torch.Tensor,
    advantages: torch.Tensor,
    training_logprobs: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    clip_low: float = 1.0,
    clip_high: float = 1.2,
    safety_threshold: float = 4.0,
) -> torch.Tensor:
    """Compute influence weights from future-KL values.

    f_t = clip(exp(FutureKL_t), 1 - ε_low, 1 + ε_high)

    With additional safety: for negative-advantage tokens where the importance
    sampling ratio is very high (>safety_threshold), the influence weights are
    further clamped to [0.8, 1.0] to prevent instability.

    Args:
        future_kl: Per-token future-KL values. Shape (T,).
        advantages: Per-token advantages (trajectory-level broadcast). Shape (T,).
        training_logprobs: Current policy log-probs. Shape (T,).
        sampling_logprobs: Old policy log-probs. Shape (T,).
        clip_low: Lower clipping bound (ε_low). Default 1.0.
        clip_high: Upper clipping bound (ε_high). Default 1.2.
        safety_threshold: IS ratio threshold for clamping negative samples. Default 4.0.

    Returns:
        Per-token influence weights. Shape (T,).
    """
    lower_bound = 1.0 - clip_low
    upper_bound = 1.0 + clip_high

    influence_weights = torch.clamp(torch.exp(future_kl), min=lower_bound, max=upper_bound)

    # Safety: for negative advantage + high IS ratio, clamp influence weights
    ratio = torch.exp(training_logprobs - sampling_logprobs)
    mask_neg_high_is = (advantages < 0) & (ratio > safety_threshold)
    influence_weights = torch.where(
        mask_neg_high_is,
        torch.clamp(influence_weights, min=0.8, max=1.0),
        influence_weights,
    )

    return influence_weights.detach()


def fipo_loss_fn(
    data: list[tinker.Datum],
    logprobs_list: list[torch.Tensor],
    clip_epsilon: float = 0.2,
    decay_half_life: float = 32.0,
    dual_clip_threshold: float = 10.0,
    influence_clip_low: float = 1.0,
    influence_clip_high: float = 1.2,
    safety_threshold: float = 4.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Custom loss function for FIPO, compatible with forward_backward_custom.

    Computes a PPO-style clipped loss where advantages are reweighted by
    future-KL influence weights.

    Args:
        data: List of training datums. Each datum must have loss_fn_inputs
            containing "logprobs" (sampling log-probs), "advantages", and "mask".
        logprobs_list: Per-datum training log-prob tensors from the forward pass.
        clip_epsilon: PPO clip range ε. Default 0.2.
        decay_half_life: τ for future-KL decay. Default 32.
        dual_clip_threshold: c for participation mask. Default 10.0.
        influence_clip_low: ε_low for influence weight clipping. Default 1.0.
        influence_clip_high: ε_high for influence weight clipping. Default 1.2.
        safety_threshold: IS ratio threshold for safety clamping. Default 4.0.

    Returns:
        (loss, metrics) tuple compatible with forward_backward_custom.
    """
    total_loss = torch.tensor(0.0)
    total_tokens = 0

    # Metrics accumulators
    all_influence_weights = []
    all_future_kl_abs = []
    all_ratios = []
    all_clipped_frac = []

    for datum, training_logprobs in zip(data, logprobs_list):
        sampling_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        advantages = datum.loss_fn_inputs["advantages"].to_torch()
        mask = datum.loss_fn_inputs["mask"].to_torch()

        # Number of action tokens
        action_mask = mask > 0
        n_action_tokens = action_mask.sum().item()
        if n_action_tokens == 0:
            continue

        # Compute future-KL and influence weights
        future_kl = compute_future_kl(
            training_logprobs=training_logprobs,
            sampling_logprobs=sampling_logprobs,
            mask=mask,
            decay_half_life=decay_half_life,
            dual_clip_threshold=dual_clip_threshold,
        )

        influence_weights = compute_fipo_influence_weights(
            future_kl=future_kl,
            advantages=advantages,
            training_logprobs=training_logprobs,
            sampling_logprobs=sampling_logprobs,
            clip_low=influence_clip_low,
            clip_high=influence_clip_high,
            safety_threshold=safety_threshold,
        )

        # Reweight advantages
        weighted_advantages = advantages * influence_weights

        # PPO-style clipped loss
        log_ratio = training_logprobs - sampling_logprobs
        ratio = torch.exp(log_ratio)

        pg_loss1 = -weighted_advantages * ratio
        pg_loss2 = -weighted_advantages * torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        pg_loss = torch.maximum(pg_loss1, pg_loss2)

        # Apply mask and sum
        masked_loss = (pg_loss * mask).sum()
        total_loss = total_loss + masked_loss
        total_tokens += n_action_tokens

        # Collect metrics
        with torch.no_grad():
            action_influence = influence_weights[action_mask]
            all_influence_weights.append(action_influence)
            all_future_kl_abs.append(future_kl[action_mask].abs())
            all_ratios.append(ratio[action_mask])
            clipped = ((ratio[action_mask] < 1 - clip_epsilon) | (ratio[action_mask] > 1 + clip_epsilon)).float()
            all_clipped_frac.append(clipped)

    if total_tokens == 0:
        return torch.tensor(0.0, requires_grad=True), {"fipo/num_tokens": 0}

    loss = total_loss / total_tokens

    # Aggregate metrics
    with torch.no_grad():
        cat_influence = torch.cat(all_influence_weights)
        cat_future_kl = torch.cat(all_future_kl_abs)
        cat_ratios = torch.cat(all_ratios)
        cat_clipped = torch.cat(all_clipped_frac)

    metrics = {
        "fipo/loss": loss.item(),
        "fipo/num_tokens": total_tokens,
        "fipo/influence_weight_mean": cat_influence.mean().item(),
        "fipo/influence_weight_std": cat_influence.std().item(),
        "fipo/future_kl_abs_mean": cat_future_kl.mean().item(),
        "fipo/is_ratio_mean": cat_ratios.mean().item(),
        "fipo/clipped_frac": cat_clipped.mean().item(),
    }

    return loss, metrics
