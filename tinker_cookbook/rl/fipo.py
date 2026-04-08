"""
FIPO: Future-KL Influenced Policy Optimization

Implements token-level advantage reweighting via future-KL divergence,
as described in "FIPO: Eliciting Deep Reasoning with Future-KL Influenced
Policy Optimization" (arXiv:2603.19835).

The core idea: instead of applying a trajectory-level advantage uniformly
to all tokens, compute per-token influence weights based on discounted
future KL divergence between the current and old policy. Tokens that lead
to large downstream policy shifts get higher/lower influence weights.

Uses forward_backward_custom with a closure that captures RL-specific fields
(sampling logprobs, advantages, mask) externally, since forward_backward_custom
only accepts target_tokens + weights in the datum format.
"""

import logging

import tinker
import torch

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


def make_fipo_loss_fn(
    rl_data_D: list[tinker.Datum],
    clip_epsilon: float = 0.2,
    decay_half_life: float = 32.0,
    dual_clip_threshold: float = 10.0,
    influence_clip_low: float = 1.0,
    influence_clip_high: float = 1.2,
    safety_threshold: float = 4.0,
):
    """Create a FIPO loss closure for forward_backward_custom.

    The closure captures the RL-specific fields (sampling logprobs, advantages,
    mask) from the original RL datums, since forward_backward_custom only
    supports target_tokens + weights in the datum format.

    The returned loss function receives training logprobs from the forward pass
    and computes the full FIPO loss: PPO-clipped policy gradient with
    future-KL-reweighted advantages.

    Args:
        rl_data_D: Original RL datums containing "logprobs", "advantages",
            and "mask" in loss_fn_inputs.
        clip_epsilon: PPO clip range. Default 0.2.
        decay_half_life: τ for future-KL decay. Default 32.
        dual_clip_threshold: c for participation mask. Default 10.0.
        influence_clip_low: ε_low for influence clipping. Default 1.0.
        influence_clip_high: ε_high for influence clipping. Default 1.2.
        safety_threshold: Safety threshold for negative samples. Default 4.0.

    Returns:
        Loss function compatible with forward_backward_custom.
    """
    # Capture RL fields from original datums
    sampling_logprobs_D = [d.loss_fn_inputs["logprobs"].to_torch() for d in rl_data_D]
    advantages_D = [d.loss_fn_inputs["advantages"].to_torch() for d in rl_data_D]
    masks_D = [d.loss_fn_inputs["mask"].to_torch() for d in rl_data_D]

    def loss_fn(
        data: list[tinker.Datum],
        logprobs_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total_loss = torch.tensor(0.0)
        total_tokens = 0
        all_influence: list[torch.Tensor] = []
        all_future_kl: list[torch.Tensor] = []

        for i, training_logprobs in enumerate(logprobs_list):
            sampling_logprobs = sampling_logprobs_D[i]
            advantages = advantages_D[i]
            mask = masks_D[i]

            action_mask = mask > 0
            n_action = int(action_mask.sum().item())
            if n_action == 0:
                continue

            # Future-KL computation
            future_kl = compute_future_kl(
                current_logprobs=training_logprobs,
                sampling_logprobs=sampling_logprobs,
                mask=mask,
                decay_half_life=decay_half_life,
                dual_clip_threshold=dual_clip_threshold,
            )

            # Influence weights
            influence = compute_influence_weights(
                future_kl=future_kl,
                advantages=advantages,
                current_logprobs=training_logprobs,
                sampling_logprobs=sampling_logprobs,
                clip_low=influence_clip_low,
                clip_high=influence_clip_high,
                safety_threshold=safety_threshold,
            )

            # FIPO loss: PPO-style clipped loss with reweighted advantages
            weighted_adv = advantages * influence.detach()
            log_ratio = training_logprobs - sampling_logprobs
            ratio = torch.exp(log_ratio)

            pg_loss1 = -weighted_adv * ratio
            pg_loss2 = -weighted_adv * torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            pg_loss = torch.maximum(pg_loss1, pg_loss2)

            total_loss = total_loss + (pg_loss * mask).sum()
            total_tokens += n_action

            with torch.no_grad():
                all_influence.append(influence[action_mask])
                all_future_kl.append(future_kl[action_mask].abs())

        if total_tokens == 0:
            return torch.tensor(0.0, requires_grad=True), {"fipo/num_tokens": 0}

        loss = total_loss / total_tokens

        with torch.no_grad():
            cat_infl = torch.cat(all_influence)
            cat_fkl = torch.cat(all_future_kl)

        metrics = {
            "fipo/loss": loss.item(),
            "fipo/num_tokens": total_tokens,
            "fipo/influence_weight_mean": cat_infl.mean().item(),
            "fipo/influence_weight_std": cat_infl.std().item(),
            "fipo/influence_weight_min": cat_infl.min().item(),
            "fipo/influence_weight_max": cat_infl.max().item(),
            "fipo/future_kl_abs_mean": cat_fkl.mean().item(),
        }
        return loss, metrics

    return loss_fn


def prepare_custom_datums(rl_data_D: list[tinker.Datum]) -> list[tinker.Datum]:
    """Convert RL datums to the format expected by forward_backward_custom.

    forward_backward_custom only supports target_tokens + weights.
    We strip out logprobs, advantages, and mask, using mask as weights.

    Args:
        rl_data_D: Original RL datums with full loss_fn_inputs.

    Returns:
        Stripped datums with only target_tokens and weights.
    """
    custom_datums = []
    for d in rl_data_D:
        custom_datums.append(
            tinker.Datum(
                model_input=d.model_input,
                loss_fn_inputs={
                    "target_tokens": d.loss_fn_inputs["target_tokens"],
                    "weights": d.loss_fn_inputs["mask"],
                },
            )
        )
    return custom_datums
