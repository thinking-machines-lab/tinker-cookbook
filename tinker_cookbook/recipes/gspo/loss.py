"""GSPO: Group Sequence Policy Optimization loss function.

Paper: https://arxiv.org/abs/2507.18071

Key difference from GRPO/PPO: importance ratios are computed at the sequence
level (geometric mean of token ratios) rather than per-token. This avoids
variance explosion from multiplying per-token ratios across long sequences,
and is theoretically correct since the policy's "action" is the full sequence.
"""

from collections.abc import Callable

import tinker.types as types
import torch

# Empirical clip bounds from the paper, section 4.2.
# Sequence-level ratios stay near 1.0 by construction (geometric mean),
# so these are much tighter than GRPO's 0.8-1.27 token-level bounds.
DEFAULT_CLIP_LOW: float = 1.0 - 3e-4
DEFAULT_CLIP_HIGH: float = 1.0 + 4e-4


def make_gspo_loss(
    old_logprobs_D: list[torch.Tensor],
    ob_lens_D: list[int],
    advantages_D: list[float],
    clip_low: float = DEFAULT_CLIP_LOW,
    clip_high: float = DEFAULT_CLIP_HIGH,
) -> Callable[[list[types.Datum], list[torch.Tensor]], tuple[torch.Tensor, dict[str, float]]]:
    """Return a GSPO loss function pre-loaded with per-datum reference data.

    forward_backward_custom only allows 'target_tokens' and 'weights' in
    loss_fn_inputs, so old logprobs and advantages are captured in a closure.

    Args:
        old_logprobs_D: Full padded logprob sequences from the sampling policy,
            one tensor per datum. Prompt positions are 0.0; completion positions
            hold the actual old-policy logprobs.
        ob_lens_D: Number of prompt positions in each datum. Used to slice out
            completion tokens from both old and new logprob tensors.
        advantages_D: Scalar group-relative advantage per datum, normalized by
            within-group std (per the GSPO paper).
        clip_low: Lower bound for sequence-level IS ratio clipping.
        clip_high: Upper bound for sequence-level IS ratio clipping.
    """

    def gspo_loss(
        data: list[types.Datum],
        new_logprobs_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del data
        if not new_logprobs_list:
            return torch.tensor(0.0), {
                "gspo_loss": 0.0,
                "clip_frac": 0.0,
                "mean_log_ratio": 0.0,
            }

        mean_log_ratios = []
        for old_lp, ob_len, new_lp in zip(
            old_logprobs_D, ob_lens_D, new_logprobs_list, strict=True
        ):
            # Slice to completion tokens only (prompt positions have old_lp=0.0).
            old_completion = old_lp.to(device=new_lp.device, dtype=new_lp.dtype)[ob_len:]
            new_completion = new_lp[ob_len:]

            # Sequence-level IS ratio: geometric mean of per-token ratios.
            # s_i = exp( (1/|y|) * sum_t [ log pi_theta(y_t) - log pi_old(y_t) ] )
            mean_log_ratios.append((new_completion - old_completion).mean())

        mean_log_ratio_tensor = torch.stack(mean_log_ratios)
        sequence_ratios = torch.exp(mean_log_ratio_tensor)
        advantages = torch.tensor(
            advantages_D,
            dtype=sequence_ratios.dtype,
            device=sequence_ratios.device,
        )

        unclipped = sequence_ratios * advantages
        clipped = torch.clamp(sequence_ratios, clip_low, clip_high) * advantages
        losses = -torch.minimum(unclipped, clipped)
        avg_loss = losses.mean()

        clipped_mask = (sequence_ratios < clip_low) | (sequence_ratios > clip_high)

        return avg_loss, {
            "gspo_loss": avg_loss.item(),
            "clip_frac": clipped_mask.float().mean().item(),
            "mean_log_ratio": mean_log_ratio_tensor.mean().item(),
        }

    return gspo_loss
