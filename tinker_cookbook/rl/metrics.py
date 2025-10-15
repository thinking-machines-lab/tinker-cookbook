"""
Metrics and KL computation functions for RL training.

Contains functions for computing KL divergences, incorporating KL penalties,
and computing training metrics.
"""

import asyncio
from typing import Any, Dict, List, cast

import numpy as np
import tinker
import torch
from tinker_cookbook.utils.misc_utils import safezip
from tinker_cookbook.utils.trace import scope


def compute_kl_sample_train(
    data_D: List[tinker.Datum], training_logprobs_D: List[torch.Tensor]
) -> Dict[str, float]:
    """Compute KL divergence metrics between sampling and training logprobs."""
    all_diffs: list[torch.Tensor] = []
    all_sampling_logprobs: list[torch.Tensor] = []

    for datum, training_logprobs in safezip(data_D, training_logprobs_D):
        # Get logprobs from sampling
        sampling_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        action_mask = datum.loss_fn_inputs["mask"].to_torch() > 0
        # Extract only action token logprobs
        sampling_logprobs_actions = sampling_logprobs[action_mask]
        training_logprobs_actions = training_logprobs[action_mask]

        if len(sampling_logprobs_actions) > 0:
            logprob_diff = sampling_logprobs_actions - training_logprobs_actions
            all_diffs.append(logprob_diff)
            all_sampling_logprobs.append(sampling_logprobs_actions)

    assert all_diffs
    flat_diffs = torch.cat(all_diffs)
    kl_sample_train_v1 = flat_diffs.mean().item()
    kl_sample_train_v2 = 0.5 * (flat_diffs**2).mean().item()

    flat_sampling_logprobs = torch.cat(all_sampling_logprobs)
    entropy_sample = -flat_sampling_logprobs.mean().item()
    return {
        "optim/kl_sample_train_v1": kl_sample_train_v1,
        "optim/kl_sample_train_v2": kl_sample_train_v2,
        "optim/entropy": entropy_sample,
    }


@scope
async def compute_post_kl(
    data_D: List[tinker.Datum], post_sampling_client: tinker.SamplingClient
) -> Dict[str, float]:
    """Compute post-update KL divergence metrics."""
    # Compute logprobs at all data items
    # This is a bit ugly, but we first reconstruct the original sequence from before we did the
    # shifting to get the inputs and targets.
    full_sequence_inputs_D = [
        datum.model_input.append_int(cast(int, datum.loss_fn_inputs["target_tokens"].data[-1]))
        for datum in data_D
    ]
    new_logprobs_D = await asyncio.gather(
        *[
            post_sampling_client.compute_logprobs_async(sequence_input)
            for sequence_input in full_sequence_inputs_D
        ]
    )

    prev_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    action_masks = [datum.loss_fn_inputs["mask"].to_torch() > 0 for datum in data_D]
    flat_diffs = [
        (prev_logprobs - torch.tensor(new_logprobs[1:]))[action_mask]
        for new_logprobs, prev_logprobs, action_mask in safezip(
            new_logprobs_D, prev_logprobs_D, action_masks
        )
    ]
    flat_diffs = torch.cat(flat_diffs)
    kl_post_v1 = flat_diffs.mean().item()
    kl_post_v2 = 0.5 * (flat_diffs**2).mean().item()

    return {"kl_pre_post_v1": kl_post_v1, "kl_pre_post_v2": kl_post_v2}


@scope
async def incorporate_kl_penalty(
    data_D: List[tinker.Datum],
    base_sampling_client: tinker.SamplingClient,
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> Dict[str, float]:
    """
    Compute KL against base model. Adjust advantages in-place by logp_base - logp_current - avg_kl,
    where avg_kl is the average of logp_base - logp_current (which is -KL[current, base])
    """
    # Compute logprobs at all data items
    full_sequence_inputs_D = [
        datum.model_input.append_int(cast(int, datum.loss_fn_inputs["target_tokens"].data[-1]))
        for datum in data_D
    ]
    base_logprobs_D = await asyncio.gather(
        *[
            base_sampling_client.compute_logprobs_async(sequence_input)
            for sequence_input in full_sequence_inputs_D
        ]
    )
    # compute the logprob differences, zeroed out when the mask == 0
    sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    float_masks = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]
    logprob_diffs = [
        (sampled_logprobs - torch.tensor(base_logprobs[1:])) * mask
        for base_logprobs, sampled_logprobs, mask in safezip(
            base_logprobs_D, sampled_logprobs_D, float_masks
        )
    ]
    avg_logp_diff = sum([diff.sum() for diff in logprob_diffs]) / sum(
        [mask.sum() for mask in float_masks]
    )
    for i, datum in enumerate(data_D):
        kl_advantages = kl_penalty_coef * float_masks[i] * (avg_logp_diff - logprob_diffs[i])
        if kl_discount_factor > 0:
            kl_advantages = torch.tensor(
                discounted_future_sum_vectorized(kl_advantages.numpy(), kl_discount_factor)
            )
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
        )

    return {"kl_policy_base": float(avg_logp_diff)}


def discounted_future_sum_vectorized(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute discounted sum of future values for each position using a vectorized approach.

    Args:
        x (np.ndarray): 1D array of rewards.
        gamma (float): Discount factor.

    Returns:
        np.ndarray: discounted sum of future values.
    """
    # Reverse x so lfilter processes from end to start
    import scipy.signal

    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1].astype(x.dtype)  # type: ignore


def compute_sampling_client_metrics(
    wrapped_trajectory_groups: List[Any],  # WrappedTrajectoryGroup
) -> dict[str, Any]:
    """Compute metrics about sampling clients used to generate trajectory groups."""
    sampling_client_steps = [
        wrapped_trajectory_group.sampling_client_step
        for wrapped_trajectory_group in wrapped_trajectory_groups
    ]
    sample_times = [
        wrapped_trajectory_group.metrics["time/trajectory_group_worker_loop/total"]
        for wrapped_trajectory_group in wrapped_trajectory_groups
    ]
    metrics = {}
    metrics["sampling_client/step_max"] = max(sampling_client_steps)
    metrics["sampling_client/step_min"] = min(sampling_client_steps)
    metrics["sampling_client/step_mean"] = sum(sampling_client_steps) / len(sampling_client_steps)
    metrics["time/sampling_time_max"] = max(sample_times)
    metrics["time/sampling_time_min"] = min(sample_times)
    metrics["time/sampling_time_mean"] = sum(sample_times) / len(sample_times)
    return metrics
