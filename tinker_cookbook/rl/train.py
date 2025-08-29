"""
Implements RL on general MDPs

"""

import asyncio
import logging
import time
from typing import Dict, List, cast

import chz
import numpy as np
import tinker
import torch
from tinker import types
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.display import colorize_example
from tinker_cookbook.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import (
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import all_same, safezip, split_list, timed

logger = logging.getLogger(__name__)


def compute_advantages(trajectory_groups_P: List[TrajectoryGroup]) -> List[torch.Tensor]:
    """Compute advantages for each trajectory, centered within groups."""
    advantages_P = []

    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards())
        # Center advantages within the group
        advantages_G = rewards_G - rewards_G.mean()
        advantages_P.append(advantages_G)

    return advantages_P


FlatObElem = int | types.ModelInputChunk
FlatOb = list[FlatObElem]


def _is_prefix(seq1: FlatOb, seq2: FlatOb) -> bool:
    """
    Check if seq1 is a prefix of seq2.
    """
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def to_data(traj: Trajectory, traj_advantage: float) -> list[types.Datum]:
    """
    Return one or more Datum objects corresponding to the trajectory.
    If the sequence grows by appending, i.e., each successive observation contains
    the previous observation+action as a prefix, then we can return a single Datum.
    However, if we get a sequence that's not an extension of the previous sequence,
    then that results in a new Datum.

    For example, let O1 denote a chunk of observation tokens, and let A1 denote an action.

    Then let's say ob_ac_pairs is as follows.

    (O1, A1)
    (O1+A1+O2, A2)
    (O3, A3)

    Then we will merge the first two observation-action pairs into a single Datum,
    and the last observation-action pair into a separate Datum.
    """

    class SequenceAccumulator:
        full_sequence = []
        sampled_logprobs = []
        advantages = []
        mask = []

        @classmethod
        def clear(cls):
            cls.full_sequence = []
            cls.sampled_logprobs = []
            cls.advantages = []
            cls.mask = []

    def make_datum_from_state():
        # TODO: generalize to multimodal
        all_tokens_T = _flat_ob_to_model_input(SequenceAccumulator.full_sequence)
        input_tokens_T, target_tokens_T = _to_input_targets(all_tokens_T)
        sampled_logprobs_T = SequenceAccumulator.sampled_logprobs[1:]
        advantages_T = SequenceAccumulator.advantages[1:]
        mask_T = SequenceAccumulator.mask[1:]
        assert (
            input_tokens_T.length
            == len(target_tokens_T)
            == len(sampled_logprobs_T)
            == len(advantages_T)
            == len(mask_T)
        )
        return types.Datum(
            model_input=input_tokens_T,
            loss_fn_inputs={  # type: ignore
                "target_tokens": torch.tensor(target_tokens_T),
                "logprobs": torch.tensor(sampled_logprobs_T),
                "advantages": torch.tensor(advantages_T),
                "mask": torch.tensor(mask_T),
            },
        )

    data = []
    for transition in traj.transitions:
        ob = transition.ob
        ob_flat = _flatten_chunks(ob.chunks)
        ac_with_logprobs = transition.ac
        if len(SequenceAccumulator.full_sequence) == 0:
            delta_ob_flat = ob_flat
        elif _is_prefix(SequenceAccumulator.full_sequence, ob_flat):
            delta_ob_flat = ob_flat[len(SequenceAccumulator.full_sequence) :]
        else:
            data.append(make_datum_from_state())
            SequenceAccumulator.clear()
            delta_ob_flat = ob_flat
        delta_ob_len = _flat_ob_token_len(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(ac_with_logprobs.tokens)
        SequenceAccumulator.sampled_logprobs.extend(
            [0.0] * delta_ob_len + ac_with_logprobs.logprobs
        )
        SequenceAccumulator.advantages.extend(
            [0] * delta_ob_len + [traj_advantage] * len(ac_with_logprobs.tokens)
        )
        SequenceAccumulator.mask.extend([0.0] * delta_ob_len + [1.0] * len(ac_with_logprobs.tokens))

    if SequenceAccumulator.full_sequence:
        data.append(make_datum_from_state())

    return data


def _flat_ob_token_len(flat_ob: FlatOb) -> int:
    out = 0
    for elem in flat_ob:
        if isinstance(elem, int):
            out += 1
        else:
            out += elem.length
    return out


def _to_input_targets(model_input: types.ModelInput) -> tuple[types.ModelInput, list[int]]:
    # TODO: make this work with multimodal data
    all_ints = model_input.to_ints()
    return types.ModelInput.from_ints(tokens=all_ints[:-1]), all_ints[1:]


def _flat_ob_to_model_input(flat_ob: FlatOb) -> types.ModelInput:
    out = []
    current_text_chunk = []

    def flush_text_chunk():
        if current_text_chunk:
            out.append(types.EncodedTextChunk(tokens=current_text_chunk))
            current_text_chunk.clear()

    for elem in flat_ob:
        if isinstance(elem, int):
            current_text_chunk.append(elem)
        else:
            flush_text_chunk()
            out.append(elem)
    flush_text_chunk()
    return types.ModelInput(chunks=out)


def _flatten_chunks(chunks: list[types.ModelInputChunk]) -> FlatOb:
    out = []
    for chunk in chunks:
        if isinstance(chunk, types.EncodedTextChunk):
            out.extend(chunk.tokens)
        else:
            out.append(chunk)
    return out


def compute_kl_sample_train(
    data_D: List[types.Datum], training_logprobs_D: List[torch.Tensor]
) -> Dict[str, float]:
    """Compute KL divergence metrics between sampling and training logprobs."""
    all_diffs = []
    all_sampling_logprobs = []

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


async def compute_post_kl(
    data_D: List[types.Datum], post_sampling_client: tinker.SamplingClient
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


async def incorporate_kl_penalty(
    data_D: List[types.Datum],
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
        datum.loss_fn_inputs["advantages"] = types.TensorData.from_torch(
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


def assemble_training_data(
    trajectory_groups_P: List[TrajectoryGroup],
    advantages_P: List[torch.Tensor],
) -> tuple[List[types.Datum], List[dict]]:
    """Convert trajectories to training data format."""
    data_D = []
    metadata_D = []

    for i_group, (traj_group, advantages_G) in enumerate(
        safezip(trajectory_groups_P, advantages_P)
    ):
        for i_traj, (traj, traj_advantage) in enumerate(
            safezip(traj_group.trajectories_G, advantages_G)
        ):
            # Build the full sequence from the trajectory
            new_data = to_data(traj, traj_advantage)
            data_D.extend(new_data)
            metadata_D.extend([dict(group_idx=i_group, traj_idx=i_traj) for _ in new_data])

    return data_D, metadata_D


def _select_representative_inds(scores: list[float], num_inds: int) -> list[int]:
    assert num_inds <= len(scores)
    sorted_inds = np.argsort(scores)
    uniform_inds = np.linspace(0, len(sorted_inds) - 1, num_inds).astype(int)
    return [int(sorted_inds[i]) for i in uniform_inds]


def print_group(traj_group: TrajectoryGroup, tokenizer: Tokenizer):
    # Cut down the number of trajectories to print
    max_trajs_to_print = 4
    if len(traj_group.trajectories_G) > max_trajs_to_print:
        inds = _select_representative_inds(traj_group.get_total_rewards(), max_trajs_to_print)
        traj_group = TrajectoryGroup(
            trajectories_G=[traj_group.trajectories_G[i] for i in inds],
            final_rewards_G=[traj_group.final_rewards_G[i] for i in inds],
            metrics_G=[traj_group.metrics_G[i] for i in inds],
        )

    rewards = traj_group.get_total_rewards()
    advantages_G = compute_advantages([traj_group])
    data_D, metadata_D = assemble_training_data([traj_group], advantages_G)
    print("====== Trajectory Group ======")
    last_metadata = None
    for datum, metadata in safezip(data_D, metadata_D):
        idx = metadata["traj_idx"]
        if metadata != last_metadata:
            print(f"****** trajectory idx={idx}, reward={rewards[idx]:.3g} ******")
        print("---- datum ----")
        print(colorize_example(datum, tokenizer, key="advantages"))
        last_metadata = metadata
    print("====== End Trajectory Group ======")


def _remove_mask(datum: types.Datum) -> types.Datum:
    return types.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


def _remove_constant_reward_groups(
    trajectory_groups_P: List[TrajectoryGroup],
) -> List[TrajectoryGroup]:
    new_groups = []
    for group in trajectory_groups_P:
        if not all_same(group.get_total_rewards()):
            new_groups.append(group)
    if not new_groups:
        logger.warning("All rewards are uniform. There will be no gradient")
        return trajectory_groups_P[0:1]  # return singleton list in case empty
        # list will cause problems
    return new_groups


async def train_step(
    data_D: List[types.Datum],
    training_client: tinker.TrainingClient,
    learning_rate: float,
    num_minibatches: int,
) -> List[torch.Tensor]:
    """Train the model on collected trajectories."""
    batches_md = split_list(data_D, min(num_minibatches, len(data_D)))
    fwd_futures_m = []
    optim_step_futures_m = []

    for batch_d in batches_md:
        # Forward-backward pass
        fwd_bwd_future = await training_client.forward_backward_async(
            list(map(_remove_mask, batch_d)), loss_fn="importance_sampling"
        )

        # Optimizer step
        adam_params = types.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
        optim_step_future = await training_client.optim_step_async(adam_params)
        fwd_futures_m.append(fwd_bwd_future)
        optim_step_futures_m.append(optim_step_future)

    fwd_bwd_results_m = [await future.result_async() for future in fwd_futures_m]
    _optim_step_results_m = [await future.result_async() for future in optim_step_futures_m]

    # Extract training logprobs from loss_fn_outputs
    training_logprobs_D = []
    for fwd_bwd_result in fwd_bwd_results_m:
        for output in fwd_bwd_result.loss_fn_outputs:
            training_logprobs = output["logprobs"].to_torch()
            training_logprobs_D.append(training_logprobs)

    # We dont display fwd_bwd_result.metrics to avoid spam
    return training_logprobs_D


@chz.chz
class Config:
    learning_rate: float
    dataset_builder: RLDatasetBuilder  # also determines batch size
    model_name: str
    max_tokens: int
    compute_post_kl: bool = False
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    lora_rank: int = 32

    kl_penalty_coef: float = 0.0
    num_minibatches: int = 1
    kl_discount_factor: float = 0.0

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str
    base_url: str | None = None

    remove_constant_reward_groups: bool = False
    eval_every: int = 1
    save_every: int = 20
    load_checkpoint_path: str | None = None


async def do_update(
    i_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
):
    """Perform a single update step in the RL training loop."""
    metrics = {
        "progress/batch": i_batch,
        "optim/lr": cfg.learning_rate,
        "progress/done_frac": (i_batch + 1) / num_batches,
    }

    t_start = time.time()

    # Save checkpoint and create sampling client
    path_dict = await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name=f"{i_batch:06d}",
        log_path=cfg.log_path,
        loop_state={"batch": i_batch},
        kind="both" if (i_batch > 0 and i_batch % cfg.save_every == 0) else "sampler",
    )
    sampling_client = training_client.create_sampling_client(path_dict["sampler_path"])

    # Run evaluations
    if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0:
        with timed("run_evals", metrics):
            for evaluator in evaluators:
                eval_metrics = await evaluator(sampling_client)
                metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

    # Get batch and sample trajectories
    env_group_builders_P = dataset.get_batch(i_batch)
    with timed("sample", metrics):
        # Create sampling client with current weights
        policy = TinkerTokenCompleter(sampling_client, max_tokens=cfg.max_tokens)
        trajectory_groups_P = await asyncio.gather(
            *[do_group_rollout(builder, policy) for builder in env_group_builders_P]
        )

    # Compute trajectory metrics
    metrics.update(compute_trajectory_metrics(trajectory_groups_P))

    # Print one trajectory
    for traj_group in trajectory_groups_P[:1]:
        print_group(traj_group, tokenizer)

    # Remove groups with constant reward if configured
    filtered_trajectory_groups_P = (
        _remove_constant_reward_groups(trajectory_groups_P)
        if cfg.remove_constant_reward_groups
        else trajectory_groups_P
    )

    # Assemble training data
    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(filtered_trajectory_groups_P)
        data_D, _metadata_D = assemble_training_data(filtered_trajectory_groups_P, advantages_P)

    # Incorporate KL penalty if configured
    if cfg.kl_penalty_coef > 0:
        with timed("kl_vs_base", metrics):
            kl_penalty_metrics = await incorporate_kl_penalty(
                data_D,
                service_client.create_sampling_client(base_model=cfg.model_name),
                # ^^^ TODO: replace with the model we load, if relevant
                cfg.kl_penalty_coef,
                cfg.kl_discount_factor,
            )
        metrics.update(kl_penalty_metrics)

    # Training step
    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D,
            training_client,
            cfg.learning_rate,
            cfg.num_minibatches,
        )

    # Compute KL metrics
    with timed("compute_kl_sample_train", metrics):
        kl_sample_train_metrics = compute_kl_sample_train(data_D, training_logprobs_D)
        metrics.update(kl_sample_train_metrics)

    # Compute post-KL metrics if configured
    if cfg.compute_post_kl:
        sampling_client_for_kl = await training_client.save_weights_and_get_sampling_client_async(
            f"{i_batch + 1:06d}_kl"
        )
        # TODO: currently this saves an extra sampling client, which is wasteful. We should
        # be able to reuse this the next loop iteration if we've created it.
        with timed("compute_post_kl", metrics):
            post_kl_metrics = await compute_post_kl(data_D, sampling_client_for_kl)
            metrics.update(post_kl_metrics)

    # Log metrics
    metrics["time/total"] = time.time() - t_start
    ml_logger.log_metrics(metrics, step=i_batch)


async def main(
    cfg: Config,
):
    """Main training loop for MDP RL."""
    logging.basicConfig(level=logging.INFO, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("tinker._base_client").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0

    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )
    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank
    )

    load_state_path: str | None = (
        resume_info["state_path"] if resume_info else cfg.load_checkpoint_path
    )
    if load_state_path:
        future = await training_client.load_state_async(load_state_path)
        _ = await future.result_async()
        logger.info(f"Loaded state from {load_state_path}")

    # Get tokenizer from training client
    tokenizer = training_client.get_tokenizer()

    # Create dataset from thunk
    dataset, maybe_test_dataset = cfg.dataset_builder()
    evaluators = [evaluator() for evaluator in cfg.evaluator_builders]
    if maybe_test_dataset is not None:
        evaluators.append(RLTestSetEvaluator(maybe_test_dataset, max_tokens=cfg.max_tokens))

    num_batches = len(dataset)
    print(f"Will train on {num_batches} batches")

    # Training loop
    for i_batch in range(start_batch, num_batches):
        await do_update(
            i_batch=i_batch,
            num_batches=num_batches,
            cfg=cfg,
            training_client=training_client,
            service_client=service_client,
            evaluators=evaluators,
            dataset=dataset,
            ml_logger=ml_logger,
            tokenizer=tokenizer,
        )

    # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=cfg.log_path,
        kind="both",
        loop_state={"batch": num_batches},
    )

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")
