"""
Implements RL on general MDPs

"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, cast

import chz
import numpy as np
import tinker
import torch
from tinker import types
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.display import colorize_example
from tinker_cookbook.evaluators import SamplingClientEvaluatorBuilder
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import (
    RLDatasetBuilder,
    Trajectory,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils.misc_utils import all_same, safezip, timed
from tinker_cookbook.utils.ml_log import setup_logging

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
    logprob_results_D = await asyncio.gather(
        *[
            post_sampling_client.compute_logprobs_async(sequence_input)
            for sequence_input in full_sequence_inputs_D
        ]
    )
    prev_logprobs_list = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    masks = [prev_logprobs != 0 for prev_logprobs in prev_logprobs_list]
    # ^^^ a bit of a hack. note that we put 0 in logprobs when building the datum in to_data
    flat_diffs = [
        (torch.tensor(result[1:]) - prev_logprobs)[mask]
        for result, prev_logprobs, mask in safezip(logprob_results_D, prev_logprobs_list, masks)
    ]
    flat_diffs = torch.cat(flat_diffs)
    kl_post_v1 = flat_diffs.mean().item()
    kl_post_v2 = 0.5 * (flat_diffs**2).mean().item()

    return {"kl_post_v1": kl_post_v1, "kl_post_v2": kl_post_v2}


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


def _remove_uniform_rewards(trajectory_groups_P: List[TrajectoryGroup]) -> List[TrajectoryGroup]:
    new_groups = []
    for group in trajectory_groups_P:
        if not all_same(group.get_total_rewards()):
            new_groups.append(group)
    if not new_groups:
        logger.warning("All rewards are uniform. There will be no gradient")
        return trajectory_groups_P
    return new_groups


async def train_step(
    trajectory_groups_P: List[TrajectoryGroup],
    training_client: tinker.TrainingClient,
    learning_rate: float,
    remove_uniform_rewards: bool = True,
) -> tuple[List[torch.Tensor], List[types.Datum]]:
    """Train the model on collected trajectories."""
    # Compute advantages
    if remove_uniform_rewards:
        trajectory_groups_P = _remove_uniform_rewards(trajectory_groups_P)

    advantages_P = compute_advantages(trajectory_groups_P)

    # Assemble training data
    data_D, _metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    # Forward-backward pass
    fwd_bwd_future = await training_client.forward_backward_async(
        list(map(_remove_mask, data_D)), loss_fn="importance_sampling"
    )

    # Optimizer step
    adam_params = types.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    optim_step_future = await training_client.optim_step_async(adam_params)

    fwd_bwd_result = await fwd_bwd_future.result_async()
    _optim_step_result = await optim_step_future.result_async()

    # Extract training logprobs from loss_fn_outputs
    training_logprobs_D = []
    for output in fwd_bwd_result.loss_fn_outputs:
        training_logprobs = output["logprobs"].to_torch()
        training_logprobs_D.append(training_logprobs)
    # We dont display fwd_bwd_result.metrics to avoid spam
    return training_logprobs_D, data_D


@chz.chz
class Config:
    learning_rate: float
    dataset_builder: RLDatasetBuilder  # also determines batch size
    model_name: str
    max_tokens: int
    compute_post_kl: bool = False
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_relpath: str
    base_url: str | None = None

    test_interval: int = 1
    load_checkpoint_path: str | None = None

    @property
    def log_base_dir(self) -> str:
        return os.path.expanduser("~/experiments")


async def main(
    cfg: Config,
):
    """Main training loop for MDP RL."""
    log_dir = str(Path(cfg.log_base_dir) / cfg.log_relpath)
    logging.basicConfig(level=logging.INFO, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("tinker._base_client").setLevel(logging.WARNING)

    ml_logger = setup_logging(
        log_dir=log_dir,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )
    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = await service_client.create_lora_training_client_async(cfg.model_name)
    if cfg.load_checkpoint_path is not None:
        future = await training_client.load_state_async(cfg.load_checkpoint_path)
        _ = await future.result_async()
        logger.info(f"Loaded state from {cfg.load_checkpoint_path}")

    # Initial weight save
    save_index = 0

    async def save_weights(training_client: tinker.TrainingClient, save_index: int) -> str:
        """Save current weights and return the path."""
        checkpoint_name = f"{save_index:04d}"
        save_sampler_future = await training_client.save_weights_for_sampler_async(checkpoint_name)
        save_state_future = await training_client.save_state_async(checkpoint_name)
        save_sampler_result = await save_sampler_future.result_async()
        save_state_result = await save_state_future.result_async()
        logger.info(f"Saved sampler weights to {save_sampler_result.path}")
        logger.info(f"Saved state to {save_state_result.path}")
        # XXX saving state due to bug
        return save_sampler_result.path

    current_weights_path = await save_weights(training_client, save_index)
    save_index += 1

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
    for i_batch in range(num_batches):
        t_start = time.time()
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        env_group_builders_P = dataset.get_batch(i_batch)
        with timed("sample", metrics):
            # Create sampling client with current weights
            sampling_client = training_client.create_sampling_client(current_weights_path)
            policy = TinkerTokenCompleter(sampling_client, max_tokens=cfg.max_tokens)
            trajectory_groups_P = await asyncio.gather(
                *[do_group_rollout(builder, policy) for builder in env_group_builders_P]
            )

        # Compute some metrics
        metrics.update(compute_trajectory_metrics(trajectory_groups_P))
        # Print one trajectory
        for traj_group in trajectory_groups_P[:1]:
            print_group(traj_group, tokenizer)

        with timed("train", metrics):
            training_logprobs_D, data_D = await train_step(
                trajectory_groups_P, training_client, cfg.learning_rate
            )

        with timed("compute_kl_sample_train", metrics):
            kl_sample_train_metrics = compute_kl_sample_train(data_D, training_logprobs_D)
            metrics.update(kl_sample_train_metrics)

        with timed("save_weights", metrics):
            current_weights_path = await save_weights(training_client, save_index)
            save_index += 1

        if cfg.compute_post_kl:
            with timed("compute_post_kl", metrics):
                post_sampling_client = training_client.create_sampling_client(current_weights_path)
                post_kl_metrics = await compute_post_kl(data_D, post_sampling_client)
                metrics.update(post_kl_metrics)

        if cfg.test_interval > 0 and i_batch % cfg.test_interval == 0:
            sampling_client = training_client.create_sampling_client(current_weights_path)
            with timed("run_evals", metrics):
                for evaluator in evaluators:
                    eval_metrics = await evaluator(sampling_client)
                    metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

        # Log metrics
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)

    return current_weights_path
