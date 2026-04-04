"""
Data processing functions for RL training.

Contains functions for computing advantages, converting trajectories to training data,
and assembling training batches.
"""

import logging

import tinker
import torch
from tinker import TensorData

from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup
from tinker_cookbook.supervised.common import (
    create_rightshifted_model_input_and_leftshifted_targets,
)
from tinker_cookbook.utils.misc_utils import all_same, safezip

logger = logging.getLogger(__name__)


def compute_advantages(trajectory_groups_P: list[TrajectoryGroup]) -> list[torch.Tensor]:
    """Compute advantages for each trajectory, centered within groups (GRPO).

    This is a convenience wrapper that delegates to
    :func:`~tinker_cookbook.rl.advantages.compute_grpo_advantages`. For
    alternative advantage estimators (e.g. REINFORCE++), use
    :func:`~tinker_cookbook.rl.advantages.compute_advantages` directly.

    Args:
        trajectory_groups_P (list[TrajectoryGroup]): Groups of trajectories,
            where each group's rewards are centered independently.

    Returns:
        list[torch.Tensor]: Per-group advantage tensors of shape ``(G,)``,
            where ``G`` is the number of trajectories in each group.
    """
    from tinker_cookbook.rl.advantages import compute_grpo_advantages

    return compute_grpo_advantages(trajectory_groups_P)


FlatObElem = int | tinker.ModelInputChunk
FlatOb = list[FlatObElem]


def _is_prefix(seq1: FlatOb, seq2: FlatOb) -> bool:
    """
    Check if seq1 is a prefix of seq2.
    """
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def _flat_ob_token_len(flat_ob: FlatOb) -> int:
    out = 0
    for elem in flat_ob:
        if isinstance(elem, int):
            out += 1
        else:
            out += elem.length
    return out


def _flat_ob_to_model_input(flat_ob: FlatOb) -> tinker.ModelInput:
    out: list[tinker.ModelInputChunk] = []
    current_text_chunk: list[int] = []

    def flush_text_chunk():
        if current_text_chunk:
            out.append(tinker.EncodedTextChunk(tokens=current_text_chunk))
            current_text_chunk.clear()

    for elem in flat_ob:
        if isinstance(elem, int):
            current_text_chunk.append(elem)
        else:
            flush_text_chunk()
            out.append(elem)
    flush_text_chunk()
    return tinker.ModelInput(chunks=out)


def _flatten_chunks(chunks: list[tinker.ModelInputChunk]) -> FlatOb:
    out: FlatOb = []
    for chunk in chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            out.extend(chunk.tokens)
        else:
            out.append(chunk)
    return out


def trajectory_to_data(traj: Trajectory, traj_advantage: float) -> list[tinker.Datum]:
    """Return one or more Datum objects corresponding to the trajectory.

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

    Args:
        traj (Trajectory): A single trajectory containing transitions
            (observation-action pairs).
        traj_advantage (float): The scalar advantage to assign to all action
            tokens in this trajectory.

    Returns:
        list[tinker.Datum]: One or more training datums, each containing
            model input, targets, sampled log-probs, advantages, and masks.
    """

    class SequenceAccumulator:
        full_sequence: list[FlatObElem] = []
        sampled_logprobs: list[float] = []
        advantages: list[float] = []
        mask: list[float] = []

        @classmethod
        def clear(cls):
            cls.full_sequence = []
            cls.sampled_logprobs = []
            cls.advantages = []
            cls.mask = []

    def make_datum_from_state():
        all_tokens_T = _flat_ob_to_model_input(SequenceAccumulator.full_sequence)
        input_tokens_T, target_tokens_T = create_rightshifted_model_input_and_leftshifted_targets(
            list(all_tokens_T.chunks)
        )
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
        return tinker.Datum(
            model_input=input_tokens_T,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens_T)),
                "logprobs": TensorData.from_torch(torch.tensor(sampled_logprobs_T)),
                "advantages": TensorData.from_torch(torch.tensor(advantages_T)),
                "mask": TensorData.from_torch(torch.tensor(mask_T)),
            },
        )

    data: list[tinker.Datum] = []
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


def assemble_training_data(
    trajectory_groups_P: list[TrajectoryGroup],
    advantages_P: list[torch.Tensor],
) -> tuple[list[tinker.Datum], list[dict[str, int]]]:
    """Convert trajectories to training data format.

    Args:
        trajectory_groups_P (list[TrajectoryGroup]): Groups of trajectories
            to convert into training datums.
        advantages_P (list[torch.Tensor]): Per-group advantage tensors,
            one per trajectory group, as returned by :func:`compute_advantages`.

    Returns:
        tuple[list[tinker.Datum], list[dict[str, int]]]: A flat list of
            training datums and a parallel list of metadata dicts mapping
            each datum back to its ``group_idx`` and ``traj_idx``.
    """
    data_D: list[tinker.Datum] = []
    metadata_D: list[dict[str, int]] = []

    for i_group, (traj_group, advantages_G) in enumerate(
        safezip(trajectory_groups_P, advantages_P)
    ):
        for i_traj, (traj, traj_advantage) in enumerate(
            safezip(traj_group.trajectories_G, advantages_G)
        ):
            # Build the full sequence from the trajectory
            new_data = trajectory_to_data(traj, float(traj_advantage))
            data_D.extend(new_data)
            metadata_D.extend([{"group_idx": i_group, "traj_idx": i_traj} for _ in new_data])

    return data_D, metadata_D


def remove_constant_reward_groups(
    trajectory_groups_P: list[TrajectoryGroup],
) -> list[TrajectoryGroup]:
    """Filter out trajectory groups where all trajectories received the same reward.

    Groups with uniform rewards produce zero advantage for every trajectory,
    contributing no gradient signal.  Removing them avoids wasted compute.
    If *all* groups are uniform, a single group is returned so downstream code
    that expects a non-empty list does not break.

    Args:
        trajectory_groups_P (list[TrajectoryGroup]): Groups of trajectories
            to filter.

    Returns:
        list[TrajectoryGroup]: The subset of groups that contain at least two
            distinct reward values, or a singleton list if every group was
            uniform.
    """
    new_groups: list[TrajectoryGroup] = []
    for group in trajectory_groups_P:
        if not all_same(group.get_total_rewards()):
            new_groups.append(group)
    if not new_groups:
        logger.warning("All rewards are uniform. There will be no gradient")
        return trajectory_groups_P[0:1]  # return singleton list in case empty
        # list will cause problems
    return new_groups
