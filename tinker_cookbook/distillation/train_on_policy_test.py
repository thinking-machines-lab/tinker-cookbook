"""Tests for the reverse-KL advantages in on-policy distillation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.distillation.train_on_policy import incorporate_kl_penalty
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import Trajectory, Transition


def test_discounted_kl_penalty_leaves_masked_tokens_untrained():
    # Prompt [1, 2], student action [10, 11], tool output [20], student action [30]:
    # targets [2, 10, 11, 20, 30] with mask [0, 1, 1, 0, 1].
    trajectory = Trajectory(
        transitions=[
            Transition(
                ob=tinker.ModelInput.from_ints([1, 2]),
                ac=TokensWithLogprobs(tokens=[10, 11], maybe_logprobs=[-1.0, -1.0]),
                reward=0.0,
                episode_done=False,
            ),
            Transition(
                ob=tinker.ModelInput.from_ints([1, 2, 10, 11, 20]),
                ac=TokensWithLogprobs(tokens=[30], maybe_logprobs=[-1.0]),
                reward=0.0,
                episode_done=True,
            ),
        ],
        final_ob=tinker.ModelInput.from_ints([]),
    )
    (datum,) = trajectory_to_data(trajectory, traj_advantage=0.0)
    teacher_client = MagicMock()
    # Teacher logprobs [-3, -1, -2, -3, -3] give action-token reverse KL [0, 1, 2],
    # so the undiscounted advantages are [0, 0, -1, 0, -2].
    teacher_client.compute_logprobs_async = AsyncMock(
        return_value=[None, -3.0, -1.0, -2.0, -3.0, -3.0]
    )

    asyncio.run(
        incorporate_kl_penalty(
            [datum],
            [teacher_client],
            dataset_indices_D=[0],
            kl_penalty_coef=1.0,
            kl_discount_factor=0.5,
        )
    )

    # Discounting moves later penalties onto earlier student tokens, but the
    # prompt token and the tool output token must keep a zero advantage.
    assert datum.loss_fn_inputs["advantages"].to_torch().tolist() == [0.0, -0.75, -1.5, 0.0, -2.0]
