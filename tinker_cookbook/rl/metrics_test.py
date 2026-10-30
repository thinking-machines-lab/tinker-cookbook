"""Tests for the KL penalty that incorporate_kl_penalty adds to RL advantages."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.metrics import incorporate_kl_penalty
from tinker_cookbook.rl.types import Trajectory, Transition


def _two_turn_datum() -> tinker.Datum:
    """Prompt [1, 2], action [10, 11], observation [20], action [30].

    The datum's targets are [2, 10, 11, 20, 30] with mask [0, 1, 1, 0, 1].
    """
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
    assert datum.loss_fn_inputs["mask"].to_torch().tolist() == [0, 1, 1, 0, 1]
    return datum


def test_discounted_kl_penalty_leaves_masked_tokens_untrained():
    datum = _two_turn_datum()
    base_client = MagicMock()
    # Per-token base logprobs [-3, -1, -2, -3, -3] give action-token KL [0, 1, 2]
    # (mean 1), so the undiscounted penalty is [0, 1, 0, 0, -1].
    base_client.compute_logprobs_async = AsyncMock(
        return_value=[None, -3.0, -1.0, -2.0, -3.0, -3.0]
    )

    asyncio.run(
        incorporate_kl_penalty([datum], base_client, kl_penalty_coef=1.0, kl_discount_factor=0.5)
    )

    # Discounting moves later penalties onto earlier action tokens, but the
    # prompt token and the observation token must keep a zero advantage.
    assert datum.loss_fn_inputs["advantages"].to_torch().tolist() == [0.0, 0.875, -0.25, 0.0, -1.0]
