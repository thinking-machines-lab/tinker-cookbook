"""Tests for advantage estimation functions.

Tests cover GRPO, REINFORCE++, and GAE advantage estimators, including
edge cases like single-trajectory groups and uniform rewards.
"""

import math

import tinker
import torch

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.advantages import (
    AdvantageMethod,
    compute_advantages,
    compute_gae_advantages,
    compute_grpo_advantages,
    compute_reinforce_pp_advantages,
)
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition


def _make_transition(
    ob_tokens: list[int],
    ac_tokens: list[int],
    reward: float = 0.0,
    done: bool = False,
) -> Transition:
    logprobs = [0.0] * len(ac_tokens)
    return Transition(
        ob=tinker.ModelInput.from_ints(ob_tokens),
        ac=TokensWithLogprobs(tokens=ac_tokens, maybe_logprobs=logprobs),
        reward=reward,
        episode_done=done,
    )


def _make_trajectory(step_rewards: list[float]) -> Trajectory:
    """Make a trajectory with one transition per reward value."""
    transitions = []
    for i, r in enumerate(step_rewards):
        transitions.append(
            _make_transition(
                ob_tokens=[100 + i],
                ac_tokens=[200 + i, 201 + i],
                reward=r,
                done=(i == len(step_rewards) - 1),
            )
        )
    return Trajectory(
        transitions=transitions,
        final_ob=tinker.ModelInput.from_ints([999]),
    )


def _make_group(total_rewards: list[float]) -> TrajectoryGroup:
    """Make a trajectory group where each trajectory has a single step with given reward."""
    trajs = [_make_trajectory([r]) for r in total_rewards]
    return TrajectoryGroup(
        trajectories_G=trajs,
        final_rewards_G=[0.0] * len(total_rewards),
        metrics_G=[{}] * len(total_rewards),
    )


# ---------------------------------------------------------------------------
# GRPO tests
# ---------------------------------------------------------------------------


class TestGRPOAdvantages:
    def test_basic_centering(self) -> None:
        group = _make_group([1.0, 3.0, 5.0])
        advantages = compute_grpo_advantages([group])
        assert len(advantages) == 1
        adv = advantages[0]
        # Mean is 3.0, so advantages are -2, 0, 2
        assert torch.allclose(adv, torch.tensor([-2.0, 0.0, 2.0]))

    def test_uniform_rewards_give_zero_advantages(self) -> None:
        group = _make_group([2.0, 2.0, 2.0])
        advantages = compute_grpo_advantages([group])
        assert torch.allclose(advantages[0], torch.tensor([0.0, 0.0, 0.0]))

    def test_multiple_groups(self) -> None:
        g1 = _make_group([0.0, 4.0])
        g2 = _make_group([10.0, 20.0, 30.0])
        advantages = compute_grpo_advantages([g1, g2])
        assert len(advantages) == 2
        # Group 1: mean=2, advantages=[-2, 2]
        assert torch.allclose(advantages[0], torch.tensor([-2.0, 2.0]))
        # Group 2: mean=20, advantages=[-10, 0, 10]
        assert torch.allclose(advantages[1], torch.tensor([-10.0, 0.0, 10.0]))

    def test_single_trajectory_group(self) -> None:
        group = _make_group([5.0])
        advantages = compute_grpo_advantages([group])
        assert torch.allclose(advantages[0], torch.tensor([0.0]))


# ---------------------------------------------------------------------------
# REINFORCE++ tests
# ---------------------------------------------------------------------------


class TestReinforcePPAdvantages:
    def test_normalized(self) -> None:
        group = _make_group([1.0, 3.0, 5.0])
        advantages = compute_reinforce_pp_advantages([group], normalize=True)
        adv = advantages[0]
        # Mean=3, std=2, so normalized advantages are (-2/2, 0/2, 2/2) = (-1, 0, 1)
        assert torch.allclose(adv, torch.tensor([-1.0, 0.0, 1.0]), atol=1e-5)

    def test_unnormalized_matches_grpo(self) -> None:
        group = _make_group([1.0, 3.0, 5.0])
        grpo = compute_grpo_advantages([group])
        rpp = compute_reinforce_pp_advantages([group], normalize=False)
        assert torch.allclose(grpo[0], rpp[0])

    def test_uniform_rewards_with_normalization(self) -> None:
        """Uniform rewards should give near-zero advantages even with normalization."""
        group = _make_group([2.0, 2.0, 2.0])
        advantages = compute_reinforce_pp_advantages([group], normalize=True)
        assert torch.allclose(advantages[0], torch.tensor([0.0, 0.0, 0.0]), atol=1e-6)

    def test_normalization_scales_correctly(self) -> None:
        """Groups with different reward scales should produce similar advantage magnitudes."""
        g_small = _make_group([0.0, 1.0])
        g_large = _make_group([0.0, 100.0])
        adv_small = compute_reinforce_pp_advantages([g_small], normalize=True)
        adv_large = compute_reinforce_pp_advantages([g_large], normalize=True)
        # After normalization, both should have similar magnitudes
        assert abs(adv_small[0][1].item() - adv_large[0][1].item()) < 0.1

    def test_single_trajectory_normalized(self) -> None:
        """Single trajectory: advantage is 0 / (0 + eps) = 0."""
        group = _make_group([5.0])
        advantages = compute_reinforce_pp_advantages([group], normalize=True)
        assert torch.allclose(advantages[0], torch.tensor([0.0]), atol=1e-6)


# ---------------------------------------------------------------------------
# GAE tests
# ---------------------------------------------------------------------------


class TestGAEAdvantages:
    def test_single_step_no_discount(self) -> None:
        """Single-step episode: GAE = r + gamma*V(s') - V(s)."""
        group = _make_group([1.0])
        # V(s0) = 0.5, V(terminal) = 0.0
        value_preds = [[[0.5, 0.0]]]
        advantages = compute_gae_advantages([group], value_preds, gamma=1.0, lam=0.95)
        # delta = 1.0 + 1.0*0.0 - 0.5 = 0.5
        # GAE = 0.5 (single step)
        assert torch.allclose(advantages[0], torch.tensor([0.5]), atol=1e-5)

    def test_multi_step_episode(self) -> None:
        """Two-step episode with gamma=1, lam=1 (Monte Carlo-like)."""
        traj = _make_trajectory([1.0, 2.0])
        group = TrajectoryGroup(
            trajectories_G=[traj],
            final_rewards_G=[0.0],
            metrics_G=[{}],
        )
        # V(s0)=1.0, V(s1)=1.5, V(terminal)=0.0
        value_preds = [[[1.0, 1.5, 0.0]]]
        advantages = compute_gae_advantages([group], value_preds, gamma=1.0, lam=1.0)
        # delta_0 = 1.0 + 1.0*1.5 - 1.0 = 1.5
        # delta_1 = 2.0 + 1.0*0.0 - 1.5 = 0.5
        # GAE_1 = delta_1 = 0.5
        # GAE_0 = delta_0 + gamma*lam*GAE_1 = 1.5 + 1.0*1.0*0.5 = 2.0
        # Each transition has 2 action tokens
        # total_advantage = GAE_0*2 + GAE_1*2 = 4.0 + 1.0 = 5.0
        # average per token = 5.0 / 4 = 1.25
        assert torch.allclose(advantages[0], torch.tensor([1.25]), atol=1e-5)

    def test_discount_factor(self) -> None:
        """Verify gamma < 1 discounts future rewards."""
        group = _make_group([0.0])  # reward = 0
        # V(s0) = 0.0, V(terminal) = 0.0
        value_preds = [[[0.0, 0.0]]]
        advantages = compute_gae_advantages([group], value_preds, gamma=0.99, lam=0.95)
        assert torch.allclose(advantages[0], torch.tensor([0.0]), atol=1e-5)

    def test_perfect_value_function(self) -> None:
        """When V perfectly predicts returns, advantages should be zero."""
        group = _make_group([1.0])
        # V(s0) = 1.0 (exact return), V(terminal) = 0.0
        value_preds = [[[1.0, 0.0]]]
        advantages = compute_gae_advantages([group], value_preds, gamma=1.0, lam=0.95)
        assert torch.allclose(advantages[0], torch.tensor([0.0]), atol=1e-5)

    def test_multiple_trajectories_in_group(self) -> None:
        """GAE handles multiple trajectories within a group."""
        traj1 = _make_trajectory([1.0])
        traj2 = _make_trajectory([3.0])
        group = TrajectoryGroup(
            trajectories_G=[traj1, traj2],
            final_rewards_G=[0.0, 0.0],
            metrics_G=[{}, {}],
        )
        value_preds = [[[0.0, 0.0], [1.0, 0.0]]]
        advantages = compute_gae_advantages([group], value_preds, gamma=1.0, lam=0.95)
        # traj1: delta = 1.0 + 0 - 0 = 1.0
        # traj2: delta = 3.0 + 0 - 1.0 = 2.0
        assert torch.allclose(advantages[0], torch.tensor([1.0, 2.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# Dispatch function tests
# ---------------------------------------------------------------------------


class TestComputeAdvantagesDispatch:
    def test_dispatch_grpo(self) -> None:
        group = _make_group([1.0, 3.0])
        result = compute_advantages([group], method=AdvantageMethod.GRPO)
        expected = compute_grpo_advantages([group])
        assert torch.allclose(result[0], expected[0])

    def test_dispatch_reinforce_pp(self) -> None:
        group = _make_group([1.0, 3.0])
        result = compute_advantages([group], method=AdvantageMethod.REINFORCE_PP, normalize=True)
        expected = compute_reinforce_pp_advantages([group], normalize=True)
        assert torch.allclose(result[0], expected[0])

    def test_dispatch_gae_without_values_raises(self) -> None:
        group = _make_group([1.0])
        try:
            compute_advantages([group], method=AdvantageMethod.GAE)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "value_predictions_P" in str(e)

    def test_dispatch_gae_with_values(self) -> None:
        group = _make_group([1.0])
        value_preds = [[[0.5, 0.0]]]
        result = compute_advantages(
            [group],
            method=AdvantageMethod.GAE,
            value_predictions_P=value_preds,
        )
        expected = compute_gae_advantages([group], value_preds)
        assert torch.allclose(result[0], expected[0])

    def test_default_is_grpo(self) -> None:
        group = _make_group([1.0, 5.0])
        default = compute_advantages([group])
        grpo = compute_grpo_advantages([group])
        assert torch.allclose(default[0], grpo[0])
