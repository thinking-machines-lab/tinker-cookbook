"""Tests for advantage estimation functions and PPO metrics.

Tests cover GRPO and REINFORCE++ advantage estimators, PPO-specific
metrics (clip fraction, approx KL), including edge cases like single-trajectory
groups and uniform rewards.
"""

import math

import tinker
import torch

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.advantages import (
    AdvantageMethod,
    compute_advantages,
    compute_grpo_advantages,
    compute_reinforce_pp_advantages,
)
from tinker_cookbook.rl.metrics import compute_ppo_metrics
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
        # Mean=3, population std=sqrt(8/3)~=1.6330
        # Normalized advantages: (-2/1.6330, 0, 2/1.6330) ~ (-1.2247, 0, 1.2247)
        pop_std = math.sqrt(8.0 / 3.0)
        expected = torch.tensor([-2.0 / pop_std, 0.0, 2.0 / pop_std])
        assert torch.allclose(adv, expected, atol=1e-5)

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
# Dispatch function tests
# ---------------------------------------------------------------------------


class TestComputeAdvantagesDispatch:
    def test_dispatch_grpo(self) -> None:
        group = _make_group([1.0, 3.0])
        advantages_P, stats = compute_advantages([group], method=AdvantageMethod.GRPO)
        expected = compute_grpo_advantages([group])
        assert torch.allclose(advantages_P[0], expected[0])
        assert "advantages/mean" in stats
        assert "advantages/std" in stats
        assert "advantages/max" in stats
        assert "advantages/min" in stats

    def test_dispatch_reinforce_pp(self) -> None:
        group = _make_group([1.0, 3.0])
        advantages_P, stats = compute_advantages(
            [group], method=AdvantageMethod.REINFORCE_PP, normalize=True
        )
        expected = compute_reinforce_pp_advantages([group], normalize=True)
        assert torch.allclose(advantages_P[0], expected[0])
        assert "reinforce_pp/baseline_value" in stats
        # Baseline should be the mean reward (2.0), not the mean advantage (~0)
        assert abs(stats["reinforce_pp/baseline_value"] - 2.0) < 1e-5

    def test_default_is_grpo(self) -> None:
        group = _make_group([1.0, 5.0])
        advantages_P, _stats = compute_advantages([group])
        grpo = compute_grpo_advantages([group])
        assert torch.allclose(advantages_P[0], grpo[0])

    def test_stats_values_correct(self) -> None:
        """Verify that advantage stats are numerically correct."""
        group = _make_group([0.0, 2.0, 4.0])
        _advantages_P, stats = compute_advantages([group], method=AdvantageMethod.GRPO)
        # Advantages are [-2, 0, 2], mean=0, std=2, max=2, min=-2
        assert abs(stats["advantages/mean"] - 0.0) < 1e-5
        assert abs(stats["advantages/max"] - 2.0) < 1e-5
        assert abs(stats["advantages/min"] - (-2.0)) < 1e-5


# ---------------------------------------------------------------------------
# PPO metrics tests
# ---------------------------------------------------------------------------


def _make_datum_with_logprobs(
    logprobs: list[float], mask: list[float]
) -> tinker.Datum:
    """Create a minimal Datum with logprobs and mask for PPO metrics testing."""
    n = len(logprobs)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(range(n + 1))),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.zeros(n, dtype=torch.long)),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(logprobs)),
            "advantages": tinker.TensorData.from_torch(torch.zeros(n)),
            "mask": tinker.TensorData.from_torch(torch.tensor(mask)),
        },
    )


class TestPPOMetrics:
    def test_no_clip_when_logprobs_match(self) -> None:
        """When old and new logprobs are identical, clip fraction should be 0."""
        datum = _make_datum_with_logprobs([-1.0, -2.0, -3.0], [1.0, 1.0, 1.0])
        training_logprobs = [torch.tensor([-1.0, -2.0, -3.0])]
        result = compute_ppo_metrics([datum], training_logprobs, clip_eps=0.2)
        assert abs(result["ppo/clip_fraction"]) < 1e-6
        assert abs(result["ppo/approx_kl"]) < 1e-6

    def test_full_clip_when_logprobs_diverge(self) -> None:
        """When logprobs diverge significantly, all samples should be clipped."""
        datum = _make_datum_with_logprobs([-1.0, -1.0], [1.0, 1.0])
        # New logprobs much higher -> ratio >> 1+eps
        training_logprobs = [torch.tensor([0.0, 0.0])]
        result = compute_ppo_metrics([datum], training_logprobs, clip_eps=0.2)
        assert result["ppo/clip_fraction"] > 0.9  # Should be ~1.0
        # approx_kl = E[log_old - log_new] = E[-1 - 0] = -1
        # Negative because new policy is higher prob (moved away from old)
        assert abs(result["ppo/approx_kl"]) > 0.5  # Significant divergence

    def test_mask_respected(self) -> None:
        """Only action tokens (mask > 0) should contribute to metrics."""
        datum = _make_datum_with_logprobs([-1.0, -1.0, -1.0], [0.0, 1.0, 0.0])
        # Only index 1 is unmasked, and logprobs match -> no clip
        training_logprobs = [torch.tensor([-1.0, -1.0, -1.0])]
        result = compute_ppo_metrics([datum], training_logprobs, clip_eps=0.2)
        assert abs(result["ppo/clip_fraction"]) < 1e-6

    def test_approx_kl_sign(self) -> None:
        """Approx KL should be positive when new policy diverges from old."""
        datum = _make_datum_with_logprobs([-2.0, -2.0], [1.0, 1.0])
        # New policy assigns higher logprobs -> log_old - log_new < 0
        # Wait: approx_kl = E[log_old - log_new] = E[-2 - (-1)] = -1
        # But KL(old || new) via first-order: negative means new is higher prob
        training_logprobs = [torch.tensor([-3.0, -3.0])]
        result = compute_ppo_metrics([datum], training_logprobs, clip_eps=0.2)
        # log_old - log_new = -2 - (-3) = 1 > 0
        assert result["ppo/approx_kl"] > 0
