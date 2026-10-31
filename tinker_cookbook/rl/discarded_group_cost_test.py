"""Groups dropped for a constant reward still cost their sampling; the tracker counts them once.

The loops themselves are exercised end to end in ``constant_batch_cost_test``."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, cast

import pytest
import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition
from tinker_cookbook.utils.cost_estimation import CostTracker, TokenCostRates, TokenUsage

RATES = TokenCostRates(
    prefill_usd_per_million=1.0,
    cached_prefill_usd_per_million=0.2,
    sample_usd_per_million=2.0,
    train_usd_per_million=3.0,
)


def _group(rewards: list[float], ob_len: int = 10, ac_len: int = 4, cache_hits: int = 6):
    """One single-turn trajectory per reward; every turn has the same token counts."""
    trajectories = [
        Trajectory(
            transitions=[
                Transition(
                    ob=tinker.ModelInput.from_ints(list(range(ob_len))),
                    ac=TokensWithLogprobs(
                        tokens=[0] * ac_len,
                        maybe_logprobs=[0.0] * ac_len,
                        prompt_cache_hit_tokens=cache_hits,
                    ),
                    reward=reward,
                    episode_done=True,
                )
            ],
            final_ob=tinker.ModelInput.from_ints([]),
        )
        for reward in rewards
    ]
    return TrajectoryGroup(
        trajectories_G=trajectories,
        final_rewards_G=[0.0] * len(rewards),
        metrics_G=[{} for _ in rewards],
    )


# A two-trajectory group: 20 prompt tokens of which 12 are cache hits, 8 sampled.
GROUP_USAGE = TokenUsage(sampled=8, prefill_cached=12, prefill_uncached=8)


def test_count_discarded_rollout_records_the_rollout_usage_as_discarded():
    tracker = CostTracker(RATES)
    train.count_discarded_rollout(tracker, _group([1.0, 1.0]))
    assert tracker.cumulative == tracker.discarded == GROUP_USAGE
    train.count_discarded_rollout(tracker, _group([1.0, 1.0], cache_hits=0))
    assert tracker.cumulative == TokenUsage(sampled=16, prefill_cached=12, prefill_uncached=28)
    # Nothing was trained; a missing tracker is a no-op.
    assert tracker.cumulative.trained == 0
    train.count_discarded_rollout(None, _group([1.0, 1.0]))


def test_drop_constant_reward_group_counts_only_what_it_drops():
    tracker = CostTracker(RATES)
    on = cast(Any, SimpleNamespace(remove_constant_reward_groups=True))
    off = cast(Any, SimpleNamespace(remove_constant_reward_groups=False))
    mixed, constant = _group([0.0, 1.0]), _group([1.0, 1.0])

    assert train.drop_constant_reward_group(mixed, on, tracker) is mixed
    assert train.drop_constant_reward_group(None, on, tracker) is None
    assert tracker.cumulative == TokenUsage()
    assert train.drop_constant_reward_group(constant, on, tracker) is None
    assert tracker.cumulative == GROUP_USAGE
    # With the filter off the group is kept (and trained), so it is not counted here.
    assert train.drop_constant_reward_group(constant, off, tracker) is constant
    assert tracker.cumulative == GROUP_USAGE


def test_discarded_sampling_is_counted_apart_from_the_step():
    tracker = CostTracker(RATES)
    tracker.record_discarded_sampling(TokenUsage(sampled=100))
    # In the totals a checkpoint would save at once, and reported as discarded.
    assert train._loop_state(1, tracker) == {
        "batch": 1,
        "cumulative_tokens": asdict(TokenUsage(sampled=100)),
        "discarded_tokens": asdict(TokenUsage(sampled=100)),
    }
    assert tracker.cumulative_metrics()["tokens/discarded/sampled"] == 100
    assert tracker.cumulative_metrics()["cost/discarded_usd"] == pytest.approx(100 * 2.0 / 1e6)

    # A step counts only its own batch: step 500, cumulative 600, discarded 100.
    step = tracker.record_step(TokenUsage(sampled=500))
    assert step["tokens/sampled"] == 500
    assert step["tokens/cumulative/sampled"] == 600
    assert step["cost/step_usd"] == pytest.approx(500 * 2.0 / 1e6)
    assert step["cost/cumulative_usd"] == pytest.approx(600 * 2.0 / 1e6)
    assert tracker.discarded == TokenUsage(sampled=100)
    assert not any(k.startswith("tokens/discarded") for k in step)

    # Resumed from the checkpoint, both totals continue without double counting.
    saved = train._loop_state(2, tracker)
    resumed = CostTracker(
        RATES,
        cumulative=TokenUsage(**saved["cumulative_tokens"]),
        discarded=TokenUsage(**saved["discarded_tokens"]),
    )
    train.count_discarded_rollout(resumed, _group([1.0, 1.0]))
    assert resumed.cumulative.sampled == 608
    assert resumed.discarded.sampled == 108
    assert resumed.cumulative_metrics()["tokens/cumulative/sampled"] == 608
