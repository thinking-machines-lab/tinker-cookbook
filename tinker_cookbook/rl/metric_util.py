import asyncio
import itertools
from typing import Dict, List

import numpy as np
import tinker_public
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.evaluators import SamplingClientEvaluator
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, TrajectoryGroup
from tinker_cookbook.utils.misc_utils import all_same, dict_mean


def _compute_by_group_metrics(trajectory_groups_P: List[TrajectoryGroup], good_thresh: float = 0.5):
    n_groups = len(trajectory_groups_P)
    n_mixed = n_good = n_bad = 0
    for tg in trajectory_groups_P:
        grp_rewards = tg.get_total_rewards()
        if all_same(grp_rewards):
            if grp_rewards[0] >= good_thresh:
                n_good += 1
            else:
                n_bad += 1
        else:
            n_mixed += 1
    return {
        "by_group/frac_mixed": n_mixed / n_groups,
        "by_group/frac_all_good": n_good / n_groups,
        "by_group/frac_all_bad": n_bad / n_groups,
    }


def compute_trajectory_metrics(trajectory_groups_P: List[TrajectoryGroup]) -> Dict[str, float]:
    """Compute metrics for the trajectory groups."""
    flat_trajs_PG = [traj for tg in trajectory_groups_P for traj in tg.trajectories_G]
    ac_tokens_by_turn = [
        len(transition.ac.tokens) for traj in flat_trajs_PG for transition in traj.transitions
    ]
    ob_tokens_by_turn = [
        transition.ob.length for traj in flat_trajs_PG for transition in traj.transitions
    ]
    turns_by_trajectory = [len(traj.transitions) for traj in flat_trajs_PG]
    # Compute metrics
    metrics = {
        "mean/ac_tokens_per_turn": sum(ac_tokens_by_turn) / sum(turns_by_trajectory),
        "mean/ob_tokens_per_turn": sum(ob_tokens_by_turn) / sum(turns_by_trajectory),
        "mean/turns_per_episode": sum(turns_by_trajectory) / len(flat_trajs_PG),
        "total/episodes": len(flat_trajs_PG),
        "total/turns": sum(turns_by_trajectory),
        "total/ac_tokens": sum(ac_tokens_by_turn),
        "total/ob_tokens": sum(ob_tokens_by_turn),
    }
    metrics["reward/total"] = np.mean(
        [reward for tg in trajectory_groups_P for reward in tg.get_total_rewards()]
    )
    # Per-transition metrics
    _all_transitions = [
        transition.metrics
        for tg in trajectory_groups_P
        for traj in tg.trajectories_G
        for transition in traj.transitions
    ]
    metrics.update(dict_mean(_all_transitions))
    # Final metrics
    metrics.update(dict_mean([metrics for tg in trajectory_groups_P for metrics in tg.metrics_G]))
    metrics.update(_compute_by_group_metrics(trajectory_groups_P))
    return metrics


def dataset_to_env_group_builders(dataset: RLDataset) -> list[EnvGroupBuilder]:
    """
    Get the whole dataset as a list of env group builders.
    """
    return list(itertools.chain(*[dataset.get_batch(i) for i in range(len(dataset))]))


class RLTestSetEvaluator(SamplingClientEvaluator):
    def __init__(self, dataset: RLDataset, max_tokens: int):
        self.env_group_builders_P = dataset_to_env_group_builders(dataset)
        self.max_tokens = max_tokens

    async def __call__(self, sampling_client: tinker_public.SamplingClient) -> dict[str, float]:
        policy = TinkerTokenCompleter(sampling_client, max_tokens=self.max_tokens)
        trajectory_groups_P = await asyncio.gather(
            *[do_group_rollout(builder, policy) for builder in self.env_group_builders_P]
        )
        metrics = compute_trajectory_metrics(trajectory_groups_P)
        return metrics
