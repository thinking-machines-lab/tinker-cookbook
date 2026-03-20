import asyncio
import itertools
import logging
from collections import defaultdict

import numpy as np
import tinker

from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.exceptions import AllTrajectoriesFailedError
from tinker_cookbook.rl.rollout_logging import (
    RolloutSummaryExportConfig,
    write_rollout_summaries_jsonl,
)
from tinker_cookbook.rl.rollout_strategy import RolloutStrategy
from tinker_cookbook.rl.rollouts import (
    RolloutErrorCounter,
    do_group_rollout,
    do_group_rollout_and_filter_constant_reward,
    get_rollout_executor,
)
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, TrajectoryGroup
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.misc_utils import all_same, dict_mean

logger = logging.getLogger(__name__)


def _compute_by_group_metrics(trajectory_groups_P: list[TrajectoryGroup], good_thresh: float = 0.5):
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


def compute_trajectory_metrics(
    trajectory_groups_P: list[TrajectoryGroup], taglist_P: list[list[str]]
) -> dict[str, float]:
    tag2trajgroups = defaultdict(list)
    for taglist, trajectory_group in zip(taglist_P, trajectory_groups_P):
        for tag in taglist:
            tag2trajgroups[tag].append(trajectory_group)
    out = {}
    have_nontrivial_tags = any(
        len(trajgroups) < len(trajectory_groups_P) for trajgroups in tag2trajgroups.values()
    )  # check if any tag gives us a strict subset of the full trajectory groups
    if have_nontrivial_tags:
        for tag, trajectory_groups in tag2trajgroups.items():
            prefixed_metrics = {
                f"env/{tag}/{k}": v
                for k, v in _compute_trajectory_metrics(trajectory_groups).items()
            }
            out.update(prefixed_metrics)
    out.update(
        {f"env/all/{k}": v for k, v in _compute_trajectory_metrics(trajectory_groups_P).items()}
    )
    return out


def _compute_trajectory_metrics(trajectory_groups_P: list[TrajectoryGroup]) -> dict[str, float]:
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
        "ac_tokens_per_turn": sum(ac_tokens_by_turn) / sum(turns_by_trajectory),
        "ob_tokens_per_turn": sum(ob_tokens_by_turn) / sum(turns_by_trajectory),
        "turns_per_episode": sum(turns_by_trajectory) / len(flat_trajs_PG),
        "total_episodes": len(flat_trajs_PG),
        "total_turns": sum(turns_by_trajectory),
        "total_ac_tokens": sum(ac_tokens_by_turn),
        "total_ob_tokens": sum(ob_tokens_by_turn),
    }
    metrics["reward/total"] = np.mean(
        [reward for tg in trajectory_groups_P for reward in tg.get_total_rewards()]
    ).item()
    # Per-transition metrics
    transition_metrics = [
        transition.metrics
        for tg in trajectory_groups_P
        for traj in tg.trajectories_G
        for transition in traj.transitions
    ]
    traj_metrics = [metrics for tg in trajectory_groups_P for metrics in tg.metrics_G]
    metrics.update(dict_mean(transition_metrics + traj_metrics))
    # combine traj_metrics and transition_metrics in case there's some key
    # (like format error) that appears in the per-step metrics for some envs
    # but the compute_group_rewards metric for other envs.
    metrics.update(_compute_by_group_metrics(trajectory_groups_P))
    return metrics


def dataset_to_env_group_builders(dataset: RLDataset) -> list[EnvGroupBuilder]:
    """
    Get the whole dataset as a list of env group builders.
    """
    return list(itertools.chain(*[dataset.get_batch(i) for i in range(len(dataset))]))


class RLTestSetEvaluator(SamplingClientEvaluator):
    def __init__(
        self,
        dataset: RLDataset,
        max_tokens: int,
        name: str = "test",
        num_groups_to_log: int = 4,
        strategy: RolloutStrategy | None = None,
    ):
        self.env_group_builders_P = dataset_to_env_group_builders(dataset)
        self.max_tokens = max_tokens
        self.name = name
        self.num_groups_to_log = num_groups_to_log
        self.strategy = strategy

    async def eval_token_completer(
        self,
        policy: TokenCompleter,
        *,
        rollout_summary_export: RolloutSummaryExportConfig | None = None,
    ) -> dict[str, float]:
        async def run_group_rollout(
            builder: EnvGroupBuilder, group_idx: int
        ) -> TrajectoryGroup | None:
            enable_logging = group_idx < self.num_groups_to_log
            try:
                with logtree.optional_enable_logging(enable=enable_logging):
                    result = await do_group_rollout(
                        builder,
                        policy,
                        strategy=self.strategy,
                    )
            except AllTrajectoriesFailedError as e:
                logger.warning(f"Eval: {e}")
                result = None
            except Exception as e:
                if self.strategy is None or not self.strategy.catches_group_errors:
                    raise
                logger.warning(f"Eval rollout error ({type(e).__name__}): {e}")
                result = None
            return result

        results = await asyncio.gather(
            *[
                run_group_rollout(builder, group_idx)
                for group_idx, builder in enumerate(self.env_group_builders_P)
            ]
        )
        return self._collect_eval_metrics(results, rollout_summary_export)

    async def __call__(
        self,
        sampling_client: tinker.SamplingClient,
        *,
        rollout_summary_export: RolloutSummaryExportConfig | None = None,
    ) -> dict[str, float]:
        if get_rollout_executor() is not None:
            # Use the executor-aware dispatch path so rollouts are offloaded
            return await self._eval_with_executor(
                sampling_client, rollout_summary_export=rollout_summary_export
            )

        policy = TinkerTokenCompleter(sampling_client, max_tokens=self.max_tokens)
        return await self.eval_token_completer(
            policy,
            rollout_summary_export=rollout_summary_export,
        )

    async def _eval_with_executor(
        self,
        sampling_client: tinker.SamplingClient,
        *,
        rollout_summary_export: RolloutSummaryExportConfig | None = None,
    ) -> dict[str, float]:
        """Run evaluation with rollouts dispatched via the rollout executor."""
        results = await asyncio.gather(
            *[
                do_group_rollout_and_filter_constant_reward(
                    sampling_client,
                    builder,
                    max_tokens=self.max_tokens,
                    temperature=1.0,
                    do_remove_constant_reward_groups=False,
                    enable_logging=i < self.num_groups_to_log,
                    strategy=self.strategy,
                )
                for i, builder in enumerate(self.env_group_builders_P)
            ]
        )
        return self._collect_eval_metrics(results, rollout_summary_export)

    def _collect_eval_metrics(
        self,
        results: list[TrajectoryGroup | None],
        rollout_summary_export: RolloutSummaryExportConfig | None,
    ) -> dict[str, float]:
        """Shared logic for collecting metrics from eval rollout results."""
        error_counter = RolloutErrorCounter()
        for result in results:
            error_counter.ingest(result)

        trajectory_groups_P = [r for r in results if r is not None]
        taglist_P = [
            builder.logging_tags()
            for builder, r in zip(self.env_group_builders_P, results)
            if r is not None
        ]
        if rollout_summary_export is not None:
            sampling_client_steps_P = (
                [rollout_summary_export.sampling_client_step] * len(trajectory_groups_P)
                if rollout_summary_export.sampling_client_step is not None
                else None
            )
            write_rollout_summaries_jsonl(
                rollout_summary_export.path,
                split=rollout_summary_export.split,
                iteration=rollout_summary_export.iteration,
                trajectory_groups_P=trajectory_groups_P,
                taglist_P=taglist_P,
                sampling_client_steps_P=sampling_client_steps_P,
            )
        metrics = compute_trajectory_metrics(trajectory_groups_P, taglist_P)
        metrics.update(error_counter.get_metrics())
        metrics = {f"{self.name}/{k}": v for k, v in metrics.items()}
        return metrics
