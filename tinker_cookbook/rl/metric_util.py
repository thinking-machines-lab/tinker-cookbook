import asyncio
import itertools
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinker_cookbook.stores.training_store import TrainingRunStore

import numpy as np
import tinker

from tinker_cookbook.completers import FireworksTokenCompleter, TokenCompleter
from tinker_cookbook.eval.benchmarks._types import BenchmarkResult
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.exceptions import AllTrajectoriesFailedError
from tinker_cookbook.rl.rollout_logging import (
    RolloutSummaryExportConfig,
    serialize_rollout_summaries,
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
    if n_groups == 0:
        return {
            "by_group/frac_mixed": 0.0,
            "by_group/frac_all_good": 0.0,
            "by_group/frac_all_bad": 0.0,
        }
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
    """Compute per-tag and aggregate trajectory metrics for a batch of rollouts.

    Metrics are computed globally (under the ``env/all/`` prefix) and, when
    non-trivial tags exist, per-tag (under ``env/<tag>/`` prefixes).  A tag is
    considered "non-trivial" if it selects a strict subset of all groups.

    Args:
        trajectory_groups_P (list[TrajectoryGroup]): One trajectory group per
            problem in the batch.
        taglist_P (list[list[str]]): Tags for each trajectory group, aligned
            with *trajectory_groups_P*.

    Returns:
        dict[str, float]: Flat dictionary of prefixed metric names to values.
    """
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
    total_turns = sum(turns_by_trajectory)
    total_episodes = len(flat_trajs_PG)
    total_ac_tokens = sum(ac_tokens_by_turn)
    total_ob_tokens = sum(ob_tokens_by_turn)
    # Compute metrics
    metrics = {
        "ac_tokens_per_turn": total_ac_tokens / total_turns if total_turns > 0 else 0.0,
        "ob_tokens_per_turn": total_ob_tokens / total_turns if total_turns > 0 else 0.0,
        "turns_per_episode": total_turns / total_episodes if total_episodes > 0 else 0.0,
        "total_episodes": total_episodes,
        "total_turns": total_turns,
        "total_ac_tokens": total_ac_tokens,
        "total_ob_tokens": total_ob_tokens,
    }
    all_rewards = [reward for tg in trajectory_groups_P for reward in tg.get_total_rewards()]
    metrics["reward/total"] = np.mean(all_rewards).item() if all_rewards else 0.0
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
    """Flatten an entire RL dataset into a single list of env group builders.

    Iterates over every batch in *dataset* and concatenates the resulting
    :class:`EnvGroupBuilder` lists.

    Args:
        dataset (RLDataset): The RL dataset to flatten.

    Returns:
        list[EnvGroupBuilder]: All env group builders across every batch.
    """
    return list(itertools.chain(*[dataset.get_batch(i) for i in range(len(dataset))]))


class RLTestSetEvaluator(SamplingClientEvaluator):
    """Evaluator that runs RL rollouts on a held-out test dataset.

    Rolls out every environment group in the dataset, collects trajectory
    metrics, and optionally writes per-rollout JSONL summaries.  Supports
    both direct ``TokenCompleter`` evaluation and executor-dispatched
    evaluation via a ``SamplingClient``.

    Example::

        eval_dataset = my_dataset_builder.build_test()
        evaluator = RLTestSetEvaluator(eval_dataset, max_tokens=1024, name="val")
        metrics = await evaluator(sampling_client)

    Args:
        dataset (RLDataset): The test/validation RL dataset.
        max_tokens (int): Maximum number of tokens per completion.
        name (str): Prefix added to all returned metric keys (default
            ``"test"``).
        num_groups_to_log (int): Number of leading groups for which full
            logtree logging is enabled (default ``4``).
        strategy (RolloutStrategy | None): Optional rollout error-handling
            strategy (e.g. ``FailFast``, ``RetryOnFailure``).
    """

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
        self.last_result: BenchmarkResult | None = None
        """Most recent :class:`BenchmarkResult`, populated after each evaluation."""

    async def eval_token_completer(
        self,
        policy: TokenCompleter,
        *,
        rollout_summary_export: RolloutSummaryExportConfig | None = None,
        store: "TrainingRunStore | None" = None,
    ) -> dict[str, float]:
        """Run evaluation rollouts using a :class:`TokenCompleter` policy.

        Args:
            policy (TokenCompleter): The token completer to use for generating
                actions during rollouts.
            rollout_summary_export (RolloutSummaryExportConfig | None): If
                provided, per-trajectory JSONL summaries are written to the
                configured path.

        Returns:
            dict[str, float]: Metric dictionary with keys prefixed by
                ``self.name``.
        """

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
        return self._collect_eval_metrics(results, rollout_summary_export, store=store)

    async def __call__(
        self,
        sampling_client: tinker.SamplingClient,
        *,
        rollout_summary_export: RolloutSummaryExportConfig | None = None,
        store: "TrainingRunStore | None" = None,
    ) -> dict[str, float]:
        """Evaluate the current policy checkpoint via a sampling client.

        Automatically chooses between direct rollout and executor-dispatched
        rollout depending on whether a rollout executor is registered.

        Args:
            sampling_client (tinker.SamplingClient): Sampling client pointing
                at the checkpoint to evaluate.
            rollout_summary_export (RolloutSummaryExportConfig | None): If
                provided, per-trajectory JSONL summaries are written to the
                configured path.

        Returns:
            dict[str, float]: Metric dictionary with keys prefixed by
                ``self.name``.
        """
        if get_rollout_executor() is not None:
            # Use the executor-aware dispatch path so rollouts are offloaded
            return await self._eval_with_executor(
                sampling_client, rollout_summary_export=rollout_summary_export, store=store
            )

        policy = FireworksTokenCompleter(sampling_client, max_tokens=self.max_tokens)
        return await self.eval_token_completer(
            policy,
            rollout_summary_export=rollout_summary_export,
            store=store,
        )

    async def _eval_with_executor(
        self,
        sampling_client: tinker.SamplingClient,
        *,
        rollout_summary_export: RolloutSummaryExportConfig | None = None,
        store: "TrainingRunStore | None" = None,
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
        return self._collect_eval_metrics(results, rollout_summary_export, store=store)

    def _collect_eval_metrics(
        self,
        results: list[TrajectoryGroup | None],
        rollout_summary_export: RolloutSummaryExportConfig | None,
        *,
        store: "TrainingRunStore | None" = None,
    ) -> dict[str, float]:
        """Shared logic for collecting metrics from eval rollout results.

        Also builds a :class:`BenchmarkResult` stored in :attr:`last_result`
        for callers that want typed, structured access to the evaluation outcome.
        """
        error_counter = RolloutErrorCounter()
        for result in results:
            error_counter.ingest(result)

        trajectory_groups_P = [r for r in results if r is not None]
        taglist_P = [
            builder.logging_tags()
            for builder, r in zip(self.env_group_builders_P, results)
            if r is not None
        ]
        if rollout_summary_export is not None and store is not None:
            sampling_client_steps_P = (
                [rollout_summary_export.sampling_client_step] * len(trajectory_groups_P)
                if rollout_summary_export.sampling_client_step is not None
                else None
            )
            records = serialize_rollout_summaries(
                split=rollout_summary_export.split,
                iteration=rollout_summary_export.iteration,
                trajectory_groups_P=trajectory_groups_P,
                taglist_P=taglist_P,
                sampling_client_steps_P=sampling_client_steps_P,
            )
            store.write_rollouts(
                rollout_summary_export.iteration,
                records,
                base_name=rollout_summary_export.base_name,
            )
        num_errors = sum(1 for r in results if r is None)
        if trajectory_groups_P:
            metrics = compute_trajectory_metrics(trajectory_groups_P, taglist_P)
        else:
            metrics = {}
        metrics.update(error_counter.get_metrics())

        # Build typed BenchmarkResult from the same data.
        # Count at the group level (one group = one test problem) so that
        # num_examples matches len(env_group_builders_P) regardless of
        # group_size.  Errors count in the denominator (scored as 0).
        num_groups_correct = sum(
            1 for tg in trajectory_groups_P if any(r > 0 for r in tg.get_total_rewards())
        )
        total_groups = len(results)
        self.last_result = BenchmarkResult(
            name=self.name,
            score=num_groups_correct / total_groups if total_groups > 0 else 0.0,
            num_examples=total_groups,
            num_correct=num_groups_correct,
            num_errors=num_errors,
            metrics=dict(metrics),
        )

        metrics = {f"{self.name}/{k}": v for k, v in metrics.items()}
        return metrics
