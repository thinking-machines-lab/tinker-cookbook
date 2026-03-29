"""Utilities for exporting per-rollout records to JSONL."""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.utils.misc_utils import safezip

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutSummaryExportConfig:
    """Location and metadata for one rollout-summary JSONL export.

    Groups the filesystem path, dataset split name, training iteration, and
    optional sampling-client step into a single configuration object that is
    passed to rollout-summary writers.

    Attributes:
        path (Path): Destination path for the JSONL file.
        split (str): Dataset split name (e.g. ``"train"``, ``"test"``).
        iteration (int): Training iteration / batch index.
        sampling_client_step (int | None): Step counter of the sampling client
            used to generate the rollouts, or ``None`` if not applicable.
    """

    path: Path
    split: str
    iteration: int
    sampling_client_step: int | None = None


@dataclass(frozen=True)
class RolloutSummaryGroup:
    """One group of trajectories to serialize into rollout-summary JSONL records.

    Bundles a :class:`TrajectoryGroup` with its logging tags and the optional
    sampling-client step so that serialization helpers can iterate over a flat
    sequence of these objects.

    Attributes:
        trajectory_group (TrajectoryGroup): The trajectory group to serialize.
        tags (list[str]): Logging / categorization tags (e.g. environment name).
        sampling_client_step (int | None): Step counter of the sampling client,
            or ``None`` if not tracked.
    """

    trajectory_group: TrajectoryGroup
    tags: list[str]
    sampling_client_step: int | None = None


def _json_safe(value: Any) -> Any:
    """Convert values to JSON-serializable form."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            logger.debug("Failed to convert %r via .item(), falling back to str()", type(value))
    return str(value)


def write_rollout_summaries_jsonl(
    path: str | Path,
    *,
    split: str,
    iteration: int,
    trajectory_groups_P: Sequence[TrajectoryGroup],
    taglist_P: Sequence[list[str]],
    sampling_client_steps_P: Sequence[int | None] | None = None,
) -> None:
    """Write one JSON record per rollout trajectory to a JSONL file.

    Each line in the output file is a self-contained JSON object describing a
    single trajectory, including per-step observations, actions, rewards, and
    metrics.  The output is intentionally disaggregated -- no aggregate or
    summary statistics are included -- so downstream consumers can slice and
    compute their own summaries.

    Args:
        path (str | Path): Destination file path.  Parent directories are
            created automatically if they do not exist.
        split (str): Dataset split identifier written into every record
            (e.g. ``"train"``, ``"test"``).
        iteration (int): Training iteration / batch index.
        trajectory_groups_P (Sequence[TrajectoryGroup]): One trajectory group
            per problem (subscript ``_P``).
        taglist_P (Sequence[list[str]]): Tags for each trajectory group,
            aligned with *trajectory_groups_P*.
        sampling_client_steps_P (Sequence[int | None] | None): Per-group
            sampling-client step counters, or ``None`` to omit.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w") as f:
        for group_idx, (trajectory_group, tags) in enumerate(
            safezip(trajectory_groups_P, taglist_P)
        ):
            total_rewards_G = trajectory_group.get_total_rewards()
            sampling_step = (
                sampling_client_steps_P[group_idx] if sampling_client_steps_P is not None else None
            )

            for traj_idx, trajectory in enumerate(trajectory_group.trajectories_G):
                steps = []
                for step_idx, transition in enumerate(trajectory.transitions):
                    steps.append(
                        {
                            "step_idx": step_idx,
                            "ob_len": transition.ob.length,
                            "ac_len": len(transition.ac.tokens),
                            "reward": transition.reward,
                            "episode_done": transition.episode_done,
                            "metrics": transition.metrics,
                            "logs": transition.logs,
                        }
                    )

                record = {
                    "schema_version": 1,
                    "split": split,
                    "iteration": iteration,
                    "group_idx": group_idx,
                    "traj_idx": traj_idx,
                    "tags": list(tags),
                    "sampling_client_step": sampling_step,
                    "total_reward": total_rewards_G[traj_idx],
                    "final_reward": trajectory_group.final_rewards_G[traj_idx],
                    "trajectory_metrics": trajectory_group.metrics_G[traj_idx],
                    "steps": steps,
                    "final_ob_len": trajectory.final_ob.length,
                }
                f.write(json.dumps(_json_safe(record)) + "\n")


def rollout_summaries_jsonl_path(iteration_dir: Path, base_name: str) -> Path:
    """Build the rollout-summary JSONL path inside an iteration subdirectory.

    Args:
        iteration_dir (Path): Directory for the current training iteration.
        base_name (str): Prefix used to distinguish different summary files
            (e.g. ``"train"`` or ``"test"``).

    Returns:
        Path: The constructed path, e.g.
            ``iteration_dir / "train_rollout_summaries.jsonl"``.
    """
    return iteration_dir / f"{base_name}_rollout_summaries.jsonl"


def write_rollout_summaries_jsonl_from_groups(
    path: Path,
    *,
    split: str,
    iteration: int,
    groups_P: Sequence[RolloutSummaryGroup],
) -> None:
    """Serialize rollout summaries from pre-grouped records to a JSONL file.

    A convenience wrapper around :func:`write_rollout_summaries_jsonl` that
    accepts a sequence of :class:`RolloutSummaryGroup` objects instead of
    parallel sequences of trajectory groups, tags, and sampling steps.

    Args:
        path (Path): Destination file path.
        split (str): Dataset split identifier (e.g. ``"train"``).
        iteration (int): Training iteration / batch index.
        groups_P (Sequence[RolloutSummaryGroup]): One summary group per
            problem, each containing a trajectory group, tags, and an optional
            sampling-client step.
    """
    write_rollout_summaries_jsonl(
        path,
        split=split,
        iteration=iteration,
        trajectory_groups_P=[group.trajectory_group for group in groups_P],
        taglist_P=[group.tags for group in groups_P],
        sampling_client_steps_P=[group.sampling_client_step for group in groups_P],
    )
