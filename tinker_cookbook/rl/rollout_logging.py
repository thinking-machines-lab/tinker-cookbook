"""Utilities for exporting per-rollout records to JSONL.

Rollout Summary Schema (v3)
---------------------------

Each trajectory produces one JSON record written to
``{iteration_dir}/{base_name}_rollout_summaries.jsonl``.

Top-level fields::

    {
        "schema_version": 3,
        "split": "train" | "eval/{label}",
        "iteration": int,           # Training batch index
        "group_idx": int,           # GRPO group index (same problem)
        "traj_idx": int,            # Trajectory index within the group
        "tags": ["math", "gsm8k"],  # From EnvGroupBuilder.logging_tags()
        "sampling_client_step": int | null,  # Checkpoint step used for sampling
        "model_name": str | null,   # Model name from training config

        # Rewards
        "total_reward": float,      # sum(step rewards) + final_reward
        "final_reward": float,      # From EnvGroupBuilder.compute_group_rewards()
        "trajectory_metrics": {},   # From compute_group_rewards() metrics

        # Conversation (v3+, null for v1-v2)
        "conversation": [           # Aggregated from per-step _conversation
            {"role": "user", "content": "...", ...},
            {"role": "assistant", "content": [...], "tool_calls": [...], ...},
            ...
        ] | null,

        # Per-step details
        "steps": [
            {
                "step_idx": int,
                "ob_len": int,      # Observation tokens (prompt length)
                "ac_len": int,      # Action tokens (model response length)
                "reward": float,    # Immediate reward from env.step()
                "episode_done": bool,
                "metrics": {"correct": 1.0, "format": 1.0, ...},
                "logs": {           # Diagnostic data
                    "_conversation": [...],  # Messages for this step (framework key)
                    "answer_extraction": "42",  # User-defined diagnostic logs
                    ...
                }
            },
            ...
        ],

        "final_ob_len": int,        # Final observation length (context window usage)
        "status": "ok" | "error" | "timeout",
        "error_type": str | null,   # Exception class name if status="error"
        "error_message": str | null,
        "stop_reason": "stop" | "length" | null  # Why sampling stopped
    }

Schema history:
    - v1: Initial schema
    - v2: Added status, error_type, error_message, stop_reason
    - v3: Added conversation, model_name; _conversation in step logs
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.utils.deprecation import deprecated
from tinker_cookbook.utils.misc_utils import safezip

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutSummaryExportConfig:
    """Metadata for one rollout-summary JSONL export.

    Attributes:
        split: Dataset split name (e.g. ``"train"``, ``"eval/gsm8k"``).
        iteration: Training iteration / batch index.
        base_name: Prefix for the JSONL file (e.g. ``"train"``, ``"eval_gsm8k"``).
            Used by ``TrainingRunStore.write_rollouts()``.
        sampling_client_step: Step counter of the sampling client
            used to generate the rollouts, or ``None`` if not applicable.
    """

    split: str
    iteration: int
    base_name: str
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
        model_name (str | None): Model name for this training run.
    """

    trajectory_group: TrajectoryGroup
    tags: list[str]
    sampling_client_step: int | None = None
    model_name: str | None = None


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


def serialize_rollout_summaries(
    *,
    split: str,
    iteration: int,
    trajectory_groups_P: Sequence[TrajectoryGroup],
    taglist_P: Sequence[list[str]],
    sampling_client_steps_P: Sequence[int | None] | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Serialize trajectory groups into JSON-safe rollout summary records.

    Returns one record per trajectory, suitable for writing to JSONL via
    ``TrainingRunStore.write_rollouts()`` or ``write_rollout_summaries_jsonl()``.

    Each record contains::

        {
            "schema_version": 3,
            "split": str, "iteration": int, "group_idx": int, "traj_idx": int,
            "tags": list[str], "sampling_client_step": int | None,
            "total_reward": float, "final_reward": float,
            "trajectory_metrics": dict, "final_ob_len": int,
            "model_name": str | None,
            "conversation": list[dict] | None,
            "steps": [{"step_idx", "ob_len", "ac_len", "reward", "episode_done", "metrics", "logs"}, ...]
        }

    Args:
        split: Dataset split identifier (e.g. ``"train"``, ``"test"``).
        iteration: Training iteration / batch index.
        trajectory_groups_P: One trajectory group per problem (subscript ``_P``).
        taglist_P: Tags for each trajectory group, aligned with *trajectory_groups_P*.
        sampling_client_steps_P: Per-group sampling-client step counters, or ``None``.
        model_name: Model name for this training run, or ``None``.

    Returns:
        List of JSON-serializable rollout summary records, one per trajectory.
    """
    records: list[dict[str, Any]] = []
    for group_idx, (trajectory_group, tags) in enumerate(safezip(trajectory_groups_P, taglist_P)):
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

            # Determine trajectory status and stop reason
            last_transition = trajectory.transitions[-1] if trajectory.transitions else None
            stop_reason = (
                getattr(last_transition.ac, "stop_reason", None) if last_transition else None
            )

            # Check for errors matching this trajectory index
            traj_error = None
            for err in trajectory_group.rollout_errors:
                # RolloutError doesn't carry traj_idx, so we can only report
                # group-level errors. If there are errors, mark all trajectories.
                traj_error = err
                break

            if traj_error is not None:
                status = "error"
            elif stop_reason == "length":
                status = "timeout"
            else:
                status = "ok"

            # Aggregate conversation from per-step logs (populated by
            # ProblemEnv and EnvFromMessageEnv via the "_conversation" key).
            conversation: list[Any] = []
            for step in steps:
                step_conv = step.get("logs", {}).get("_conversation")
                if isinstance(step_conv, list):
                    conversation.extend(step_conv)

            records.append(
                _json_safe(
                    {
                        "schema_version": 3,
                        "split": split,
                        "iteration": iteration,
                        "group_idx": group_idx,
                        "traj_idx": traj_idx,
                        "tags": list(tags),
                        "sampling_client_step": sampling_step,
                        "model_name": model_name,
                        "total_reward": total_rewards_G[traj_idx],
                        "final_reward": trajectory_group.final_rewards_G[traj_idx],
                        "trajectory_metrics": trajectory_group.metrics_G[traj_idx],
                        "conversation": conversation or None,
                        "steps": steps,
                        "final_ob_len": trajectory.final_ob.length,
                        "status": status,
                        "error_type": traj_error.error_type if traj_error else None,
                        "error_message": traj_error.error_message if traj_error else None,
                        "stop_reason": stop_reason,
                    }
                )
            )
    return records


@deprecated(
    message="Use serialize_rollout_summaries() + store.write_rollouts() instead.",
    removal_version="0.4.0",
)
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

    .. deprecated::
        Prefer ``serialize_rollout_summaries()`` + ``store.write_rollouts()``.

    Args:
        path (str | Path): Destination file path.
        split (str): Dataset split identifier.
        iteration (int): Training iteration / batch index.
        trajectory_groups_P (Sequence[TrajectoryGroup]): One trajectory group per problem.
        taglist_P (Sequence[list[str]]): Tags for each trajectory group.
        sampling_client_steps_P (Sequence[int | None] | None): Per-group step counters.
    """
    records = serialize_rollout_summaries(
        split=split,
        iteration=iteration,
        trajectory_groups_P=trajectory_groups_P,
        taglist_P=taglist_P,
        sampling_client_steps_P=sampling_client_steps_P,
    )
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


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


def serialize_rollout_summaries_from_groups(
    *,
    split: str,
    iteration: int,
    groups_P: Sequence[RolloutSummaryGroup],
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Serialize pre-grouped rollout summaries into JSON-safe dicts.

    Convenience wrapper around :func:`serialize_rollout_summaries` that accepts
    :class:`RolloutSummaryGroup` objects.
    """
    # Per-group model_name takes precedence, fall back to the explicit kwarg.
    resolved_model_name = model_name
    if groups_P and groups_P[0].model_name is not None:
        resolved_model_name = groups_P[0].model_name
    return serialize_rollout_summaries(
        split=split,
        iteration=iteration,
        trajectory_groups_P=[group.trajectory_group for group in groups_P],
        taglist_P=[group.tags for group in groups_P],
        sampling_client_steps_P=[group.sampling_client_step for group in groups_P],
        model_name=resolved_model_name,
    )


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
