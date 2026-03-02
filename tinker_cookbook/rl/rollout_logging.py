"""Utilities for exporting per-rollout records to JSONL."""

import json
from pathlib import Path
from typing import Any, Sequence

from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.utils.misc_utils import safezip


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
            pass
    return str(value)


def write_rollout_summaries_jsonl(
    path: str | Path,
    *,
    split: str,
    iteration: int,
    trajectory_groups_P: Sequence[TrajectoryGroup],
    taglist_P: Sequence[list[str]],
    sampling_client_steps_P: Sequence[int] | None = None,
) -> None:
    """
    Write one JSON record per rollout trajectory.

    This is intentionally disaggregated: no aggregate or summary statistics.
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
