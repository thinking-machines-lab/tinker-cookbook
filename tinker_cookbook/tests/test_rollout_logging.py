import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tinker_cookbook.rl.rollout_logging import write_rollout_summaries_jsonl
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition


def test_write_rollout_summaries_jsonl_handles_numpy_scalars(tmp_path: Path):
    transition = Transition(
        ob=SimpleNamespace(length=np.int64(5)),
        ac=SimpleNamespace(tokens=[1, 2, 3]),
        reward=np.float32(0.25),
        episode_done=True,
        metrics={"score": np.float32(1.5)},
        logs={"rank": np.int64(2)},
    )
    trajectory = Trajectory(transitions=[transition], final_ob=SimpleNamespace(length=np.int64(8)))
    trajectory_group = TrajectoryGroup(
        trajectories_G=[trajectory],
        final_rewards_G=[np.float32(0.75)],
        metrics_G=[{"traj_metric": np.float32(3.0)}],
    )
    output_path = tmp_path / "rollouts.jsonl"

    write_rollout_summaries_jsonl(
        output_path,
        split="train",
        iteration=1,
        trajectory_groups_P=[trajectory_group],
        taglist_P=[["unit-test"]],
        sampling_client_steps_P=[7],
    )

    record = json.loads(output_path.read_text().strip())
    assert record["iteration"] == 1
    assert record["sampling_client_step"] == 7
    assert record["total_reward"] == 1.0
    assert record["final_reward"] == 0.75
    assert record["steps"][0]["reward"] == 0.25
    assert record["steps"][0]["metrics"]["score"] == 1.5
    assert record["steps"][0]["logs"]["rank"] == 2
