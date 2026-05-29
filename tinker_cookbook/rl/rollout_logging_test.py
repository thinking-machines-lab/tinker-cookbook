import json
from typing import cast

import numpy as np
import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.rollout_logging import serialize_rollout_summaries
from tinker_cookbook.rl.types import Logs, Metrics, Trajectory, TrajectoryGroup, Transition


def test_serialize_rollout_summaries_handles_numpy_scalars():
    transition = Transition(
        ob=tinker.ModelInput.from_ints([101, 102, 103, 104, 105]),
        ac=TokensWithLogprobs(tokens=[1, 2, 3], maybe_logprobs=[-0.1, -0.2, -0.3]),
        reward=cast(float, np.float32(0.25)),
        episode_done=True,
        metrics=cast(Metrics, {"score": np.float32(1.5)}),
        logs=cast(Logs, {"rank": np.int64(2)}),
    )
    trajectory = Trajectory(transitions=[transition], final_ob=tinker.ModelInput.from_ints([1] * 8))
    trajectory_group = TrajectoryGroup(
        trajectories_G=[trajectory],
        final_rewards_G=[cast(float, np.float32(0.75))],
        metrics_G=[cast(Metrics, {"traj_metric": np.float32(3.0)})],
    )
    records = serialize_rollout_summaries(
        split="train",
        iteration=1,
        trajectory_groups_P=[trajectory_group],
        taglist_P=[["unit-test"]],
        sampling_client_steps_P=[7],
    )

    record = json.loads(json.dumps(records[0]))
    assert record["iteration"] == 1
    assert record["sampling_client_step"] == 7
    assert record["total_reward"] == 1.0
    assert record["final_reward"] == 0.75
    assert record["steps"][0]["reward"] == 0.25
    assert record["steps"][0]["metrics"]["score"] == 1.5
    assert record["steps"][0]["logs"]["rank"] == 2


class _FakeTokenizer:
    def decode(self, tokens: list[int]) -> str:
        return " ".join(str(token) for token in tokens)


def test_serialize_rollout_summaries_can_include_decoded_text():
    transition = Transition(
        ob=tinker.ModelInput.from_ints([10, 20]),
        ac=TokensWithLogprobs(tokens=[30, 40], maybe_logprobs=[-0.1, -0.2]),
        reward=1.0,
        episode_done=True,
    )
    trajectory_group = TrajectoryGroup(
        trajectories_G=[
            Trajectory(transitions=[transition], final_ob=tinker.ModelInput.from_ints([]))
        ],
        final_rewards_G=[0.0],
        metrics_G=[{}],
    )

    records = serialize_rollout_summaries(
        split="train",
        iteration=3,
        trajectory_groups_P=[trajectory_group],
        taglist_P=[["unit-test"]],
        tokenizer=_FakeTokenizer(),
    )

    assert records[0]["steps"][0]["ob_text"] == "10 20"
    assert records[0]["steps"][0]["ac_text"] == "30 40"
    assert records[0]["steps"][0]["stop_reason"] == "stop"
