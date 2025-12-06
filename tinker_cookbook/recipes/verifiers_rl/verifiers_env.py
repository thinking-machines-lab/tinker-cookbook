from __future__ import annotations

from contextvars import ContextVar
from typing import Sequence

import chz
import tinker
import verifiers as vf

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)

_vf_env_ctx: ContextVar[vf.Environment | None] = ContextVar("vf_env", default=None)


def set_vf_env(env: vf.Environment) -> None:
    """Set the verifiers environment for the current context."""
    _vf_env_ctx.set(env)


def get_vf_env() -> vf.Environment | None:
    """Get the verifiers environment from the current context."""
    return _vf_env_ctx.get()


def convert_states_to_trajectory_group(states: list[vf.State]) -> TrajectoryGroup:
    """Convert verifiers States to tinker TrajectoryGroup."""
    trajectories_G: list[Trajectory] = []
    final_rewards_G: list[float] = []
    metrics_G: list[dict[str, float | int]] = []

    for state in states:
        transitions: list[Transition] = []
        trajectory_steps = state.get("trajectory", [])

        for i, step in enumerate(trajectory_steps):
            tokens_data = step.get("tokens")
            if tokens_data is not None:
                prompt_ids = tokens_data.get("prompt_ids", [])
                ob = tinker.ModelInput.from_ints(prompt_ids)
                completion_ids = tokens_data.get("completion_ids", [])
                completion_logprobs = tokens_data.get("completion_logprobs", [])
                ac = TokensWithLogprobs(
                    tokens=completion_ids,
                    maybe_logprobs=completion_logprobs,
                )
            else:
                ob = tinker.ModelInput.empty()
                ac = TokensWithLogprobs(tokens=[], maybe_logprobs=[])

            is_last = i == len(trajectory_steps) - 1
            transition = Transition(
                ob=ob,
                ac=ac,
                reward=0.0,
                episode_done=is_last,
                metrics={},
            )
            transitions.append(transition)

        trajectory = Trajectory(transitions=transitions, final_ob=tinker.ModelInput.empty())
        trajectories_G.append(trajectory)
        final_rewards_G.append(state.get("reward") or 0.0)
        metrics_G.append(state.get("metrics") or {})

    return TrajectoryGroup(
        trajectories_G=trajectories_G,
        final_rewards_G=final_rewards_G,
        metrics_G=metrics_G,
    )


class VerifiersRLDataset(RLDataset):
    def __init__(
        self,
        rows: list[dict],
        vf_env: vf.Environment,
        groups_per_batch: int,
    ):
        self.rows = rows
        self.vf_env = vf_env
        self.groups_per_batch = groups_per_batch

    def __len__(self) -> int:
        return (len(self.rows) + self.groups_per_batch - 1) // self.groups_per_batch

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.groups_per_batch
        end = min(len(self.rows), start + self.groups_per_batch)
        builders: list[EnvGroupBuilder] = []
        for j in range(start, end):
            row = self.rows[j]
            builders.append(
                VerifiersEnvGroupBuilder(
                    vf_env=self.vf_env,
                    prompt=row["prompt"],
                    example_id=row["example_id"],
                    task=row["task"],
                    answer=row.get("answer", ""),
                    info=row.get("info", {}),
                )
            )
        return builders


@chz.chz
class VerifiersRLDatasetBuilder(RLDatasetBuilder):
    vf_env_id: str
    vf_env_args: dict = chz.field(default_factory=dict)
    groups_per_batch: int = 32
    dataset_n: int = -1
    dataset_seed: int | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        vf_env = get_vf_env()
        if vf_env is None:
            vf_env = vf.load_environment(self.vf_env_id, **self.vf_env_args)
            set_vf_env(vf_env)
        ds = vf_env.get_dataset(n=self.dataset_n, seed=self.dataset_seed)
        rows = [
            {
                "prompt": ds["prompt"][i],
                "example_id": ds["example_id"][i],
                "task": ds["task"][i],
                **({"answer": ds["answer"][i]} if "answer" in ds.column_names else {}),
                **({"info": ds["info"][i]} if "info" in ds.column_names else {}),
            }
            for i in range(len(ds))
        ]
        return VerifiersRLDataset(rows, vf_env, self.groups_per_batch), None


class VerifiersEnvGroupBuilder(EnvGroupBuilder):
    def __init__(
        self,
        vf_env: vf.Environment,
        prompt: vf.Messages,
        example_id: int,
        task: str,
        answer: str = "",
        info: dict | None = None,
    ):
        self.vf_env = vf_env
        self.prompt = prompt
        self.example_id = example_id
        self.task = task
        self.answer = answer
        self.info = info or {}

    def get_rollout_inputs(self, group_size: int) -> list[vf.RolloutInput]:
        return [
            vf.RolloutInput(
                prompt=self.prompt,
                answer=self.answer,
                task=self.task,
                info=self.info,
                example_id=self.example_id,
            )
            for _ in range(group_size)
        ]

    async def make_envs(self):
        return []  # unused when using custom_do_group_rollout

    def logging_tags(self) -> list[str]:
        return [self.task] if self.task else []
