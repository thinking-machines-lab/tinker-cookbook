from __future__ import annotations

from typing import Sequence

import chz
import verifiers as vf
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder


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
                    answer=row.get("answer", ""),
                    task=row.get("task", "default"),
                    info=row.get("info", {}),
                )
            )
        return builders


@chz.chz
class VerifiersRLDatasetBuilder(RLDatasetBuilder):
    vf_env: vf.Environment
    groups_per_batch: int
    dataset_n: int
    dataset_seed: int | None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        ds = self.vf_env.get_dataset(n=self.dataset_n, seed=self.dataset_seed)
        rows = [
            {
                "prompt": ds["prompt"][i],
                **({"answer": ds["answer"][i]} if "answer" in ds.column_names else {}),
                **({"task": ds["task"][i]} if "task" in ds.column_names else {}),
                **({"info": ds["info"][i]} if "info" in ds.column_names else {}),
            }
            for i in range(len(ds))
        ]
        return VerifiersRLDataset(rows, self.vf_env, self.groups_per_batch), None


class VerifiersEnvGroupBuilder(EnvGroupBuilder):
    def __init__(
        self,
        vf_env: vf.Environment,
        prompt: vf.Messages,
        answer: str,
        task: str,
        info: dict,
    ):
        self.vf_env = vf_env
        self.prompt = prompt
        self.answer = answer
        self.task = task
        self.info = info

    async def make_envs(self):
        return []  # unused when using custom_do_group_rollout

    def logging_tags(self):
        return [self.task] if self.task else []
