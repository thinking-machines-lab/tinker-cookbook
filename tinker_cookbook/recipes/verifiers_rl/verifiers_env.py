"""Native verifiers v1 dataset and rollout integration."""

from __future__ import annotations

import asyncio
import tomllib
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from pathlib import Path

import chz
import tinker
import verifiers.v1 as vf
from verifiers.v1.decorators import discover_decorated

from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.exceptions import AllTrajectoriesFailedError
from tinker_cookbook.recipes.verifiers_rl.tinker_client import TinkerClient
from tinker_cookbook.rl.types import (
    DirectEnvGroupBuilder,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    RolloutError,
    Trajectory,
    TrajectoryGroup,
    Transition,
)


class _VerifiersRuntime:
    def __init__(self, env: vf.Environment, max_concurrent: int) -> None:
        self.env = env
        self.semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        self._serving: AbstractAsyncContextManager[None] | None = None

    async def start(self) -> None:
        if self._serving is not None:
            return
        self._serving = self.env.serving()
        await self._serving.__aenter__()

    async def close(self) -> None:
        if self._serving is None:
            return
        serving = self._serving
        self._serving = None
        await serving.__aexit__(None, None, None)


def trace_to_trajectory(trace: vf.Trace) -> Trajectory:
    """Convert every v1 trace branch into Tinker's prefix/action representation."""
    transitions: list[Transition] = []
    trained_nodes: set[int] = set()

    for branch in trace.branches:
        prefix: list[int] = []
        for node in branch.nodes:
            if not node.sampled or not any(node.mask):
                prefix.extend(node.token_ids)
                continue
            first_sampled = node.mask.index(True)
            if any(not sampled for sampled in node.mask[first_sampled:]):
                raise ValueError("verifiers trace has a non-contiguous sampled token span")
            prompt_ids = prefix + node.token_ids[:first_sampled]
            completion_ids = node.token_ids[first_sampled:]
            if len(node.logprobs) != len(completion_ids):
                raise ValueError("verifiers trace completion tokens and logprobs are misaligned")
            transitions.append(
                Transition(
                    ob=tinker.ModelInput.from_ints(prompt_ids),
                    ac=TokensWithLogprobs(
                        tokens=completion_ids,
                        maybe_logprobs=node.logprobs,
                    ),
                    reward=0.0,
                    episode_done=False,
                    action_mask=0.0 if id(node) in trained_nodes else 1.0,
                )
            )
            trained_nodes.add(id(node))
            prefix.extend(node.token_ids)

    if transitions:
        transitions[-1].episode_done = True
    return Trajectory(
        transitions=transitions,
        final_ob=tinker.ModelInput.empty(),
        stop_reason=trace.stop_condition,
    )


def traces_to_trajectory_group(
    traces: list[vf.Trace], *, requires_group_scoring: bool = False
) -> TrajectoryGroup:
    trajectories: list[Trajectory] = []
    rewards: list[float] = []
    metrics: list[dict[str, float | int]] = []
    errors: list[RolloutError] = []

    if requires_group_scoring and any(trace.has_error for trace in traces):
        raise AllTrajectoriesFailedError(
            "verifiers group-scored episode contains a failed rollout"
        )

    for trace in traces:
        if trace.error is not None:
            errors.append(RolloutError(trace.error.type, trace.error.message))
            continue
        trajectory = trace_to_trajectory(trace)
        if not trajectory.transitions:
            errors.append(RolloutError("EmptyTraceError", "trace contains no sampled tokens"))
            continue
        trajectories.append(trajectory)
        rewards.append(trace.reward)
        metrics.append(dict(trace.metrics))

    if not trajectories:
        raise AllTrajectoriesFailedError("all verifiers rollouts failed or produced no tokens")
    return TrajectoryGroup(trajectories, rewards, metrics, rollout_errors=errors)


class VerifiersRLDataset(RLDataset):
    def __init__(
        self,
        tasks: list[vf.Task],
        env: vf.Environment,
        groups_per_batch: int,
        group_size: int,
        model_name: str,
        renderer_model_name: str,
        renderer_pool_size: int,
        max_concurrent: int,
    ) -> None:
        self.tasks = tasks
        self.runtime = _VerifiersRuntime(env, max_concurrent)
        self.groups_per_batch = groups_per_batch
        self.group_size = group_size
        self.model_name = model_name
        self.renderer_model_name = renderer_model_name
        self.renderer_pool_size = renderer_pool_size
        self.requires_group_scoring = bool(
            tasks and discover_decorated(tasks[0], "group_reward")
        )

    def __len__(self) -> int:
        return (len(self.tasks) + self.groups_per_batch - 1) // self.groups_per_batch

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.groups_per_batch
        return [
            VerifiersEnvGroupBuilder(
                task=task,
                runtime=self.runtime,
                group_size=self.group_size,
                model_name=self.model_name,
                renderer_model_name=self.renderer_model_name,
                renderer_pool_size=self.renderer_pool_size,
                requires_group_scoring=self.requires_group_scoring,
            )
            for task in self.tasks[start : start + self.groups_per_batch]
        ]

    async def start(self) -> None:
        await self.runtime.start()

    async def close(self) -> None:
        await self.runtime.close()


@chz.chz
class VerifiersRLDatasetBuilder(RLDatasetBuilder):
    env_config_path: str
    model_name: str
    renderer_model_name: str
    groups_per_batch: int = 32
    group_size: int = 8
    num_tasks: int | None = None
    renderer_pool_size: int = 16
    max_concurrent: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        raw_config = tomllib.loads(Path(self.env_config_path).read_text())
        config = vf.EnvConfig.model_validate(raw_config)
        env = vf.Environment(config)
        tasks = list(env.taskset.load())
        if self.num_tasks is not None:
            tasks = tasks[: self.num_tasks]
        dataset = VerifiersRLDataset(
            tasks=tasks,
            env=env,
            groups_per_batch=self.groups_per_batch,
            group_size=self.group_size,
            model_name=self.model_name,
            renderer_model_name=self.renderer_model_name,
            renderer_pool_size=self.renderer_pool_size,
            max_concurrent=self.max_concurrent,
        )
        return dataset, None


class VerifiersEnvGroupBuilder(DirectEnvGroupBuilder):
    def __init__(
        self,
        task: vf.Task,
        runtime: _VerifiersRuntime,
        group_size: int,
        model_name: str,
        renderer_model_name: str,
        renderer_pool_size: int,
        requires_group_scoring: bool,
    ) -> None:
        self.task = task
        self.runtime = runtime
        self.group_size = group_size
        self.model_name = model_name
        self.renderer_model_name = renderer_model_name
        self.renderer_pool_size = renderer_pool_size
        self.requires_group_scoring = requires_group_scoring

    async def rollout_group(self, policy: TokenCompleter) -> TrajectoryGroup:
        if not isinstance(policy, TinkerTokenCompleter):
            raise TypeError("verifiers v1 rollouts require TinkerTokenCompleter")
        client = TinkerClient(
            policy.sampling_client,
            renderer_model_name=self.renderer_model_name,
            renderer_pool_size=self.renderer_pool_size,
        )
        context = vf.ModelContext(
            model=self.model_name,
            client=client,
            sampling=vf.SamplingConfig(
                max_tokens=policy.max_tokens,
                temperature=policy.temperature,
            ),
        )
        traces = await self.runtime.env.episode(self.task, context, n=self.group_size).run(
            self.runtime.semaphore
        )
        return traces_to_trajectory_group(
            traces, requires_group_scoring=self.requires_group_scoring
        )

    def logging_tags(self) -> list[str]:
        return [type(self.task).__name__]
