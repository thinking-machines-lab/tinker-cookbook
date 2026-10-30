"""Bridge between the verifiers v1 stack and the tinker RL data model.

An environment is a ``vf.Env`` (taskset + agents + episode shape). One tinker
group = one ``vf.Task`` rolled out ``group_size`` times, each rollout a
``vf.Episode`` whose single agent trace carries the sampled token IDs and
logprobs on its message graph. ``convert_episodes_to_trajectory_group`` turns
those into a ``TrajectoryGroup``.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextvars import ContextVar

import chz
import tinker
import verifiers.v1 as vf

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)

_vf_env_ctx: ContextVar[vf.Env | None] = ContextVar("vf_env", default=None)


def set_vf_env(env: vf.Env) -> None:
    """Set the verifiers environment for the current context."""
    _vf_env_ctx.set(env)


def get_vf_env() -> vf.Env | None:
    """Get the verifiers environment from the current context."""
    return _vf_env_ctx.get()


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_vf_env(vf_env_id: str, vf_env_args: dict | None = None) -> vf.Env:
    """Load a verifiers environment for the installed taskset ``vf_env_id``.

    ``vf_env_args`` is raw ``vf.EnvConfig`` data merged over the taskset id —
    e.g. ``{"agent": {"harness": {"id": "null"}}}`` to run a plain chat agent,
    or ``{"taskset": {"task": {...}}}`` for per-task config.
    """
    raw = _deep_merge({"taskset": {"id": vf_env_id}}, vf_env_args or {})
    return vf.load_environment(vf.resolve_env_config(raw))


def _trace_metrics(trace: vf.Trace, failed: bool = False) -> Metrics:
    metrics: Metrics = {
        name: float(value) for name, value in trace.metrics.items() if value is not None
    }
    if failed:
        metrics["rollout_failed"] = 1.0
    return metrics


def _episode_to_trajectory(episode: vf.Episode) -> tuple[Trajectory, float, Metrics]:
    """One episode -> (trajectory, final reward, metrics).

    A failed episode (not ``ok``, no trace, a trace that is not ``ok``, or a
    trace with no sampled tokens) becomes an empty trajectory with a one-hot
    ``rollout_failed`` metric: it contributes no training tokens but stays in
    the group, so reward centering sees the failure instead of the group
    silently shrinking. Standing this recipe cannot train (a non-trainable
    agent, or a trace whose only branches are non-trainable) is rejected rather
    than skipped, for the same reason; a non-trainable branch beside the real
    path is excluded, as verifiers intends.
    """
    failed = Trajectory(
        transitions=[
            Transition(
                ob=tinker.ModelInput.empty(),
                ac=TokensWithLogprobs(tokens=[], maybe_logprobs=[]),
                reward=0.0,
                episode_done=True,
            )
        ],
        final_ob=tinker.ModelInput.empty(),
    )
    if not episode.traces:
        return failed, 0.0, {"rollout_failed": 1.0}
    if len(episode.traces) > 1:
        raise ValueError(
            f"episode carries {len(episode.traces)} agent traces; this recipe trains "
            "single-agent verifiers environments (one trace per episode)"
        )
    trace = episode.traces[0]
    if not trace.agent.trainable:
        raise ValueError(
            f"trace from agent {trace.agent.name!r} is not trainable; this recipe trains "
            "single-agent verifiers environments whose one agent is the policy"
        )
    if not episode.ok or not trace.ok:
        return failed, trace.reward, _trace_metrics(trace, failed=True)
    # verifiers marks a rejected compaction attempt (a leaf the harness never
    # resumed from) as a non-trainable branch since #2521 (after 0.3.1); earlier
    # releases have no compaction, so every branch there is trainable.
    branches = [b for b in trace.branches if getattr(b, "trainable", True)]
    if len(branches) > 1:
        raise ValueError(
            f"trace has {len(branches)} trainable branches; this recipe trains linear "
            "rollouts (one root-to-leaf path per trace)"
        )
    if not branches:
        if trace.branches:
            raise ValueError(
                "trace's only branches are compaction attempts the harness never resumed "
                "from; verifiers marks them non-trainable and this recipe cannot train them"
            )
        return failed, trace.reward, _trace_metrics(trace, failed=True)
    branch = branches[0]

    # Walk the branch: every sampled node is one action, and its observation is
    # the full token context before it (previous nodes plus the node's own
    # unsampled generation-prompt scaffold). Observations therefore extend each
    # other by prefix, which trajectory_to_data merges into a single datum.
    spans: list[tuple[list[int], list[int], list[float]]] = []
    context: list[int] = []
    for node in branch.nodes:
        num_sampled = sum(node.mask)
        if not node.sampled or num_sampled == 0:
            context.extend(node.token_ids)
            continue
        if not all(node.mask[len(node.mask) - num_sampled :]):
            raise ValueError(
                "sampled tokens are not a suffix of the assistant node; cannot map "
                "this trace onto (observation, action) transitions"
            )
        if len(node.logprobs) != num_sampled:
            raise ValueError(
                f"assistant node carries {len(node.logprobs)} logprobs for {num_sampled} "
                "sampled tokens; the generate endpoint must return one logprob per token"
            )
        scaffold = node.token_ids[: len(node.token_ids) - num_sampled]
        sampled = node.token_ids[len(node.token_ids) - num_sampled :]
        spans.append((context + scaffold, sampled, list(node.logprobs)))
        context.extend(node.token_ids)

    if not spans:
        return failed, trace.reward, _trace_metrics(trace, failed=True)
    transitions = [
        Transition(
            ob=tinker.ModelInput.from_ints(observation),
            ac=TokensWithLogprobs(tokens=action, maybe_logprobs=logprobs),
            reward=0.0,
            episode_done=index == len(spans) - 1,
        )
        for index, (observation, action, logprobs) in enumerate(spans)
    ]
    trajectory = Trajectory(transitions=transitions, final_ob=tinker.ModelInput.from_ints(context))
    return trajectory, trace.reward, _trace_metrics(trace)


def convert_episodes_to_trajectory_group(episodes: Sequence[vf.Episode]) -> TrajectoryGroup:
    """Convert one group's episodes into a tinker TrajectoryGroup."""
    trajectories_G: list[Trajectory] = []
    final_rewards_G: list[float] = []
    metrics_G: list[Metrics] = []
    for episode in episodes:
        trajectory, reward, metrics = _episode_to_trajectory(episode)
        trajectories_G.append(trajectory)
        final_rewards_G.append(reward)
        metrics_G.append(metrics)
    return TrajectoryGroup(
        trajectories_G=trajectories_G,
        final_rewards_G=final_rewards_G,
        metrics_G=metrics_G,
    )


class VerifiersRLDataset(RLDataset):
    def __init__(self, tasks: list[vf.Task], vf_env: vf.Env, groups_per_batch: int):
        self.tasks = tasks
        self.vf_env = vf_env
        self.groups_per_batch = groups_per_batch

    def __len__(self) -> int:
        return (len(self.tasks) + self.groups_per_batch - 1) // self.groups_per_batch

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.groups_per_batch
        end = min(len(self.tasks), start + self.groups_per_batch)
        return [
            VerifiersEnvGroupBuilder(vf_env=self.vf_env, task=self.tasks[j])
            for j in range(start, end)
        ]


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
            vf_env = load_vf_env(self.vf_env_id, self.vf_env_args)
            set_vf_env(vf_env)
        taskset = vf_env.taskset
        if self.dataset_n >= 0:
            taskset = taskset.head(self.dataset_n)
        elif taskset.INFINITE:
            raise ValueError(f"taskset {self.vf_env_id!r} is infinite; bound it with dataset_n")
        if self.dataset_seed is not None:
            taskset = taskset.shuffle(self.dataset_seed)
        tasks = list(taskset)
        return VerifiersRLDataset(tasks, vf_env, self.groups_per_batch), None


class VerifiersEnvGroupBuilder(EnvGroupBuilder):
    """EnvGroupBuilder for the verifiers integration: one task, rolled out
    ``group_size`` times by ``train.py``'s group rollout override.

    Pickle support: ``vf.Env`` holds live serving resources and is not
    pickleable. On deserialization it is recovered from the ``_vf_env_ctx``
    context variable (set via ``set_vf_env()``). Raises ``RuntimeError`` if the
    context variable is not set — expected in cross-process scenarios, since
    this integration requires single-process execution (the group rollout
    override in train.py is a closure over shared state).
    """

    def __init__(self, vf_env: vf.Env, task: vf.Task):
        self.vf_env = vf_env
        self.task = task

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["vf_env"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        vf_env = state.pop("vf_env", None) or get_vf_env()
        if vf_env is None:
            raise RuntimeError(
                "VerifiersEnvGroupBuilder unpickled without a vf.Env. In cross-process "
                "scenarios (ProcessPoolExecutor, Ray), the worker process must call "
                "set_vf_env(load_vf_env(...)) before unpickling builders. See "
                "verifiers_rl/train.py for reference."
            )
        self.vf_env = vf_env
        self.task = state["task"]

    async def make_envs(self):
        return []  # unused: train.py overrides the group rollout wholesale

    def logging_tags(self) -> list[str]:
        name = self.task.data.name
        return [name] if name else []
