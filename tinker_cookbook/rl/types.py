"""
Basic interfaces and types for reinforcement learning.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import chz
import tinker
from typing_extensions import TypedDict

from tinker_cookbook.completers import StopCondition, TokensWithLogprobs
from tinker_cookbook.utils.misc_utils import safezip

Action: TypeAlias = list[int]
Observation: TypeAlias = tinker.ModelInput
Logprobs: TypeAlias = list[float]
Metrics: TypeAlias = dict[str, float | int]
Logs: TypeAlias = dict[str, str | int | float]


@dataclass
class StepResult:
    """Result returned by Env.step()."""

    reward: float
    """Immediate reward for this step."""
    episode_done: bool
    """Whether the episode has ended."""
    next_observation: Observation
    """Observation for the next step (or final observation if episode_done)."""
    next_stop_condition: StopCondition
    """Stop condition for the next generation."""
    metrics: Metrics = field(default_factory=dict)
    """Numeric values aggregated and reported in training logs (e.g., timing, counts)."""
    logs: Logs = field(default_factory=dict)
    """Diagnostic info for display/debugging tools (not aggregated like metrics)."""


@dataclass
class Transition:
    """A single (observation, action, reward) tuple from a trajectory."""

    ob: Observation
    """Observation the agent saw before taking the action."""
    ac: TokensWithLogprobs
    """Action taken (tokens and their log-probabilities)."""
    reward: float
    """Immediate reward received after taking the action."""
    episode_done: bool
    """Whether this transition ended the episode."""
    metrics: Metrics = field(default_factory=dict)
    """Numeric values aggregated and reported in training logs."""
    logs: Logs = field(default_factory=dict)
    """Diagnostic info for display/debugging tools (not aggregated like metrics)."""


class ActionExtra(TypedDict, total=False):
    """Extra metadata passed alongside an action to :meth:`Env.step`.

    All fields are optional so that callers and env implementations can
    ignore keys they don't care about.  Values must be picklable (the
    rollout executor may serialise them across process boundaries).
    """

    stop_reason: tinker.StopReason
    """Why sampling stopped — ``"stop"`` (hit a stop sequence) or ``"length"``
    (hit *max_tokens* without a stop sequence)."""


class Env(ABC):
    """
    Stateful environment that a single agent interacts with.
    Discard after running for one episode.
    """

    @abstractmethod
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        pass

    @abstractmethod
    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        pass


@dataclass(frozen=True)
class Trajectory:
    """
    A sequence of observations and actions, resulting from running a single agent in a single
    environment.
    """

    transitions: list[Transition]
    final_ob: Observation


@dataclass(frozen=True)
class RolloutError:
    """A captured error from a failed trajectory rollout.

    Stored on :class:`TrajectoryGroup` so error information flows through
    return values (including across process boundaries via pickle) without
    requiring shared mutable state.
    """

    error_type: str
    """The exception class name, e.g. ``'BadRequestError'``."""
    error_message: str
    """``str(exception)`` — the human-readable error description."""


class EnvGroupBuilder(ABC):
    """
    Builds a group of environments. The group will be used in the following way:

    - Algorithms like GRPO will center rewards across the group.
    - The reward function (compute_group_rewards) has access to the trajectories from the
      whole group, even though many reward functions will evaluate each one independently.

      - For example, this enables us to use pairwise reward models that look at a pair of
        trajectories at a time. With such a reward model, we effectively have a multi-agent
        environment, where the agents are playing a zero-sum game.

    Groups can be used in two ways, in practice:

    - To define a multi-agent environment
    - As a part of the *algorithm* (e.g. GRPO), when dealing with single-agent tasks.

    **Picklability:** Implementations must be pickleable (via standard ``pickle``) to support
    distributed rollout execution where builders are serialized and sent to remote workers.
    Avoid storing live network connections, file handles, or other unpickleable objects as
    fields. Use ``get_renderer()`` to create Renderers (which are automatically pickle-safe).
    Store configuration strings (model names, connection params) and construct heavy objects
    in ``make_envs()`` when possible. See ``HarborEnvGroupBuilder`` for a reference
    implementation of the lazy-construction pattern.
    """

    @abstractmethod
    async def make_envs(self) -> Sequence[Env]:
        pass

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """
        This computes a final reward for each trajectory that depends on the whole group.
        Note that there are also per-timestep rewards returned by the Env.step() method.
        The total reward is the sum of the per-timestep rewards plus the final group reward
        computed here. Defining a group reward is optional -- by default, the group reward
        is 0 and we only use the per-timestep rewards.
        """
        return [(0.0, {}) for _ in trajectory_group]

    async def cleanup(self) -> None:
        """Clean up resources created by make_envs().

        Called after rollouts and reward computation complete, regardless
        of success or failure. Override this to release expensive resources
        like cloud sandboxes, remote browsers, etc.

        Default is a no-op. Implementations should be idempotent (safe to
        call multiple times) and handle exceptions internally, as `do_group_rollout`
        does not catch exceptions from this method.
        """
        pass

    def logging_tags(self) -> list[str]:
        """
        This is just used for logging. We often want to aggregate metrics (like rewards
        or episode lengths) per-environment, or across a group of related environments.

        Most commonly, you'd return a short name for the environment, such as ['gsm'] for
        grade school math. You also might want a few tags at different levels of granularity,
        e.g., ['gsm', 'math', 'rlvr']
        """
        return []


@dataclass
class TrajectoryGroup:
    """
    A group of trajectories, resulting from instantiating a group of environments using an
    EnvGroupBuilder, doing a rollout for each environment, and computing the rewards.
    """

    trajectories_G: list[Trajectory]
    final_rewards_G: list[float]  # computed by the EnvGroupBuilder, looking at whole group
    metrics_G: list[Metrics]

    # Error tracking — populated by do_group_rollout when using error-tolerant strategies.
    # Empty list means no trajectory errors occurred.
    rollout_errors: list[RolloutError] = field(default_factory=list)

    def get_total_rewards(self) -> list[float]:
        """
        Get the total reward (i.e., the return) of each trajectory (episode) in the group.
        The total reward is the sum of the per-timestep rewards plus the final group reward
        computed by the EnvGroupBuilder.
        """
        return [
            sum(transition.reward for transition in trajectory.transitions) + final_reward
            for trajectory, final_reward in safezip(self.trajectories_G, self.final_rewards_G)
        ]


class RLDataset(ABC):
    """
    A dataset that produces batches of EnvGroups. This is the kind of dataset used by
    training algorithms.
    """

    @abstractmethod
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


@chz.chz
class RLDatasetBuilder:
    """
    Abstract class for building RL datasets.
    """

    @abstractmethod
    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        """
        Return RLDataset (for training) and an optional RL dataset for testing
        """
        pass
