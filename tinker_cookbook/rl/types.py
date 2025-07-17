"""
Basic interfaces and types for reinforcement learning.
"""

from dataclasses import dataclass, field
from typing import Sequence

import chz
from tinker_cookbook.completers import StopCondition, TokensWithLogprobs
from tinker_cookbook.utils.misc_utils import safezip
from tinker_public import types

type Action = list[int]
type Observation = types.ModelInput
type Logprobs = list[float]
Metrics = dict[str, float | int]


@dataclass
class StepResult:
    reward: float
    episode_done: bool
    next_observation: Observation
    next_stop_condition: StopCondition
    metrics: Metrics = field(default_factory=dict)


@dataclass
class Transition:
    ob: Observation
    ac: TokensWithLogprobs
    reward: float
    episode_done: bool
    metrics: Metrics = field(default_factory=dict)


class Env:
    """
    Stateful environment that a single agent interacts with.
    Discard after running for one episode.
    """

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        raise NotImplementedError

    async def step(self, action: Action) -> StepResult:
        raise NotImplementedError


@dataclass(frozen=True)
class Trajectory:
    """
    A sequence of observations and actions, resulting from running a single agent in a single
    environment.
    """

    transitions: list[Transition]
    final_ob: Observation


class EnvGroupBuilder:
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
    """

    async def make_envs(self) -> Sequence[Env]:
        raise NotImplementedError

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory]
    ) -> list[tuple[float, Metrics]]:
        """
        This computes a final reward for each trajectory that depends on the whole group.
        Note that there are also per-timestep rewards returned by the Env.step() method.
        The total reward is the sum of the per-timestep rewards plus the final group reward
        computed here. Defining a group reward is optional -- by default, the group reward
        is 0 and we only use the per-timestep rewards.
        """
        return [(0.0, {}) for _ in trajectory_group]


@dataclass
class TrajectoryGroup:
    """
    A group of trajectories, resulting from instantiating a group of environments using an
    EnvGroupBuilder, doing a rollout for each environment, and computing the rewards.
    """

    trajectories_G: list[Trajectory]
    final_rewards_G: list[float]  # computed by the EnvGroupBuilder, looking at whole group
    metrics_G: list[Metrics]

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


class RLDataset:
    """
    A dataset that produces batches of EnvGroups. This is the kind of dataset used by
    training algorithms.
    """

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


@chz.chz
class RLDatasetBuilder:
    """
    Abstract class for building RL datasets.
    """

    def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        """
        Return RLDataset (for training) and an optional RL dataset for testing
        """
        raise NotImplementedError
