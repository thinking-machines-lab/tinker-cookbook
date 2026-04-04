"""Basic interfaces and types for reinforcement learning.

This module defines the core abstractions for RL training in Tinker:

- **Env** and **EnvGroupBuilder**: the environment protocol that users implement.
- **Trajectory**, **Transition**, **TrajectoryGroup**: data containers that flow
  through the rollout and training pipeline.
- **RLDataset** and **RLDatasetBuilder**: dataset interfaces consumed by training
  loops.

Type aliases
------------
- ``Action`` – ``list[int]`` of token IDs produced by the agent.
- ``Observation`` – ``tinker.ModelInput`` fed to the model.
- ``Logprobs`` – ``list[float]`` of per-token log-probabilities.
- ``Metrics`` – ``dict[str, float | int]`` of numeric values aggregated in logs.
- ``Logs`` – ``dict[str, str | int | float]`` of diagnostic info for display.
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
    """Result returned by :meth:`Env.step`.

    Attributes:
        reward (float): Immediate reward for this step.
        episode_done (bool): Whether the episode has ended.
        next_observation (Observation): Observation for the next step
            (or final observation if ``episode_done`` is True).
        next_stop_condition (StopCondition): Stop condition for the next
            generation.
        metrics (Metrics): Numeric values aggregated and reported in training
            logs (e.g., timing, counts).
        logs (Logs): Diagnostic info for display/debugging tools (not
            aggregated like metrics).
    """

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
    teacher_observation: Observation | None = None
    """Optional teacher observation: the same conversation rendered with a different
    system prompt. When set, incorporate_kl_penalty uses this for computing teacher
    logprobs instead of the student's observation."""


@dataclass
class Transition:
    """A single (observation, action, reward) tuple from a trajectory.

    Attributes:
        ob (Observation): Observation the agent saw before taking the action.
        ac (TokensWithLogprobs): Action taken (tokens and their
            log-probabilities).
        reward (float): Immediate reward received after taking the action.
        episode_done (bool): Whether this transition ended the episode.
        metrics (Metrics): Numeric values aggregated and reported in training
            logs.
        logs (Logs): Diagnostic info for display/debugging tools (not
            aggregated like metrics).
    """

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
    teacher_ob: Observation | None = None
    """Optional teacher observation for the same conversation, rendered with a
    different system prompt. Used by incorporate_kl_penalty to compute teacher
    logprobs conditioned on the teacher's prompt instead of the student's."""


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
    """Stateful environment that a single agent interacts with.

    Each ``Env`` instance is **single-use**: create it, run one episode, then
    discard it. Environments are created by :meth:`EnvGroupBuilder.make_envs`.

    Implementors must override :meth:`initial_observation` and :meth:`step`.

    Example::

        class MyEnv(Env):
            def __init__(self, question: str, answer: str, renderer):
                self.question = question
                self.answer = answer
                self.renderer = renderer

            async def initial_observation(self):
                messages = [{"role": "user", "content": self.question}]
                model_input, _ = self.renderer.build_generation_prompt(messages)
                return model_input, self.renderer.get_stop_sequences()

            async def step(self, action, *, extra=None):
                response = self.renderer.tokenizer.decode(action)
                reward = 1.0 if self.answer in response else 0.0
                return StepResult(
                    reward=reward,
                    episode_done=True,
                    next_observation=tinker.ModelInput.from_ints([]),
                    next_stop_condition=[],
                )
    """

    @abstractmethod
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Return the starting observation and stop condition for this episode.

        Returns:
            tuple[Observation, StopCondition]: The initial observation (model input)
                and the stop condition for the first generation step.
        """
        pass

    @abstractmethod
    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        """Advance the environment by one step given the agent's action.

        Args:
            action (Action): Token IDs produced by the agent.
            extra (ActionExtra | None): Optional metadata about the action,
                such as the stop reason.

        Returns:
            StepResult: The reward, next observation, and whether the episode
                is done.
        """
        pass

    async def teacher_initial_observation(self) -> Observation | None:
        """Return the initial observation rendered with the teacher's system prompt.

        Override this in envs that support separate teacher/student system prompts.
        Returns None by default (teacher uses the same prompt as student).
        """
        return None


@dataclass(frozen=True)
class Trajectory:
    """A complete episode: a sequence of transitions from one agent in one environment.

    A trajectory is produced by running an :class:`Env` to completion.  It
    contains all transitions (observation-action-reward triples) plus the
    final observation after the last action.

    Attributes:
        transitions (list[Transition]): Ordered list of transitions in the
            episode.
        final_ob (Observation): The observation returned after the last
            action (i.e., the terminal state).
    """

    transitions: list[Transition]
    final_ob: Observation


@dataclass(frozen=True)
class RolloutError:
    """A captured error from a failed trajectory rollout.

    Stored on :class:`TrajectoryGroup` so error information flows through
    return values (including across process boundaries via pickle) without
    requiring shared mutable state.

    Attributes:
        error_type (str): The exception class name, e.g.
            ``'BadRequestError'``.
        error_message (str): ``str(exception)`` -- the human-readable error
            description.
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
        """Create the environments for this group.

        Returns:
            Sequence[Env]: The environments to run rollouts in.
        """
        pass

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """Compute a final reward for each trajectory that depends on the whole group.

        This is called after all rollouts in the group complete.  The total
        reward for each trajectory is the sum of the per-timestep rewards
        (from :meth:`Env.step`) plus the final group reward returned here.

        Override this when the reward depends on comparing trajectories within
        the group (e.g., pairwise reward models).  The default implementation
        returns ``(0.0, {})`` for every trajectory, so only per-timestep
        rewards are used.

        Args:
            trajectory_group (list[Trajectory]): The completed trajectories,
                one per environment in the group.
            env_group (Sequence[Env]): The corresponding environments (same
                order as ``trajectory_group``).

        Returns:
            list[tuple[float, Metrics]]: A list of ``(reward, metrics)`` pairs,
                one per trajectory.  The reward is added to the per-timestep
                total; the metrics dict is merged into training logs.
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
        """Return tags used to aggregate metrics in training logs.

        Tags let the training loop group metrics (rewards, episode lengths,
        etc.) by environment type.  Return a short list of names at different
        levels of granularity.

        Returns:
            list[str]: Tag strings for this environment group.  Default is
                an empty list.

        Example::

            def logging_tags(self) -> list[str]:
                return ["gsm", "math", "rlvr"]
        """
        return []


@dataclass
class TrajectoryGroup:
    """A group of trajectories produced by one :class:`EnvGroupBuilder`.

    Created by the rollout executor after running all environments in a group
    and computing group rewards.  This is the primary data structure consumed
    by RL training algorithms.

    The ``_G`` suffix follows the project convention for tensors/lists indexed
    over the group dimension.

    Attributes:
        trajectories_G (list[Trajectory]): One trajectory per environment in
            the group.
        final_rewards_G (list[float]): Group-level rewards computed by
            :meth:`EnvGroupBuilder.compute_group_rewards`, one per trajectory.
        metrics_G (list[Metrics]): Group-level metrics dicts, one per
            trajectory.
        rollout_errors (list[RolloutError]): Errors captured during rollout
            when using error-tolerant strategies.  Empty list means no errors.
    """

    trajectories_G: list[Trajectory]
    final_rewards_G: list[float]  # computed by the EnvGroupBuilder, looking at whole group
    metrics_G: list[Metrics]

    # Error tracking — populated by do_group_rollout when using error-tolerant strategies.
    # Empty list means no trajectory errors occurred.
    rollout_errors: list[RolloutError] = field(default_factory=list)

    def get_total_rewards(self) -> list[float]:
        """Get the total reward (return) for each trajectory in the group.

        The total reward is the sum of the per-timestep rewards (from
        :meth:`Env.step`) plus the final group reward (from
        :meth:`EnvGroupBuilder.compute_group_rewards`).

        Returns:
            list[float]: Total rewards, one per trajectory in the group.
        """
        return [
            sum(transition.reward for transition in trajectory.transitions) + final_reward
            for trajectory, final_reward in safezip(self.trajectories_G, self.final_rewards_G)
        ]


class RLDataset(ABC):
    """A dataset that produces batches of :class:`EnvGroupBuilder` instances.

    This is the primary dataset interface consumed by RL training loops.
    Implementations must define :meth:`get_batch` and :meth:`__len__`.

    Example::

        class MyDataset(RLDataset):
            def __init__(self, builders: list[EnvGroupBuilder], batch_size: int):
                self.builders = builders
                self.batch_size = batch_size

            def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
                start = index * self.batch_size
                return self.builders[start : start + self.batch_size]

            def __len__(self) -> int:
                return len(self.builders) // self.batch_size
    """

    @abstractmethod
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Return a batch of EnvGroupBuilder instances at the given index.

        Args:
            index (int): The batch index (``0 <= index < len(self)``).

        Returns:
            Sequence[EnvGroupBuilder]: The environment group builders for
                this batch.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of batches in the dataset.

        Returns:
            int: Total number of batches available.
        """
        pass


@chz.chz
class RLDatasetBuilder:
    """Abstract builder that constructs training and optional test RL datasets.

    Subclasses are ``chz`` dataclasses whose fields capture the configuration
    needed to build the datasets (e.g., data paths, group sizes, renderer
    names).  The builder is called once at the start of training.

    Implementations must be decorated with ``@chz.chz`` so they can be
    serialized and configured via the ``chz`` CLI.
    """

    @abstractmethod
    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        """Build the training dataset and an optional test dataset.

        Returns:
            tuple[RLDataset, RLDataset | None]: A two-element tuple where the
                first element is the training dataset and the second is an
                optional test dataset (``None`` if no test set is needed).
        """
        pass
