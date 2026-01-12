"""Message-level environment abstraction.

MessageEnv operates at the message level (list[Message]) rather than token level,
making it easier to implement environments for tool-using agents.

EnvFromMessageEnv bridges MessageEnv to the token-level Env interface used by
the RL training loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import tinker

from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl import types


@dataclass
class MessageStepResult:
    """Result of a message-level step: reward, done flag, next messages, metrics."""

    reward: float
    episode_done: bool
    next_messages: list[Message]
    metrics: dict[str, float] = field(default_factory=dict)
    next_stop_condition: StopCondition | None = None


class MessageEnv(ABC):
    """Abstract base class for message-level environments.

    Unlike the token-level Env, MessageEnv works with Message dicts directly,
    making it easier to implement environments for conversational agents.
    """

    @abstractmethod
    async def initial_observation(self) -> list[Message]:
        """Return the initial conversation history as renderer messages."""
        ...

    @abstractmethod
    async def step(self, message: Message) -> MessageStepResult:
        """Process an assistant message and return reward/next state."""
        ...


class EnvFromMessageEnv(types.Env):
    """Adapter that wraps a MessageEnv to implement the token-level Env interface.

    This bridges the message-level abstraction to the token-level interface
    expected by the RL training loop.
    """

    def __init__(
        self,
        renderer: Renderer,
        message_env: MessageEnv,
        failed_parse_reward: float = -1.0,
        terminate_on_parse_error: bool = True,
    ):
        self.renderer = renderer
        self.message_env = message_env
        self.failed_parse_reward = failed_parse_reward
        self.terminate_on_parse_error = terminate_on_parse_error
        self._base_stop_condition = renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        """Render initial messages into ModelInput plus stop condition."""
        messages = await self.message_env.initial_observation()
        return self.renderer.build_generation_prompt(messages), self._base_stop_condition

    async def step(self, action: types.Action) -> types.StepResult:
        """Parse tokens to a message, delegate to MessageEnv, and render response."""
        assistant_message, parse_success = self.renderer.parse_response(action)

        if not parse_success:
            return types.StepResult(
                reward=self.failed_parse_reward,
                episode_done=self.terminate_on_parse_error,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self._base_stop_condition,
                metrics={"parse_error": 1.0},
            )

        msg_step = await self.message_env.step(assistant_message)
        next_observation = self.renderer.build_generation_prompt(msg_step.next_messages)
        next_stop_condition = msg_step.next_stop_condition or self._base_stop_condition

        return types.StepResult(
            reward=msg_step.reward,
            episode_done=msg_step.episode_done,
            next_observation=next_observation,
            next_stop_condition=next_stop_condition,
            metrics=msg_step.metrics,
        )
