"""Message-level environment abstraction.

MessageEnv operates at the message level (list[Message]) rather than token level.

EnvFromMessageEnv bridges MessageEnv to the token-level Env interface used by
the RL training loop.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import tinker

from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import Message, message_to_jsonable
from tinker_cookbook.rl import types


@dataclass
class MessageStepResult:
    """Result of a message-level step."""

    reward: float
    episode_done: bool
    next_messages: list[Message]
    metrics: dict[str, float] = field(default_factory=dict)
    logs: types.Logs = field(default_factory=dict)
    next_stop_condition: StopCondition | None = None


class MessageEnv(ABC):
    """Abstract base class for message-level environments."""

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
        max_trajectory_tokens: int | None = None,
        max_generation_tokens: int | None = None,
        context_overflow_reward: float = -0.1,
    ):
        self.renderer = renderer
        self.message_env = message_env
        self.failed_parse_reward = failed_parse_reward
        self.terminate_on_parse_error = terminate_on_parse_error
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self.context_overflow_reward = context_overflow_reward
        self._base_stop_condition = renderer.get_stop_sequences()

        # Forward example_id from the inner MessageEnv for trajectory storage.
        # This ensures truncated examples (where MessageEnv.step() never runs)
        # still get the correct example_id in stored trajectories.
        self.example_id: str | None = getattr(message_env, "example_id", None)

    async def _render_in_thread(self, messages: list[Message], **kwargs) -> tinker.ModelInput:
        """Run build_generation_prompt in a thread to avoid blocking the event loop.

        Tokenization is CPU-bound. With many concurrent tasks on the same event
        loop, running it synchronously starves other coroutines. HuggingFace
        tokenizers release the GIL, so threads give true parallelism.
        """
        return await asyncio.to_thread(self.renderer.build_generation_prompt, messages, **kwargs)

    def _exceeds_context_limit(self, observation_length: int) -> bool:
        """Check if the observation + generation budget exceeds the context limit."""
        if self.max_trajectory_tokens is None:
            return False
        generation_reserve = self.max_generation_tokens or 0
        return observation_length + generation_reserve > self.max_trajectory_tokens

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        messages = await self.message_env.initial_observation()
        self._initial_messages: list[Message] | None = messages
        model_input = await self._render_in_thread(messages)

        if self._exceeds_context_limit(model_input.length):
            generation_reserve = self.max_generation_tokens or 0
            raise ValueError(
                f"Initial observation ({model_input.length} tokens) + "
                f"max_generation_tokens ({generation_reserve}) = "
                f"{model_input.length + generation_reserve} exceeds "
                f"max_trajectory_tokens ({self.max_trajectory_tokens}). "
                f"This task's prompt is too long for the model's context window."
            )

        return model_input, self._base_stop_condition

    def _build_step_conversation(self, *messages: Message | None) -> list[dict[str, object]]:
        """Build the _conversation list for this step's logs.

        On the first call, prepends the cached initial prompt messages.
        Subsequent calls include only the new messages from this step.
        """
        conv: list[dict[str, object]] = []
        initial = getattr(self, "_initial_messages", None)
        if initial is not None:
            conv.extend(message_to_jsonable(m) for m in initial)
            self._initial_messages = None
        for m in messages:
            if m is not None:
                conv.append(message_to_jsonable(m))
        return conv

    async def step(
        self, action: types.Action, *, extra: types.ActionExtra | None = None
    ) -> types.StepResult:
        """Parse tokens to a message, delegate to MessageEnv, and render response."""
        # If the model hit max_tokens without producing a stop sequence, terminate
        # the episode early. Previous turns' logprobs are preserved in the trajectory.
        stop_reason = (extra or {}).get("stop_reason", "stop")
        if stop_reason == "length":
            return types.StepResult(
                reward=self.context_overflow_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self._base_stop_condition,
                metrics={"max_tokens_reached": 1.0},
                logs={"_conversation": self._build_step_conversation()},
            )

        assistant_message, parse_success = self.renderer.parse_response(action)

        if not parse_success:
            return types.StepResult(
                reward=self.failed_parse_reward,
                episode_done=self.terminate_on_parse_error,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self._base_stop_condition,
                metrics={"parse_error": 1.0},
                logs={"_conversation": self._build_step_conversation(assistant_message)},
            )

        msg_step = await self.message_env.step(assistant_message)
        next_observation = await self._render_in_thread(msg_step.next_messages)
        next_stop_condition = msg_step.next_stop_condition or self._base_stop_condition

        # Build conversation for this step: assistant message + any new env
        # response messages (tool results, environment observations, etc.).
        conv = self._build_step_conversation(assistant_message)
        # msg_step.next_messages is the full conversation history. New env
        # response messages (tool results, etc.) appear after the last
        # assistant message. Capture them for the conversation log.
        if msg_step.next_messages:
            last_assistant_idx = -1
            for i in range(len(msg_step.next_messages) - 1, -1, -1):
                if msg_step.next_messages[i]["role"] == "assistant":
                    last_assistant_idx = i
                    break
            if last_assistant_idx >= 0:
                for m in msg_step.next_messages[last_assistant_idx + 1 :]:
                    conv.append(message_to_jsonable(m))

        step_logs = {**msg_step.logs, "_conversation": conv}

        # Check if the full trajectory + generation budget fits in the context window.
        # next_observation is the entire rendered conversation so far, which becomes
        # the prompt for the next sampling call. Only check when the episode continues —
        # if episode_done, there is no next sampling call and the real reward should be kept.
        if not msg_step.episode_done and self._exceeds_context_limit(next_observation.length):
            return types.StepResult(
                reward=self.context_overflow_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self._base_stop_condition,
                metrics={**msg_step.metrics, "context_overflow": 1.0},
                logs=step_logs,
            )

        return types.StepResult(
            reward=msg_step.reward,
            episode_done=msg_step.episode_done,
            next_observation=next_observation,
            next_stop_condition=next_stop_condition,
            metrics=msg_step.metrics,
            logs=step_logs,
        )
