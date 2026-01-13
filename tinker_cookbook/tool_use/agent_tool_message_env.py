"""Tool-using agent environment."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Callable

from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import Message, ToolCall
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.tool_use.tools import (
    ToolInterface,
    handle_tool_call,
)


@dataclass
class AgentToolMessageEnv(MessageEnv):
    """Generic tool-use MessageEnv for agents."""

    tools: list[ToolInterface]
    initial_messages: list[Message]
    max_turns: int
    reward_fn: Callable[[list[Message], Message], tuple[float, bool, dict[str, float]]]
    history: list[Message] = field(default_factory=list)

    _turn_count: int = 0
    _tool_dict: dict[str, ToolInterface] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._tool_dict = {t.name: t for t in self.tools}

    async def initial_observation(self) -> list[Message]:
        if not self.history:
            self.history = list(self.initial_messages)
        return self.history

    async def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> list[Message]:
        """Execute tool calls and append results to history."""
        results = await asyncio.gather(
            *[handle_tool_call(self._tool_dict, tc) for tc in tool_calls]
        )
        for msg in results:
            self.history.append(msg)
        return list(results)

    async def step(self, message: Message) -> MessageStepResult:
        """Execute any tools, update rewards, and return next messages.

        The episode ends when one of the following conditions is met:
        - No tool calls in message
        - reward_fn returns done=True
        - max_turns reached
        """
        self._turn_count += 1
        reward = 0.0
        metrics: dict[str, float] = {}

        # Append the message to history
        self.history.append(message)

        # Extract and execute tool calls if present
        tool_calls: list[ToolCall] = list(message.get("tool_calls") or [])
        results: list[Message] = []
        if tool_calls:
            results = await self._handle_tool_calls(tool_calls)

        # TODO: Update reward_fn to take the full history as input.
        reward_result = self.reward_fn(results, message)
        if inspect.iscoroutine(reward_result):
            reward_result = await reward_result
        reward_delta, done_from_reward, reward_metrics = reward_result
        reward += reward_delta
        metrics.update(reward_metrics)

        no_tool_calls = len(tool_calls) == 0
        done = no_tool_calls or done_from_reward

        if self._turn_count >= self.max_turns and not done:
            done = True
            metrics["max_turns"] = 1.0

        return MessageStepResult(
            reward=reward,
            episode_done=done,
            next_messages=self.history,
            metrics=metrics,
        )


def build_agent_tool_env(
    renderer: Renderer,
    tools: list[ToolInterface],
    initial_messages: list[Message],
    reward_fn: Callable[[list[Message], Message], tuple[float, bool, dict[str, float]]],
    *,
    max_turns: int = 5,
    failed_parse_reward: float = -0.1,
) -> EnvFromMessageEnv:
    """Convenience method to build an EnvFromMessageEnv for tool-using agents.

    Args:
        renderer: The renderer for tokenizing messages.
        tools: List of tools the agent can call.
        initial_messages: Initial conversation history (system prompt, user message, etc.).
        reward_fn: Function that grades each step. Takes (tool_results, assistant_message)
            and returns (reward, done, metrics).
        max_turns: Maximum turns before episode ends.
        failed_parse_reward: Reward when model output fails to parse.

    Returns:
        An EnvFromMessageEnv ready for RL training.
    """
    msg_env = AgentToolMessageEnv(
        tools=tools,
        initial_messages=initial_messages,
        max_turns=max_turns,
        reward_fn=reward_fn,
    )
    return EnvFromMessageEnv(
        renderer=renderer,
        message_env=msg_env,
        failed_parse_reward=failed_parse_reward,
    )
