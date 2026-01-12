"""Tool-using agent environment.

AgentToolMessageEnv combines the MessageEnv abstraction with tool execution,
providing a generic environment for tool-using agents.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable

from tinker_cookbook.renderers.base import Message, ToolCall
from tinker_cookbook.rl.message_env import MessageEnv, MessageStepResult
from tinker_cookbook.tool_use.llm_tools import (
    ToolInterface,
    handle_tool_call,
)


@dataclass
class AgentToolMessageEnv(MessageEnv):
    """Generic MessageEnv for tool-using agents.

    Episode ends when the model emits no tool calls or when max_turns is reached.
    """

    tools: list[ToolInterface]
    initial_messages: list[Message]
    max_turns: int
    reward_fn: Callable[[list[Message], Message], tuple[float, bool, dict[str, float]]]
    history: list[Message] = field(default_factory=list)
    turns: int = 0
    _tool_dict: dict[str, ToolInterface] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._tool_dict = {t.name: t for t in self.tools}

    async def initial_observation(self) -> list[Message]:
        """Return initial messages."""
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
        """Apply an assistant message, execute any tools, update rewards, and return next messages.

        Termination follows chat_rollouts pattern that the episode ends when:
        - No tool calls in message
        - reward_fn returns done=True
        - max_turns reached
        """
        self.turns += 1
        reward = 0.0
        metrics: dict[str, float] = {}

        # Append the message to history
        self.history.append(message)

        # Extract and execute tool calls if present
        tool_calls: list[ToolCall] = list(message.get("tool_calls") or [])
        results: list[Message] = []
        if tool_calls:
            results = await self._handle_tool_calls(tool_calls)

        # TODO: should probably give reward_fn the full history.
        reward_delta, done_from_reward, reward_metrics = self.reward_fn(results, message)
        reward += reward_delta
        metrics.update(reward_metrics)

        no_tool_calls = len(tool_calls) == 0
        done = no_tool_calls or done_from_reward

        if self.turns >= self.max_turns and not done:
            done = True
            metrics["max_turns"] = 1.0

        return MessageStepResult(
            reward=reward,
            episode_done=done,
            next_messages=self.history,
            metrics=metrics,
        )
