"""Tool-using agent environment."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import (
    Message,
    ToolCall,
    format_content_as_string,
    get_text_content,
)
from tinker_cookbook.rl import types
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.tool_use.tools import handle_tool_call
from tinker_cookbook.tool_use.types import Tool, ToolResult

# Reward functions return (reward, metrics) or (reward, metrics, logs). The
# optional third element carries grading diagnostics (e.g. test-run details)
# into ``StepResult.logs`` for display/capture; it does not affect training.
RewardResult = tuple[float, dict[str, float]] | tuple[float, dict[str, float], types.Logs]
RewardFn = Callable[[list[Message]], Awaitable[RewardResult]]
# TODO(tyler): Consider supporting stateful tools that need to grade rollouts based on
# information not contained in the message history (e.g., internal tool state that changes
# during execution).


@dataclass
class AgentToolMessageEnv(MessageEnv):
    """Generic tool-use MessageEnv for agents."""

    tools: list[Tool]
    initial_messages: list[Message]
    max_turns: int
    reward_fn: RewardFn
    history: list[Message] = field(default_factory=list)

    _turn_count: int = 0
    _tool_dict: dict[str, Tool] = field(default_factory=dict, init=False)
    _should_stop: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._tool_dict = {t.name: t for t in self.tools}

    async def initial_observation(self) -> list[Message]:
        if not self.history:
            self.history = list(self.initial_messages)
        return self.history

    async def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls and append result messages to history.

        Returns the per-call :class:`ToolResult`\\ s, aligned 1:1 with
        *tool_calls*, so the caller can record structured per-call info
        (error type, should_stop) alongside the messages.
        """
        tool_results = await asyncio.gather(
            *[handle_tool_call(self._tool_dict, tc) for tc in tool_calls]
        )

        for tool_result in tool_results:
            # Append messages to history
            self.history.extend(tool_result.messages)

            # Check if any tool signals to stop
            if tool_result.should_stop:
                self._should_stop = True

        return list(tool_results)

    async def step(self, message: Message) -> MessageStepResult:
        """Execute any tools and return next messages.

        The episode ends when:
        - no tool calls in message (model decided to stop)
        - a tool returns should_stop=True
        - max_turns reached

        reward_fn is called once at episode end to grade the full trajectory.
        """
        self._turn_count += 1
        metrics: dict[str, float] = {}
        logs: types.Logs = {}

        # Append the message to history
        self.history.append(message)

        # Log assistant content (handles both str and multimodal content)
        assistant_text = get_text_content(message)
        if assistant_text:
            logs["assistant_content"] = assistant_text

        # Extract and execute tool calls if present
        tool_calls: list[ToolCall] = list(message.get("tool_calls") or [])
        tool_call_records: list[types.ToolCallRecord] = []
        if tool_calls:
            # Deprecated: positional tool_call_{i} log strings are kept for
            # display/JSONL compatibility; prefer the structured tool_calls
            # records below.
            for i, tc in enumerate(tool_calls):
                logs[f"tool_call_{i}"] = f"{tc.function.name}({tc.function.arguments})"

            tool_results = await self._handle_tool_calls(tool_calls)

            tool_result_messages = [msg for result in tool_results for msg in result.messages]
            for i, msg in enumerate(tool_result_messages):
                logs[f"tool_result_{i}"] = format_content_as_string(msg["content"])

            # Structured per-call records: error_type comes from the "error"
            # metadata key set by error_tool_result() (e.g. "tool_not_found",
            # "validation_failed", "execution_failed").
            for tc, result in zip(tool_calls, tool_results, strict=True):
                error = result.metadata.get("error")
                tool_call_records.append(
                    types.ToolCallRecord(
                        name=tc.function.name,
                        args_json=tc.function.arguments,
                        error_type=str(error) if error is not None else None,
                        should_stop=result.should_stop,
                    )
                )

        # Determine if episode is done
        no_tool_calls = len(tool_calls) == 0
        max_turns_reached = self._turn_count >= self.max_turns
        done = no_tool_calls or max_turns_reached or self._should_stop

        if max_turns_reached and not no_tool_calls:
            metrics["max_turns"] = 1.0
        if self._should_stop:
            metrics["tool_stopped"] = 1.0

        reward = 0.0
        if done:
            reward_result = await self.reward_fn(self.history)
            reward = reward_result[0]
            metrics.update(reward_result[1])
            if len(reward_result) == 3:
                logs.update(reward_result[2])

        return MessageStepResult(
            reward=reward,
            episode_done=done,
            next_messages=self.history,
            metrics=metrics,
            logs=logs,
            tool_calls=tool_call_records or None,
        )


def build_agent_tool_env(
    renderer: Renderer,
    tools: list[Tool],
    initial_messages: list[Message],
    reward_fn: RewardFn,
    *,
    max_turns: int = 5,
    failed_parse_reward: float = -0.1,
    max_trajectory_tokens: int | None = None,
    max_generation_tokens: int | None = None,
    context_overflow_reward: float = -0.1,
) -> EnvFromMessageEnv:
    """Convenience method to build an EnvFromMessageEnv for tool-using agents.

    Args:
        renderer: The renderer for tokenizing messages.
        tools: List of tools the agent can call (must implement Tool protocol).
        initial_messages: Initial conversation history (system prompt, user message, etc.).
        reward_fn: Function that grades a completed episode. Takes the full message
            history and returns (reward, metrics). Called once at episode end.
        max_turns: Maximum turns before episode ends.
        failed_parse_reward: Reward when model output fails to parse.
        max_trajectory_tokens: Maximum tokens in trajectory before terminating episode.
        max_generation_tokens: Maximum tokens per generation. When set, the episode
            terminates if the trajectory + generation budget would exceed
            *max_trajectory_tokens*, preventing context overflow errors.
        context_overflow_reward: Reward assigned when the episode is terminated due to
            context overflow. Defaults to -0.1.

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
        max_trajectory_tokens=max_trajectory_tokens,
        max_generation_tokens=max_generation_tokens,
        context_overflow_reward=context_overflow_reward,
    )
