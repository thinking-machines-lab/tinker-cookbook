"""Tests for AgentToolMessageEnv.tool_execution (sequential / parallel /
concurrent_safe)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tinker_cookbook.renderers.base import Message, ToolCall, ToolSpec
from tinker_cookbook.tool_use.agent_tool_message_env import AgentToolMessageEnv
from tinker_cookbook.tool_use.tools import simple_tool_result
from tinker_cookbook.tool_use.types import ToolInput, ToolResult


class _EventLogTool:
    """Records start/end events into a shared log; the first invocation
    sleeps so that concurrent execution demonstrably interleaves."""

    def __init__(self, name: str, events: list[str], delay_seconds: float = 0.0):
        self._name = name
        self._events = events
        self._delay_seconds = delay_seconds

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"{self._name} tool"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def to_spec(self) -> ToolSpec:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    async def run(self, input: ToolInput) -> ToolResult:
        self._events.append(f"{self._name}:start")
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)
        self._events.append(f"{self._name}:end")
        return simple_tool_result("done", call_id=input.call_id or "", name=self._name)


def _tool_call(name: str, call_id: str) -> ToolCall:
    return ToolCall(id=call_id, function=ToolCall.FunctionBody(name=name, arguments="{}"))


def _two_call_message() -> Message:
    return {
        "role": "assistant",
        "content": "calling two tools",
        "tool_calls": [_tool_call("slow", "call_1"), _tool_call("fast", "call_2")],
    }


async def _reward(history: list[Message]) -> tuple[float, dict[str, float]]:
    return 0.0, {}


def _run_one_turn(tool_execution: Any, events: list[str]) -> AgentToolMessageEnv:
    env = AgentToolMessageEnv(
        tools=[
            _EventLogTool("slow", events, delay_seconds=0.05),
            _EventLogTool("fast", events),
        ],
        initial_messages=[{"role": "user", "content": "hi"}],
        max_turns=5,
        reward_fn=_reward,
        tool_execution=tool_execution,
    )

    async def run() -> None:
        await env.initial_observation()
        await env.step(_two_call_message())

    asyncio.run(run())
    return env


class TestSequentialExecution:
    def test_calls_run_in_request_order_without_overlap(self):
        """Sequential: the first (slow) tool fully finishes before the second
        starts, even though the second would complete first under gather."""
        events: list[str] = []
        _run_one_turn("sequential", events)
        assert events == ["slow:start", "slow:end", "fast:start", "fast:end"]

    def test_result_messages_in_request_order(self):
        events: list[str] = []
        env = _run_one_turn("sequential", events)
        tool_messages = [m for m in env.history if m["role"] == "tool"]
        assert [m.get("tool_call_id") for m in tool_messages] == ["call_1", "call_2"]


class TestParallelExecution:
    def test_default_is_parallel(self):
        env = AgentToolMessageEnv(
            tools=[],
            initial_messages=[],
            max_turns=5,
            reward_fn=_reward,
        )
        assert env.tool_execution == "parallel"

    def test_calls_overlap(self):
        """Parallel (asyncio.gather): both tools start before the slow one
        finishes — the fast tool completes while the slow one is sleeping."""
        events: list[str] = []
        _run_one_turn("parallel", events)
        assert events == ["slow:start", "fast:start", "fast:end", "slow:end"]

    def test_result_messages_still_in_request_order(self):
        """gather preserves result order, so history order matches the
        request order regardless of completion order."""
        events: list[str] = []
        env = _run_one_turn("parallel", events)
        tool_messages = [m for m in env.history if m["role"] == "tool"]
        assert [m.get("tool_call_id") for m in tool_messages] == ["call_1", "call_2"]


class TestConcurrentSafe:
    def test_concurrent_safe_raises_not_implemented(self):
        """Reserved mode: the Tool protocol has no concurrency-safety marker
        yet, so there is nothing to key the safe/unsafe split off."""
        with pytest.raises(NotImplementedError, match="concurrent_safe"):
            AgentToolMessageEnv(
                tools=[],
                initial_messages=[],
                max_turns=5,
                reward_fn=_reward,
                tool_execution="concurrent_safe",
            )
