"""Core types for tool-use library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from tinker_cookbook.renderers.base import Message, ToolSpec


@dataclass
class ToolInput:
    """Input to a tool invocation."""

    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass
class ToolResult:
    """Result from a tool invocation."""

    messages: list[Message]
    should_stop: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Tool(Protocol):
    """Protocol for tools that can be used by LLM agents."""

    @property
    def name(self) -> str:
        """Tool name shown to the model."""
        ...

    @property
    def description(self) -> str:
        """Tool description shown to the model."""
        ...

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema for tool parameters shown to the model."""
        ...

    async def run(self, input: ToolInput) -> ToolResult:
        """Execute the tool with validated arguments. Returns a ToolResult."""
        ...

    def to_spec(self) -> ToolSpec:
        """Convert to ToolSpec for renderer integration."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }
