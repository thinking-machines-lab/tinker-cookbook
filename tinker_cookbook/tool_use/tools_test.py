"""Tests for the arguments the @tool decorator passes to tool functions."""

import asyncio
from dataclasses import dataclass

from pydantic import BaseModel

from tinker_cookbook.tool_use import ToolInput, simple_tool_result, tool
from tinker_cookbook.tool_use.types import ToolResult


class Location(BaseModel):
    city: str


@dataclass
class Window:
    start: int
    end: int


@tool
async def describe(location: Location, windows: list[Window]) -> ToolResult:
    """Describe a location and the length of each window."""
    lengths = [window.end - window.start for window in windows]
    return simple_tool_result(f"{location.city}: {lengths}")


def test_tool_receives_validated_nested_arguments():
    result = asyncio.run(
        describe.run(
            ToolInput(
                arguments={"location": {"city": "Paris"}, "windows": [{"start": 3, "end": 10}]},
                call_id="call_1",
            )
        )
    )

    assert result.messages[0]["content"] == "Paris: [7]"
