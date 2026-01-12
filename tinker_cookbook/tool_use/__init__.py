"""Tool-use infrastructure for LLM agents.

This module provides:
- ToolInterface: Abstract base class for tools
- @tool decorator: Create tools from functions/methods
- AgentToolMessageEnv: Environment for tool-using agents
"""

from tinker_cookbook.tool_use.llm_tools import (
    FunctionTool,
    ToolInterface,
    extract_tool_payload,
    handle_tool_call,
    tool,
)
from tinker_cookbook.tool_use.tool_env import AgentToolMessageEnv

__all__ = [
    "AgentToolMessageEnv",
    "FunctionTool",
    "ToolInterface",
    "extract_tool_payload",
    "handle_tool_call",
    "tool",
]
