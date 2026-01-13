"""Tool-use infrastructure for LLM agents.

This module provides:
- ToolInterface: Abstract base class for tools
- @tool decorator: Create tools from functions/methods
- AgentToolMessageEnv: Environment for tool-using agents
"""

from tinker_cookbook.tool_use.tools import (
    FunctionTool,
    ToolInterface,
    extract_tool_payload,
    handle_tool_call,
    tool,
)
from tinker_cookbook.tool_use.agent_tool_message_env import (
    AgentToolMessageEnv,
    build_agent_tool_env,
)

__all__ = [
    "AgentToolMessageEnv",
    "build_agent_tool_env",
    "FunctionTool",
    "ToolInterface",
    "extract_tool_payload",
    "handle_tool_call",
    "tool",
]
