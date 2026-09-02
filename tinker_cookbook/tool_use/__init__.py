"""Tool-use library."""

from tinker_cookbook.tool_use.agent_tool_message_env import (
    AgentToolMessageEnv,
    build_agent_tool_env,
)
from tinker_cookbook.tool_use.openai_compat import (
    linearize_tool_history_for_text_chat,
    normalize_xml_tool_call_message,
    parse_xml_tool_calls,
    strip_xml_thinking,
)
from tinker_cookbook.tool_use.tools import (
    FunctionTool,
    error_tool_result,
    handle_tool_call,
    simple_tool_result,
    tool,
)
from tinker_cookbook.tool_use.types import (
    Tool,
    ToolInput,
    ToolResult,
    ToolSpec,
)

__all__ = [
    "AgentToolMessageEnv",
    "build_agent_tool_env",
    "FunctionTool",
    "Tool",
    "ToolInput",
    "ToolResult",
    "ToolSpec",
    "error_tool_result",
    "handle_tool_call",
    "linearize_tool_history_for_text_chat",
    "normalize_xml_tool_call_message",
    "parse_xml_tool_calls",
    "simple_tool_result",
    "strip_xml_thinking",
    "tool",
]
