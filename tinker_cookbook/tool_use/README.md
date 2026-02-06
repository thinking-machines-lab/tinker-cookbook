# Tool Use Library

> **Note:** This library is currently experimental and may change without warning.

A library for training tool-use agents with Tinker.

## Overview

The `tool_use` library provides:

- **`@tool` decorator** - Define tools from Python functions with automatic schema extraction
- **`Tool` protocol** - Interface for implementing custom tools
- **`AgentToolMessageEnv`** - RL environment for training tool-use agents

## Quick Example

```python
from tinker_cookbook.tool_use import tool, simple_tool_result, build_agent_tool_env

@tool
async def search(query: Annotated[str, "Search query"]) -> ToolResult:
    """Search for information."""
    results = await do_search(query)
    return simple_tool_result(json.dumps(results))

env = build_agent_tool_env(
    renderer=renderer,
    tools=[search],
    initial_messages=messages,
    reward_fn=my_reward_fn,
    max_turns=5,
)
```

## Stateful Tools

Stateful tools, including tools that share state, can be constructed by adding the `@tool` decorator to class methods with instance state:

```python
class MyTools:
    def __init__(self, api_key: str):
        self._api_key = api_key

    @tool
    async def search(self, query: Annotated[str, "Query"]) -> ToolResult:
        """Search using the configured API."""
        results = await search_api(query, self._api_key)
        return simple_tool_result(json.dumps(results))

    @tool
    async def lookup(self, id: Annotated[str, "Document ID"]) -> ToolResult:
        """Look up a document by ID."""
        result = await lookup_api(id, self._api_key)
        return simple_tool_result(json.dumps(result))

# Usage - both tools share the same api_key
tools_obj = MyTools(api_key="...")
env = build_agent_tool_env(..., tools=[tools_obj.search, tools_obj.lookup])
```

## Tool Lifetimes

The lifetime of an instantiated tool can be controlled by where it's instantiated:

| Instantiation Location | Lifetime |
|------------------------|----------|
| In environment construction | Per trajectory |
| In environment group construction | Per task (shared across trajectories) |
| In full dataset construction | Entire training run |

**Per-trajectory** (fresh state each rollout):
```python
# A fresh tool is instantiated for each Env
async def make_envs(self) -> Sequence[Env]:
    return [
        build_agent_tool_env(tools=[CodeTool(self.task).run])
        for _ in range(self.group_size)
    ]
```

**Shared across trajectories** (stateless or shared client):
```python
# A single tool is instantiated, and shared across Envs
def __init__(self, task, chroma_tool: ChromaTool):
    self.chroma_tool = chroma_tool  # Created once, reused

async def make_envs(self) -> Sequence[Env]:
    return [
        build_agent_tool_env(tools=[self.chroma_tool.search])  # Same instance
        for _ in range(self.group_size)
    ]
```

## Examples

For examples of using the tool-use library, see the the following:

- [code_rl recipe](../recipes/code_rl/) - Code generation with python execution tool
- [search_tool recipe](../recipes/search_tool/) - Multi-hop QA with search tool
