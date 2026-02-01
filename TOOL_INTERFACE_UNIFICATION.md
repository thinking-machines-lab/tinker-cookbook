# Tool Interface Unification

## Summary

We've unified the tool interface between tinker-cookbook and projects/worm to make it easy to contribute tools from worm back to the cookbook.

## What Changed

### Adopted Worm's Protocol

The cookbook now uses the same `Tool` protocol as worm:

```python
@runtime_checkable
class Tool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters_schema(self) -> dict[str, Any]: ...

    async def run(self, input: ToolInput) -> ToolResult: ...

    def to_spec(self) -> ToolSpec: ...
```

### Key Types

**ToolInput**: Arguments passed to the tool
```python
@dataclass
class ToolInput:
    arguments: dict[str, Any]
    call_id: str | None = None
```

**ToolResult**: Rich return type with messages, metrics, and metadata
```python
@dataclass
class ToolResult:
    messages: list[Message]
    should_stop: bool = False  # Tool can signal early termination
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Method Name Change

- **Old**: `async def invoke(arguments: dict) -> str`
- **New**: `async def run(input: ToolInput) -> ToolResult`

This matches worm's interface exactly.

## Usage

### Option 1: @tool Decorator (Recommended for Simple Tools)

The decorator automatically wraps your function to implement the Tool protocol:

```python
from tinker_cookbook.tool_use import tool

@tool
async def search(query: str) -> str:
    """Search for information."""
    results = await do_search(query)
    return json.dumps({"results": results})
```

**What the decorator does:**
- Extracts schema from type hints
- Validates arguments with Pydantic
- Wraps return value in ToolResult
- Handles errors gracefully

### Option 2: Manual Protocol Implementation (For Complex Tools)

Implement the Tool protocol directly for full control:

```python
from tinker_cookbook.tool_use import Tool, ToolInput, ToolResult

class BashTool:
    def __init__(self, sandbox):
        self._sandbox = sandbox

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return "Execute bash commands in sandbox"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to execute"}
            },
            "required": ["command"]
        }

    async def run(self, input: ToolInput) -> ToolResult:
        result = await self._sandbox.run(input.arguments["command"])
        return ToolResult(
            messages=[{
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": input.call_id or "",
                "name": self.name,
            }],
            should_stop=False,
            metrics={"execution_time": result.duration},
            metadata={"exit_code": result.exit_code}
        )

    def to_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema
        )
```

### Stateful Tools (Class Methods)

The @tool decorator works with class methods via descriptor protocol:

```python
class ChromaTool:
    def __init__(self, client):
        self._client = client

    @tool
    async def search(self, query: str) -> str:
        """Search using ChromaDB."""
        results = await self._client.search(query)
        return json.dumps({"results": results})

# Usage
chroma_tool = ChromaTool(client)
bound_tool = chroma_tool.search  # Bound to instance
# bound_tool implements Tool protocol
```

## Benefits

### 1. Easy Contribution from Worm → Cookbook

Tools written in worm can be copied with minimal changes:

```python
# In worm: projects/worm/rl/tools/bash_tool.py
class BashTool:
    async def run(self, input: ToolInput) -> ToolResult:
        ...

# Copy to cookbook: tinker_cookbook/tool_use/tools/bash_tool.py
# Just adjust imports - the logic is identical!
```

### 2. Richer Return Type

**Metrics collection:**
```python
return ToolResult(
    messages=[...],
    metrics={
        "execution_time": 0.5,
        "tokens_used": 100,
        "cache_hit": 1.0,
    }
)
```

**Early stopping:**
```python
# Tool can signal to end the episode
return ToolResult(
    messages=[...],
    should_stop=True,  # Episode will end immediately
)
```

**Metadata for debugging:**
```python
return ToolResult(
    messages=[...],
    metadata={
        "sandbox_id": "abc123",
        "retry_count": 2,
    }
)
```

### 3. Backward Compatible

Existing tools using @tool decorator continue to work - the decorator handles the wrapping automatically.

## Migration Guide

If you have custom tools implementing the old `ToolInterface` ABC:

**Old code:**
```python
class MyTool(ToolInterface):
    async def invoke(self, arguments: dict[str, Any]) -> str:
        result = await process(arguments)
        return json.dumps(result)
```

**New code:**
```python
class MyTool:  # No inheritance needed (protocol-based)
    async def run(self, input: ToolInput) -> ToolResult:
        result = await process(input.arguments)
        return ToolResult(
            messages=[{
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": input.call_id or "",
                "name": self.name,
            }]
        )
```

## Testing

Comprehensive tests in `tinker_cookbook/tests/test_tool_protocol.py` cover:
- @tool decorator functionality
- Parameter validation
- Error handling
- Manual protocol implementation
- Stateful tools
- call_id preservation

Run tests:
```bash
uv run pytest tinker_cookbook/tests/test_tool_protocol.py -v
```

## Next Steps

1. **Copy worm tools to cookbook**: Tools like BashTool, WikiSearchTool can now be easily shared
2. **Create shared tool library**: Build a collection of reusable tools
3. **Adopt ToolResource protocol** (future): Consider adopting worm's resource management pattern

## Comparison with Worm

| Aspect | tinker-cookbook | projects/worm |
|--------|----------------|---------------|
| **Core protocol** | ✅ Same | ✅ Same |
| **Method name** | `run()` | `run()` |
| **Return type** | `ToolResult` | `ToolResult` |
| **@tool decorator** | ✅ Yes | ❌ No (manual implementation) |
| **Resource management** | Manual | `ToolResource` protocol |
| **Distribution** | Not built-in | Ray-based |

The cookbook adds convenience (@tool decorator) on top of worm's protocol, while keeping the core interface identical for easy tool sharing.
