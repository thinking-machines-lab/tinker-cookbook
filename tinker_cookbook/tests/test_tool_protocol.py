"""Tests for the unified Tool protocol."""

import asyncio

from tinker_cookbook.tool_use import Tool, ToolInput, ToolResult, tool


def test_tool_decorator_returns_tool_result():
    """Test that @tool decorator creates tools that return ToolResult."""

    async def _test():
        @tool
        async def my_tool(x: int, y: int) -> str:
            """Add two numbers."""
            return str(x + y)

        # Check it implements Tool protocol
        assert hasattr(my_tool, "name")
        assert hasattr(my_tool, "description")
        assert hasattr(my_tool, "parameters_schema")
        assert hasattr(my_tool, "run")

        # Check properties
        assert my_tool.name == "my_tool"
        assert "Add two numbers" in my_tool.description

        # Check run returns ToolResult
        result = await my_tool.run(ToolInput(arguments={"x": 2, "y": 3}))
        assert isinstance(result, ToolResult)
        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "tool"
        assert result.messages[0]["content"] == "5"
        assert result.should_stop is False

    asyncio.run(_test())


def test_tool_with_validation_error():
    """Test that validation errors are returned as ToolResult with error message."""

    async def _test():
        @tool
        async def my_tool(x: int) -> str:
            """Takes an integer."""
            return str(x)

        # Invalid argument type
        result = await my_tool.run(ToolInput(arguments={"x": "not_an_int"}))
        assert isinstance(result, ToolResult)
        assert len(result.messages) == 1
        assert "error" in result.messages[0]["content"].lower()
        assert "validation" in result.messages[0]["content"].lower()
        assert result.metadata.get("error") == "validation_failed"

    asyncio.run(_test())


def test_tool_with_execution_error():
    """Test that execution errors are returned as ToolResult with error message."""

    async def _test():
        @tool
        async def failing_tool(x: int) -> str:
            """A tool that always fails."""
            raise ValueError("Something went wrong")

        result = await failing_tool.run(ToolInput(arguments={"x": 1}))
        assert isinstance(result, ToolResult)
        assert len(result.messages) == 1
        assert "error" in result.messages[0]["content"].lower()
        assert "Something went wrong" in result.messages[0]["content"]
        assert result.metadata.get("error") == "execution_failed"

    asyncio.run(_test())


def test_tool_with_call_id():
    """Test that call_id is preserved in the result message."""

    async def _test():
        @tool
        async def my_tool(x: int) -> str:
            """Test tool."""
            return str(x)

        result = await my_tool.run(ToolInput(arguments={"x": 42}, call_id="call_123"))
        assert result.messages[0]["tool_call_id"] == "call_123"
        assert result.messages[0]["name"] == "my_tool"

    asyncio.run(_test())


def test_tool_protocol_manual_implementation():
    """Test that manually implementing Tool protocol works."""

    async def _test():
        from tinker_cookbook.tool_use.types import ToolSpec

        class CustomTool:
            @property
            def name(self) -> str:
                return "custom"

            @property
            def description(self) -> str:
                return "A custom tool"

            @property
            def parameters_schema(self) -> dict:
                return {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                }

            async def run(self, input: ToolInput) -> ToolResult:
                content = f"Got: {input.arguments['value']}"
                return ToolResult(
                    messages=[
                        {
                            "role": "tool",
                            "content": content,
                            "tool_call_id": input.call_id or "",
                            "name": self.name,
                        }
                    ],
                    should_stop=False,
                    metrics={"custom_metric": 1.0},
                    metadata={"custom": True},
                )

            def to_spec(self) -> ToolSpec:
                return ToolSpec(
                    name=self.name,
                    description=self.description,
                    parameters=self.parameters_schema,
                )

        tool_obj = CustomTool()

        # Check it has the required protocol methods
        assert isinstance(tool_obj, Tool)

        # Test execution
        result = await tool_obj.run(ToolInput(arguments={"value": "hello"}))
        assert isinstance(result, ToolResult)
        assert "Got: hello" in result.messages[0]["content"]
        assert result.metrics["custom_metric"] == 1.0
        assert result.metadata["custom"] is True

    asyncio.run(_test())


def test_tool_with_stateful_class():
    """Test @tool decorator on class methods (stateful tools)."""

    async def _test():
        class StatefulTool:
            def __init__(self, prefix: str):
                self.prefix = prefix

            @tool
            async def process(self, text: str) -> str:
                """Process text with prefix."""
                return f"{self.prefix}: {text}"

        tool_obj = StatefulTool(prefix="Hello")

        # Access the bound tool
        bound_tool = tool_obj.process
        assert bound_tool.name == "process"

        # Execute it
        result = await bound_tool.run(ToolInput(arguments={"text": "world"}))
        assert isinstance(result, ToolResult)
        assert result.messages[0]["content"] == "Hello: world"

    asyncio.run(_test())
