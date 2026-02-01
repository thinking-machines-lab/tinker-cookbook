"""Tool-use library for LLM agents."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import (
    Annotated,
    Any,
    Callable,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from tinker_cookbook.renderers.base import Message, ToolCall
from tinker_cookbook.tool_use.types import Tool, ToolInput, ToolResult, ToolSpec


def _extract_annotated_info(annotation: Any) -> tuple[Any, FieldInfo | None, str | None]:
    """
    Extract the base type, FieldInfo, and description from an Annotated type.

    This is used by the @tool decorator to extract info about the tool's parameters.
    """
    if get_origin(annotation) is not Annotated:
        return annotation, None, None

    args = get_args(annotation)
    base_type = args[0]
    field_info = None
    description = None

    for meta in args[1:]:
        if isinstance(meta, str) and description is None:
            description = meta
        elif isinstance(meta, FieldInfo):
            field_info = meta
            if meta.description and description is None:
                description = meta.description

    return base_type, field_info, description


class FunctionTool:
    """
    A tool created from a decorated function or method.

    Implements the Tool protocol. Used internally by the @tool decorator.
    """

    def __init__(self, fn: Callable[..., Any]):
        self._fn = fn
        self._instance: Any = None  # Will be set when accessed as descriptor
        self._name = fn.__name__
        self._description = fn.__doc__ or ""
        self._params_model = self._build_params_model()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def _build_params_model(self) -> type[BaseModel]:
        """Build a Pydantic model from the function signature."""
        hints = get_type_hints(self._fn, include_extras=True)
        sig = inspect.signature(self._fn)

        fields: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            annotation = hints.get(param_name, Any)
            base_type, field_info, desc = _extract_annotated_info(annotation)

            if param.default is inspect.Parameter.empty:
                default = ...
            else:
                default = param.default

            if field_info is not None:
                if field_info.default is PydanticUndefined and default is not ...:
                    field_info.default = default
                fields[param_name] = (base_type, field_info)
            else:
                fields[param_name] = (base_type, Field(default, description=desc))

        return create_model(f"{self._name}_params", **fields)

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        return self._params_model.model_json_schema()

    def to_spec(self) -> ToolSpec:
        """Convert to ToolSpec for renderer integration."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
        )

    async def run(self, input: ToolInput) -> ToolResult:
        """Execute the tool with validated arguments. Returns a ToolResult."""
        # Validate arguments
        try:
            validated = self._params_model.model_validate(input.arguments)
        except Exception as e:
            error_msg = json.dumps({"error": f"Parameter validation failed: {e}"})
            return ToolResult(
                messages=[
                    {
                        "role": "tool",
                        "content": error_msg,
                        "tool_call_id": input.call_id or "",
                        "name": self.name,
                    }
                ],
                should_stop=False,
                metrics={},
                metadata={"error": "validation_failed"},
            )

        # Execute function
        try:
            kwargs = validated.model_dump()
            args = (self._instance,) if self._instance is not None else ()
            if asyncio.iscoroutinefunction(self._fn):
                result = await self._fn(*args, **kwargs)
            else:
                result = self._fn(*args, **kwargs)

            # Serialize result to string
            if isinstance(result, str):
                content = result
            elif isinstance(result, (dict, list)):
                content = json.dumps(result)
            else:
                content = str(result)

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
                metrics={},
                metadata={},
            )

        except Exception as e:
            error_msg = json.dumps({"error": f"Tool execution failed: {e}"})
            return ToolResult(
                messages=[
                    {
                        "role": "tool",
                        "content": error_msg,
                        "tool_call_id": input.call_id or "",
                        "name": self.name,
                    }
                ],
                should_stop=False,
                metrics={},
                metadata={"error": "execution_failed"},
            )

    def __get__(self, obj: Any, objtype: type | None = None) -> FunctionTool:
        """Descriptor protocol: bind to instance when accessed as method."""
        if obj is None:
            return self
        # Create a bound copy
        bound = FunctionTool.__new__(FunctionTool)
        bound._fn = self._fn
        bound._instance = obj
        bound._name = self._name
        bound._description = self._description
        bound._params_model = self._params_model
        return bound


def tool(fn: Callable[..., Any]) -> FunctionTool:
    """
    Decorator to create a tool from a function or method.

    The decorated function should return a string (JSON for structured data).
    It will automatically be wrapped to implement the Tool protocol.

    Usage:
        @tool
        async def search(query: Annotated[str, "The search query"]) -> str:
            '''Search for information.'''
            return json.dumps({"results": await do_search(query)})

        # As class method with shared state:
        class MySharedStateTools:
            def __init__(self, api_key: str):
                self.api_key = api_key

            @tool
            async def search(self, query: Annotated[str, "Query"]) -> str:
                '''Search for information.'''
                return json.dumps(await do_search(query, api_key=self.api_key))
    """
    return FunctionTool(fn)


async def handle_tool_call(
    tools: dict[str, Tool],
    tool_call: ToolCall,
) -> ToolResult:
    """Handle a single tool call, returning a ToolResult."""
    tool_name = tool_call.function.name
    tool_call_id = tool_call.id or ""

    if tool_name not in tools:
        return ToolResult(
            messages=[
                {
                    "role": "tool",
                    "content": json.dumps({"error": f"Tool '{tool_name}' not found"}),
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                }
            ],
            should_stop=False,
            metrics={},
            metadata={"error": "tool_not_found"},
        )

    tool_obj = tools[tool_name]
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        return ToolResult(
            messages=[
                {
                    "role": "tool",
                    "content": json.dumps({"error": f"Failed to parse tool arguments: {e}"}),
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                }
            ],
            should_stop=False,
            metrics={},
            metadata={"error": "json_decode_failed"},
        )

    return await tool_obj.run(ToolInput(arguments=arguments, call_id=tool_call_id))
