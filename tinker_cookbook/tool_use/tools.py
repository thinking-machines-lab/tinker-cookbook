"""Tool-use library for LLM agents."""

from __future__ import annotations

import asyncio
import inspect
import json
from abc import ABC, abstractmethod
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

from tinker_cookbook.renderers.base import Message, ToolCall, ToolSpec


class ToolInterface(ABC):
    """Abstract base class for tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name shown to the model."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description shown to the model."""
        ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema for tool parameters shown to the model."""
        ...

    @abstractmethod
    async def invoke(self, arguments: dict[str, Any]) -> str:
        """Execute the tool with validated arguments. Returns content string."""
        ...

    def to_spec(self) -> ToolSpec:
        """Convert to ToolSpec for renderer integration."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }


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


class FunctionTool(ToolInterface):
    """
    A tool created from a decorated function or method.

    Used internally by the @tool decorator.
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

    async def invoke(self, arguments: dict[str, Any]) -> str:
        """Invoke the tool with the given arguments dict."""
        try:
            validated = self._params_model.model_validate(arguments)
        except Exception as e:
            return json.dumps({"error": f"Parameter validation failed: {e}"})

        try:
            kwargs = validated.model_dump()
            args = (self._instance,) if self._instance is not None else ()
            if asyncio.iscoroutinefunction(self._fn):
                result = await self._fn(*args, **kwargs)
            else:
                result = self._fn(*args, **kwargs)

            # Serialize result to string
            if isinstance(result, str):
                return result
            if isinstance(result, (dict, list)):
                return json.dumps(result)
            return str(result)

        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {e}"})

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
    tools: dict[str, ToolInterface],
    tool_call: ToolCall,
) -> Message:
    """Handle a single tool call, returning a tool result message."""
    tool_name = tool_call.function.name
    tool_call_id = tool_call.id or ""

    if tool_name not in tools:
        return {
            "role": "tool",
            "content": json.dumps({"error": f"Tool '{tool_name}' not found"}),
            "tool_call_id": tool_call_id,
            "name": tool_name,
        }

    tool_obj = tools[tool_name]
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        return {
            "role": "tool",
            "content": json.dumps({"error": f"Failed to parse tool arguments: {e}"}),
            "tool_call_id": tool_call_id,
            "name": tool_name,
        }

    content = await tool_obj.invoke(arguments)
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": tool_call_id,
        "name": tool_name,
    }
