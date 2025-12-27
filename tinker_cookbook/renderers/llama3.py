"""
Llama3Renderer - Llama 3 chat format.

Format like this:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>

    What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Note: The HF template prepends "Cutting Knowledge Date: December 2023\\nToday Date: {date}"
to system messages. We chose not to do this because it seemed janky. If you want to match
the HF template exactly, modify render_message to prepend this info for system messages.
"""

import json
import re

import tinker

from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    ToolCall,
    ToolSpec,
    UnparsedToolCall,
    ensure_text,
    parse_response_for_stop_token,
)


class Llama3Renderer(Renderer):
    """
    Format like this:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>

        What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    Note: The HF template prepends "Cutting Knowledge Date: December 2023\\nToday Date: {date}"
    to system messages. We chose not to do this because it seemed janky. If you want to match
    the HF template exactly, modify render_message to prepend this info for system messages.
    """

    @property
    def has_extension_property(self) -> bool:
        """Llama3 satisfies the extension property - no content is stripped from history."""
        return True

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        # Determine role for header
        # Tool responses use "ipython" role in Llama 3 format
        role = message["role"]
        if role == "tool":
            role = "ipython"

        header_str = f"<|start_header_id|>{role}<|end_header_id|>\n\n"

        # Build output content
        output_str = ensure_text(message["content"])

        # Handle tool calls in assistant messages
        # Llama 3 format: <function=function_name>{"arg": "value"}</function>
        if "tool_calls" in message and message["tool_calls"]:
            tool_call_strs = []
            for tool_call in message["tool_calls"]:
                func_name = tool_call.function.name
                args = tool_call.function.arguments
                tool_call_strs.append(f"<function={func_name}>{args}</function>")
            output_str += "".join(tool_call_strs)

        output_str += "<|eot_id|>"

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_str, add_special_tokens=False)
            )
        ]
        return RenderedMessage(header=header, output=output)

    @property
    def _bos_tokens(self) -> list[int]:
        return self.tokenizer.encode("<|begin_of_text|>", add_special_tokens=False)

    @property
    def _end_message_token(self) -> int:
        (token,) = self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)
        return token

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def _parse_llama_tool_calls(
        self, content: str
    ) -> tuple[list[ToolCall], list[UnparsedToolCall]]:
        """Parse tool calls from Llama 3 format.

        Llama 3 uses: <function=function_name>{"arg": "value"}</function>

        Returns:
            Tuple of (successfully parsed tool calls, failed parses).
        """
        tool_calls: list[ToolCall] = []
        unparsed_tool_calls: list[UnparsedToolCall] = []

        for match in re.finditer(
            r"<function=(\w+)>(.*?)</function>",
            content,
            re.DOTALL,
        ):
            raw_text = match.group(0)
            func_name = match.group(1)
            args_str = match.group(2).strip()
            try:
                # Validate JSON
                json.loads(args_str)
                tool_calls.append(
                    ToolCall(
                        function=ToolCall.FunctionBody(name=func_name, arguments=args_str),
                    )
                )
            except json.JSONDecodeError as e:
                unparsed_tool_calls.append(
                    UnparsedToolCall(raw_text=raw_text, error=f"Invalid JSON: {e}")
                )

        return tool_calls, unparsed_tool_calls

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            return assistant_message, False

        assert isinstance(assistant_message["content"], str)
        content = assistant_message["content"]

        # Parse tool calls
        tool_calls, unparsed_tool_calls = self._parse_llama_tool_calls(content)
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        if unparsed_tool_calls:
            assistant_message["unparsed_tool_calls"] = unparsed_tool_calls

        # Strip all function blocks from content (both parsed and unparsed)
        if tool_calls or unparsed_tool_calls:
            content = re.sub(
                r"\s*<function=\w+>.*?</function>",
                "",
                content,
                flags=re.DOTALL,
            )
            assistant_message["content"] = content.strip()

        return assistant_message, True

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Create system message with Llama 3 tool specifications.

        Llama 3.1 supports two tool calling formats. We use the function-tag format
        which works for custom tools without special environment setup.

        Reference: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md
        """
        tools_text = ""
        if tools:
            tool_lines = "\n\n".join(json.dumps(tool, indent=2) for tool in tools)
            tools_text = f"""You have access to the following functions:

{tool_lines}

If you choose to call a function, ONLY reply in the following format with no prefix or suffix:

<function=example_function_name>{{"example_name": "example_value"}}</function>

Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line

"""

        return [Message(role="system", content=tools_text + system_prompt)]
