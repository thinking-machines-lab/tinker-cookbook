"""
GptOssRenderer - OpenAI's open source model format (Harmony).

Format like this (no newlines between messages, last message should end with <|return|> but
be replaced by <|end|> when continuing the convo):
    <|start|>system<|message|>You are ChatGPT...<|end|><|start|>user<|message|>How much is 1+1?<|end|><|start|>assistant<|channel|>final<|message|>2<|end|><|start|>

Harmony channels:
- analysis: Chain-of-thought (CoT) / reasoning traces (not shown to end users)
- commentary: Tool calls for developer-defined functions; also user-visible "preambles"
- final: User-facing answer text

Tool calling format:
- Tool definitions go in developer message with TypeScript-ish syntax in `functions` namespace
- Tool calls: <|start|>assistant<|channel|>commentary to=functions.name <|constrain|>json<|message|>{args}<|call|>
- Tool results: <|start|>functions.name to=assistant<|channel|>commentary<|message|>{result}<|end|>

Reference: https://raw.githubusercontent.com/openai/openai-cookbook/main/articles/openai-harmony.md
"""

import json
import re
from datetime import datetime

import tinker

from tinker_cookbook.renderers.base import (
    ContentPart,
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolSpec,
    UnparsedToolCall,
    ensure_list,
    ensure_text,
    parse_response_for_stop_token,
)
from tinker_cookbook.tokenizer_utils import Tokenizer


class GptOssRenderer(Renderer):
    """
    Renderer for OpenAI's open source models using the Harmony format.

    Format: <|start|>role<|channel|>channel<|message|>content<|end|>
    No newlines between messages. Last assistant message should end with <|return|> but
    be replaced by <|end|> when continuing the conversation.

    Harmony channels:
    - analysis: Chain-of-thought (CoT) / reasoning traces (not shown to end users)
    - commentary: Tool calls for developer-defined functions; also user-visible "preambles"
    - final: User-facing answer text

    Tool calling uses the commentary channel with special formatting:
    - Tool calls: <|start|>assistant<|channel|>commentary to=functions.name <|constrain|>json<|message|>{args}<|call|>
    - Tool results: <|start|>functions.name to=assistant<|channel|>commentary<|message|>{result}<|end|>

    Reference: https://raw.githubusercontent.com/openai/openai-cookbook/main/articles/openai-harmony.md
    """

    system_prompt = "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: {current_date}\n\nReasoning: {reasoning_effort}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
    use_system_prompt: bool = False
    reasoning_effort: str | None = None
    current_date: str | None = (
        None  # If use_system_prompt=True, will use the current date if this is None. Set this to a fixed date for deterministic system prompt.
    )

    def __init__(
        self,
        tokenizer: Tokenizer,
        use_system_prompt: bool = False,
        reasoning_effort: str | None = None,
        current_date: str | None = None,
    ):
        super().__init__(tokenizer)
        self.use_system_prompt = use_system_prompt
        self.reasoning_effort = reasoning_effort
        self.current_date = current_date
        assert use_system_prompt == (reasoning_effort is not None), (
            "Reasoning effort must be set iff using system prompt"
        )

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role = message["role"]

        # Handle tool result messages (role="tool")
        if role == "tool":
            return self._render_tool_result_message(message, ctx)

        # HF template maps "system" role to "developer" with special formatting
        if role == "system":
            role = "developer"
        header_str = f"<|start|>{role}"
        output_str = ""

        if message["role"] == "assistant":
            # Assistant channels. See https://cookbook.openai.com/articles/openai-harmony
            # Extract text and thinking from content list
            parts = ensure_list(message["content"])
            text_content = "".join(p["text"] for p in parts if p["type"] == "text")
            thinking_content = "".join(p["thinking"] for p in parts if p["type"] == "thinking")

            # Analysis channel (CoT) - always included for last message to match HF template
            if ctx.is_last:
                output_str += (
                    f"<|channel|>analysis<|message|>{thinking_content}<|end|><|start|>assistant"
                )

            # Handle tool calls (goes in commentary channel)
            if "tool_calls" in message and message["tool_calls"]:
                output_str += self._render_tool_calls(message["tool_calls"])
            else:
                # Final channel (Response Content)
                output_str += f"<|channel|>final<|message|>{text_content}"
        elif message["role"] == "system":
            # HF wraps system content as developer instructions
            output_str += f"<|message|># Instructions\n\n{ensure_text(message['content'])}\n\n"
        else:
            output_str += f"<|message|>{ensure_text(message['content'])}"

        # Determine the end token
        if ctx.is_last and message["role"] == "assistant":
            if "tool_calls" in message and message["tool_calls"]:
                # Tool call ends with <|call|>
                output_str += "<|call|>"
            else:
                # Normal assistant response ends with <|return|>
                output_str += "<|return|>"
        else:
            output_str += "<|end|>"

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_str, add_special_tokens=False)
            )
        ]
        return RenderedMessage(header=header, output=output)

    def _render_tool_calls(self, tool_calls: list[ToolCall]) -> str:
        """Render tool calls in Harmony commentary channel format.

        Each tool call becomes a separate commentary message:
        <|channel|>commentary to=functions.name <|constrain|>json<|message|>{args}

        Multiple tool calls are separated by <|end|><|start|>assistant.
        """
        result_parts = []
        for i, tc in enumerate(tool_calls):
            # Format: <|channel|>commentary to=functions.name <|constrain|>json<|message|>{args}
            result_parts.append(
                f"<|channel|>commentary to=functions.{tc.function.name} <|constrain|>json<|message|>"
                f"{tc.function.arguments}"
            )
            # If not the last tool call, close message and start new assistant message
            if i < len(tool_calls) - 1:
                result_parts.append("<|call|><|start|>assistant")
        return "".join(result_parts)

    def _render_tool_result_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render a tool result message.

        Format: <|start|>functions.name to=assistant<|channel|>commentary<|message|>{result}<|end|>
        """
        # Get the tool name from the tool_call_id or name field
        tool_name = message.get("name", "")
        if not tool_name and "tool_call_id" in message:
            # Try to extract tool name from tool_call_id (e.g., "call_get_weather_123")
            # But typically the name field should be present
            tool_name = "unknown"

        # Build the header with tool name as role and to=assistant
        header_str = f"<|start|>functions.{tool_name} to=assistant"

        # Tool results go in commentary channel
        content = ensure_text(message["content"])
        output_str = f"<|channel|>commentary<|message|>{content}<|end|>"

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_str, add_special_tokens=False)
            )
        ]
        return RenderedMessage(header=header, output=output)

    def _build_system_prompt(self) -> str:
        current_date = (
            self.current_date
            if self.current_date is not None
            else datetime.now().strftime("%Y-%m-%d")
        )
        return self.system_prompt.format(
            current_date=current_date, reasoning_effort=self.reasoning_effort
        )

    @property
    def _bos_tokens(self) -> list[int]:
        tokens = []
        if self.use_system_prompt:
            tokens.extend(
                self.tokenizer.encode(self._build_system_prompt(), add_special_tokens=False)
            )
        return tokens

    @property
    def _return_token(self) -> int:
        res = self.tokenizer.encode("<|return|>", add_special_tokens=False)
        assert len(res) == 1, f"Expected single token for <|return|>, got {len(res)}"
        return res[0]

    @property
    def _call_token(self) -> int:
        res = self.tokenizer.encode("<|call|>", add_special_tokens=False)
        assert len(res) == 1, f"Expected single token for <|call|>, got {len(res)}"
        return res[0]

    def get_stop_sequences(self) -> list[int]:
        # Both <|return|> and <|call|> are stop tokens
        # <|return|> for normal completion, <|call|> for tool calls
        return [self._return_token, self._call_token]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        # Check if response ends with <|call|> (tool call) or <|return|> (normal response)
        has_return = self._return_token in response
        has_call = self._call_token in response

        if has_call:
            # Parse tool call response
            return self._parse_tool_call_response(response)
        elif has_return:
            # Parse normal response
            return self._parse_normal_response(response)
        else:
            # No stop token - format error
            str_response = self.tokenizer.decode(response)
            return Message(role="assistant", content=str_response), False

    def _parse_normal_response(self, response: list[int]) -> tuple[Message, bool]:
        """Parse a normal response ending with <|return|>."""
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._return_token
        )
        if not parse_success:
            return assistant_message, False

        assert isinstance(assistant_message["content"], str)
        content = assistant_message["content"]

        # Parse GptOss multi-channel format into content parts
        # Format: <|channel|>channel_name<|message|>content<|end|> or <|return|>
        # Channels: analysis (thinking), commentary (tool calls), final (response)
        parts = self._parse_gptoss_channels(content)
        if parts:
            assistant_message["content"] = parts
        # else: keep as string for backward compatibility

        return assistant_message, True

    def _parse_tool_call_response(self, response: list[int]) -> tuple[Message, bool]:
        """Parse a tool call response ending with <|call|>.

        Format: <|channel|>commentary to=functions.name <|constrain|>json<|message|>{args}<|call|>
        """
        call_count = response.count(self._call_token)
        if call_count == 0:
            str_response = self.tokenizer.decode(response)
            return Message(role="assistant", content=str_response), False
        elif call_count > 1:
            raise ValueError(
                f"When parsing response, expected at most 1 <|call|> token, but got {call_count}. "
                "You probably are using the wrong stop tokens when sampling"
            )

        str_response = self.tokenizer.decode(response[: response.index(self._call_token)])

        # Parse out tool calls from the commentary channel(s)
        # Format: <|channel|>commentary to=functions.name <|constrain|>json<|message|>{args}
        tool_calls, unparsed = self._parse_tool_calls_from_response(str_response)

        # Also extract any text content (e.g., from analysis channel)
        parts = self._parse_gptoss_channels(str_response)
        content: list[ContentPart] | str = parts if parts else ""

        message: Message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if unparsed:
            message["unparsed_tool_calls"] = unparsed

        return message, True

    def _parse_tool_calls_from_response(
        self, content: str
    ) -> tuple[list[ToolCall], list[UnparsedToolCall]]:
        """Parse tool calls from Harmony commentary channel format.

        Pattern: <|channel|>commentary to=functions.name <|constrain|>json<|message|>{args}
        """
        tool_calls: list[ToolCall] = []
        unparsed: list[UnparsedToolCall] = []

        # Pattern to match tool call in commentary channel
        # <|channel|>commentary to=functions.name <|constrain|>json<|message|>{args}
        pattern = re.compile(
            r"<\|channel\|>commentary\s+to=functions\.(\w+)\s*<\|constrain\|>json<\|message\|>(.*?)(?:<\|end\|>|<\|call\|>|$)",
            re.DOTALL,
        )

        for match in pattern.finditer(content):
            tool_name = match.group(1)
            args_json = match.group(2).strip()
            raw_text = match.group(0)

            try:
                # Validate JSON and create ToolCall
                json.loads(args_json)  # Validate JSON
                tool_calls.append(
                    ToolCall(
                        function=ToolCall.FunctionBody(name=tool_name, arguments=args_json),
                        id=None,  # Harmony format doesn't include call IDs
                    )
                )
            except json.JSONDecodeError as e:
                unparsed.append(UnparsedToolCall(raw_text=raw_text, error=f"Invalid JSON: {e}"))

        return tool_calls, unparsed

    def _parse_gptoss_channels(self, content: str) -> list[ContentPart]:
        """Parse GptOss channel format into ContentPart list.

        Channels:
        - analysis: Chain-of-thought (maps to ThinkingPart)
        - final: User-facing answer (maps to TextPart)
        - commentary: Tool calls or preambles (maps to TextPart for now)
        """
        if "<|channel|>" not in content:
            return []

        parts: list[ContentPart] = []

        # Pattern to match channel messages
        # <|channel|>channel_name<|message|>content<|end|> or <|return|>
        pattern = re.compile(
            r"<\|channel\|>(\w+)(?:[^<]*)?<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)",
            re.DOTALL,
        )

        for match in pattern.finditer(content):
            channel = match.group(1)
            msg_content = match.group(2)

            if not msg_content.strip():
                continue

            if channel == "analysis":
                parts.append(ThinkingPart(type="thinking", thinking=msg_content))
            elif channel == "final":
                parts.append(TextPart(type="text", text=msg_content))
            elif channel == "commentary":
                # Commentary without tool recipient is user-visible preamble
                # (tool calls would need additional parsing of to=functions.x)
                parts.append(TextPart(type="text", text=msg_content))

        return parts

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Create conversation prefix with tools in Harmony format.

        Tools are defined in a developer message using TypeScript-ish syntax
        in a `functions` namespace, following the OpenAI Harmony spec.

        Reference: https://raw.githubusercontent.com/openai/openai-cookbook/main/articles/openai-harmony.md
        """
        tools_text = ""
        if tools:
            # Format tools as TypeScript-ish function signatures in functions namespace
            tool_defs = []
            for tool in tools:
                # Build TypeScript-style type string from JSON schema properties
                params_str = self._json_schema_to_typescript(tool["parameters"])
                tool_defs.append(f"  function {tool['name']}(_: {params_str}): any;")

            tools_text = f"""

# Tools

namespace functions {{
{chr(10).join(tool_defs)}
}}"""

        content = system_prompt + tools_text
        return [Message(role="system", content=content)]

    def _json_schema_to_typescript(self, schema: dict) -> str:
        """Convert JSON schema to TypeScript-ish type string for Harmony tools.

        Harmony uses TypeScript-style inline types for tool parameters.
        """
        if schema.get("type") != "object":
            return "any"

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        type_parts = []
        for prop_name, prop_schema in properties.items():
            prop_type = self._json_type_to_typescript(prop_schema)
            optional = "" if prop_name in required else "?"
            type_parts.append(f"{prop_name}{optional}: {prop_type}")

        return "{ " + ", ".join(type_parts) + " }"

    def _json_type_to_typescript(self, schema: dict) -> str:
        """Convert a single JSON schema type to TypeScript."""
        json_type = schema.get("type", "any")

        if json_type == "string":
            # Check for enum
            if "enum" in schema:
                return " | ".join(f'"{v}"' for v in schema["enum"])
            return "string"
        elif json_type == "number" or json_type == "integer":
            return "number"
        elif json_type == "boolean":
            return "boolean"
        elif json_type == "array":
            items_type = self._json_type_to_typescript(schema.get("items", {}))
            return f"{items_type}[]"
        elif json_type == "object":
            return self._json_schema_to_typescript(schema)
        else:
            return "any"
