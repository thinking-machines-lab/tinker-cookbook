"""
GptOssRenderer - OpenAI's open source model format (Harmony format).

Format like this (no newlines between messages, last message should end with <|return|> but
be replaced by <|end|> when continuing the convo):
    <|start|>system<|message|>You are ChatGPT...<|end|><|start|>user<|message|>How much is 1+1?<|end|><|start|>assistant<|channel|>final<|message|>2<|end|><|start|>

Harmony format uses channels for different content types:
- analysis: Chain-of-thought reasoning (CoT)
- commentary: Tool calls with to=functions.{name} routing
- final: The actual response content

See: https://cookbook.openai.com/articles/openai-harmony
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
)
from tinker_cookbook.tokenizer_utils import Tokenizer


# Regex patterns for parsing Harmony format
_COMMENTARY_TOOL_CALL_RE = re.compile(
    r"<\|channel\|>commentary to=functions\.(\w+)<\|constrain\|>json<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>)",
    re.DOTALL,
)
_ANALYSIS_CHANNEL_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>)", re.DOTALL
)
_FINAL_CHANNEL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.DOTALL
)


def _extract_tool_name_from_id(tool_call_id: str) -> str:
    """Extract function name from tool_call_id (format: 'functions.{name}:{idx}')."""
    if not tool_call_id or "." not in tool_call_id:
        return ""
    _, remainder = tool_call_id.split(".", 1)
    return remainder.split(":", 1)[0] if remainder else ""


def _parse_commentary_tool_calls(
    content: str,
) -> tuple[list[ToolCall], list[UnparsedToolCall]]:
    """Parse tool calls from commentary channel in Harmony format."""
    tool_calls: list[ToolCall] = []
    unparsed_tool_calls: list[UnparsedToolCall] = []

    for idx, match in enumerate(_COMMENTARY_TOOL_CALL_RE.finditer(content)):
        raw_text = match.group(0)
        func_name = match.group(1)
        args_str = match.group(2).strip()

        try:
            json.loads(args_str)  # Validate JSON
            tool_calls.append(
                ToolCall(
                    function=ToolCall.FunctionBody(name=func_name, arguments=args_str),
                    id=f"functions.{func_name}:{idx}",
                )
            )
        except json.JSONDecodeError as e:
            unparsed_tool_calls.append(
                UnparsedToolCall(raw_text=raw_text, error=f"Invalid JSON: {e}")
            )

    return tool_calls, unparsed_tool_calls


class GptOssRenderer(Renderer):
    """
    OpenAI Harmony format renderer.

    Format like this (no newlines between messages):
        <|start|>system<|message|>You are ChatGPT...<|end|><|start|>user<|message|>How much is 1+1?<|end|><|start|>assistant<|channel|>final<|message|>2<|end|><|start|>

    Channels:
    - analysis: Chain-of-thought reasoning (CoT)
    - commentary: Tool calls with to=functions.{name} routing
    - final: The actual response content
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
        """Render a message in Harmony format.

        Harmony format uses channels for different content types:
        - analysis: Chain-of-thought reasoning (CoT)
        - commentary: Tool calls with to=functions.{name} routing
        - final: The actual response content

        Tool responses use functions.{name} to=assistant routing.
        """
        role = message["role"]

        # Handle tool responses - they come from functions.{name} to assistant
        if role == "tool":
            tool_call_id = message.get("tool_call_id", "")
            func_name = _extract_tool_name_from_id(tool_call_id)
            header_str = f"<|start|>functions.{func_name}<|channel|>commentary to=assistant"
            output_str = f"<|message|>{ensure_text(message['content'])}<|end|>"

            header = tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
            )
            output: list[tinker.ModelInputChunk] = [
                tinker.types.EncodedTextChunk(
                    tokens=self.tokenizer.encode(output_str, add_special_tokens=False)
                )
            ]
            return RenderedMessage(header=header, output=output)

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

            # Handle tool calls - they go in commentary channel with to=functions.{name}
            tool_calls = message.get("tool_calls", [])
            for idx, tool_call in enumerate(tool_calls):
                func_name = tool_call.function.name
                args = tool_call.function.arguments
                output_str += f"<|channel|>commentary to=functions.{func_name}<|constrain|>json<|message|>{args}<|call|><|start|>assistant"

            # Final channel (Response Content) - only if there's text content
            if text_content:
                output_str += f"<|channel|>final<|message|>{text_content}"
                if ctx.is_last:
                    output_str += "<|return|>"
                else:
                    output_str += "<|end|>"
            elif tool_calls:
                # Tool-only message - remove trailing <|start|>assistant
                output_str = output_str.removesuffix("<|start|>assistant")
            else:
                # Empty assistant message
                output_str += "<|channel|>final<|message|>"
                if ctx.is_last:
                    output_str += "<|return|>"
                else:
                    output_str += "<|end|>"
        elif message["role"] == "system":
            # HF wraps system content as developer instructions
            output_str += (
                f"<|message|># Instructions\n\n{ensure_text(message['content'])}\n\n<|end|>"
            )
        else:
            output_str += f"<|message|>{ensure_text(message['content'])}<|end|>"

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
        """Token for <|call|> which ends a tool call in commentary channel."""
        res = self.tokenizer.encode("<|call|>", add_special_tokens=False)
        assert len(res) == 1, f"Expected single token for <|call|>, got {len(res)}"
        return res[0]

    def get_stop_sequences(self) -> list[int]:
        # Stop on both <|return|> (end of response) and <|call|> (tool call)
        return [self._return_token, self._call_token]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        """Parse a Harmony format response, extracting thinking, tool calls, and content.

        Handles two formats for analysis channel:
        1. Full format: <|channel|>analysis<|message|>{thinking}<|end|>
        2. Continuation format: {thinking}<|end|> (when generation prompt ended with analysis channel)

        Tool calls are extracted from commentary channel with to=functions.{name} routing.

        Returns:
            Tuple of (message, format_correct) where format_correct is True if the
            response ended with a valid stop token (<|return|> or <|call|>).

        Raises:
            ValueError: If response contains multiple stop tokens (indicates wrong sampling config).
        """
        raw_content = self.tokenizer.decode(response)

        # Validate that we don't have multiple stop tokens
        return_count = raw_content.count("<|return|>")
        call_count = raw_content.count("<|call|>")
        total_stop_tokens = return_count + call_count
        if total_stop_tokens > 1:
            raise ValueError(
                f"When parsing response, expected to split into 1 or 2 pieces using stop tokens, but got {total_stop_tokens + 1}. "
                "You probably are using the wrong stop tokens when sampling"
            )

        # Check if response ends with valid stop token
        format_correct = raw_content.endswith("<|return|>") or raw_content.endswith("<|call|>")

        # For simple responses without Harmony format markers, use simpler parsing
        if not any(marker in raw_content for marker in ["<|channel|>", "<|end|>", "<|start|>"]):
            content = raw_content
            if content.endswith("<|return|>"):
                content = content[: -len("<|return|>")]
            if content.endswith("<|call|>"):
                content = content[: -len("<|call|>")]
            return {"role": "assistant", "content": content}, format_correct

        # Use raw_content for pattern matching (patterns include stop tokens like <|call|>)
        content = raw_content

        assistant_message: Message = {"role": "assistant", "content": ""}
        content_parts: list[ContentPart] = []

        # Extract analysis channel content (CoT/reasoning) into ThinkingPart
        # Two cases:
        # 1. Full format: <|channel|>analysis<|message|>{thinking}<|end|>
        # 2. Continuation format: {thinking}<|end|> (when prompt ended with analysis channel)
        analysis_match = _ANALYSIS_CHANNEL_RE.search(content)
        if analysis_match:
            thinking = analysis_match.group(1).strip()
            if thinking:
                content_parts.append(ThinkingPart(type="thinking", thinking=thinking))
        else:
            # Check for continuation format - content before first <|end|> or <|start|>
            continuation_match = re.match(r"^(.*?)(?:<\|end\|>|<\|start\|>)", content, re.DOTALL)
            if continuation_match:
                thinking = continuation_match.group(1).strip()
                # Only treat as thinking if there's a final channel after it
                if thinking and "<|channel|>final" in content:
                    content_parts.append(ThinkingPart(type="thinking", thinking=thinking))

        # Parse tool calls from commentary channel
        tool_calls, unparsed_tool_calls = _parse_commentary_tool_calls(content)
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        if unparsed_tool_calls:
            assistant_message["unparsed_tool_calls"] = unparsed_tool_calls

        # Extract final channel content
        final_match = _FINAL_CHANNEL_RE.search(content)
        if final_match:
            text_content = final_match.group(1).strip()
            if text_content:
                content_parts.append(TextPart(type="text", text=text_content))

        # Set content - use list if we have parts, otherwise empty string
        if content_parts:
            assistant_message["content"] = content_parts
        else:
            # Fallback for unparseable content
            assistant_message["content"] = content

        return assistant_message, format_correct

    def _json_schema_to_typescript(self, schema: dict, indent: int = 0) -> str:
        """Convert JSON Schema to TypeScript-style type definitions.

        Per the Harmony format, tools are defined using TypeScript-style syntax
        in the developer message.
        """
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        lines = []

        for name, prop in props.items():
            optional = "?" if name not in required else ""
            prop_type = prop.get("type", "any")

            # Handle enums
            if "enum" in prop:
                type_str = " | ".join(f'"{v}"' for v in prop["enum"])
            elif prop_type == "string":
                type_str = "string"
            elif prop_type == "number" or prop_type == "integer":
                type_str = "number"
            elif prop_type == "boolean":
                type_str = "boolean"
            elif prop_type == "array":
                items = prop.get("items", {})
                item_type = items.get("type", "any")
                type_str = f"{item_type}[]"
            elif prop_type == "object":
                type_str = "object"
            else:
                type_str = "any"

            prefix = "  " * indent
            lines.append(f"{prefix}  {name}{optional}: {type_str},")

        return "\n".join(lines)

    def create_system_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Create developer message with TypeScript-style tool definitions.

        Per the Harmony format, tools are defined in a namespace block with
        TypeScript-style type definitions.

        Example output:
            namespace functions {
              // Get weather for a city
              type get_weather = (_: {
                city: string,
                unit?: "celsius" | "fahrenheit",
              }) => any;
            }
        """
        messages: list[Message] = []

        # Build TypeScript-style tool definitions
        if tools:
            tool_defs = ["namespace functions {"]
            for tool in tools:
                desc = tool.get("description", "")
                if desc:
                    tool_defs.append(f"  // {desc}")
                params_schema = tool.get("parameters", {})
                params_ts = self._json_schema_to_typescript(params_schema, indent=1)
                tool_defs.append(f"  type {tool['name']} = (_: {{")
                if params_ts:
                    tool_defs.append(params_ts)
                tool_defs.append("  }) => any;")
            tool_defs.append("}")

            # Developer message with tools
            dev_content = "\n".join(tool_defs)
            messages.append(Message(role="system", content=dev_content))

        # System prompt as second message if provided
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))

        return messages

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Delegates to create_system_prefix_with_tools for GptOss Harmony format."""
        return self.create_system_prefix_with_tools(tools, system_prompt)
