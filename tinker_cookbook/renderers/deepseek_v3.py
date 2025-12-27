"""
DeepSeek V3 family renderers.

Includes:
- DeepSeekV3ThinkingRenderer: V3 models in thinking mode
- DeepSeekV3DisableThinkingRenderer: V3 models with thinking disabled
"""

import json
import re

import tinker

from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    ToolCall,
    ToolSpec,
    UnparsedToolCall,
    ensure_text,
    parse_response_for_stop_token,
    parse_think_blocks,
    remove_thinking,
)
from tinker_cookbook.tokenizer_utils import Tokenizer


class DeepSeekV3ThinkingRenderer(Renderer):
    """
    Renderer for DeepSeek V3 models in THINKING mode.

    Format:
        <|begin_of_sentence|><|User|>question<|Assistant|><think>reasoning</think>answer<|end_of_sentence|>

    For non-thinking mode, use DeepSeekV3DisableThinkingRenderer instead.

    System messages at position 0 are rendered without role tokens (matching HF template).
    System messages at later positions require system_role_as_user=True to convert to user role.

    The default strip_thinking_from_history=True matches HF behavior where thinking
    traces are removed from historical assistant messages in multi-turn conversations.
    Use strip_thinking_from_history=False for multi-turn RL to get the extension property.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        system_role_as_user: bool = False,
        strip_thinking_from_history: bool = True,
    ):
        super().__init__(tokenizer)
        self.system_role_as_user = system_role_as_user
        self.strip_thinking_from_history = strip_thinking_from_history

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render a single message to tokens.

        Args:
            message: The message to render.
            ctx: Context about the message's position, including:
                - idx: The index of this message (0-based)
                - is_last: Whether this is the last message (affects thinking stripping)
                - prev_message: The previous message, used to detect post-tool formatting
        """
        # Check if this assistant message follows a tool response
        follows_tool = ctx.prev_message is not None and ctx.prev_message["role"] == "tool"

        content = message["content"]

        if message["role"] == "system":
            # HF template collects all system messages at the start without role tokens
            # We only support this for idx=0; later system messages need system_role_as_user=True
            content_str = ensure_text(content)
            if ctx.idx == 0:
                header_tokens: list[int] = []
                output_str = content_str
            elif self.system_role_as_user:
                # Convert later system messages to user role
                role_token = self._get_special_token("User")
                header_tokens = [role_token]
                output_str = content_str
            else:
                raise ValueError(
                    "DeepSeek only supports system message at start. "
                    "Use system_role_as_user=True to convert later system messages to user role."
                )
        elif message["role"] == "user":
            role_token = self._get_special_token("User")
            header_tokens = [role_token]
            output_str = ensure_text(content)
        elif message["role"] == "assistant":
            has_tool_calls = "tool_calls" in message and message["tool_calls"]

            if isinstance(content, list):
                # Structured content - handle with list operations
                parts = content
                if self.strip_thinking_from_history and not has_tool_calls and not ctx.is_last:
                    # Remove thinking parts for historical messages
                    parts = remove_thinking(parts)
                # Render parts in order, preserving interleaved thinking/text structure.
                # No separator needed - whitespace is preserved in TextPart for roundtrip identity.
                rendered_parts = []
                for p in parts:
                    if p["type"] == "thinking":
                        rendered_parts.append(f"<think>{p['thinking']}</think>")
                    elif p["type"] == "text":
                        rendered_parts.append(p["text"])
                    # ToolCallPart handled via message's tool_calls field
                output_content = "".join(rendered_parts)
            else:
                # String content - pass through as-is.
                # Note: strip_thinking_from_history only works with list-based content.
                # For stripping to work on historical messages, use structured content
                # with ThinkingPart separated from text (as returned by parse_response).
                output_content = content

            if follows_tool:
                # Post-tool assistant: no role token, content flows directly after tool output
                header_tokens = []
                output_str = output_content
            else:
                # Normal assistant message
                role_token = self._get_special_token("Assistant")
                header_tokens = [role_token]
                output_str = output_content
        elif message["role"] == "tool":
            # Tool responses use special tool output tokens to match HF template
            header_tokens = self.tokenizer.encode(
                "<｜tool▁output▁begin｜>", add_special_tokens=False
            )
            output_str = ensure_text(content) + "<｜tool▁output▁end｜>"
        else:
            raise ValueError(f"Unsupported role: {message['role']}")

        # Handle tool calls in assistant messages
        # HF format: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>name<｜tool▁sep｜>args<｜tool▁call▁end｜><｜tool▁calls▁end｜>
        if "tool_calls" in message and message["tool_calls"]:
            output_str += "<｜tool▁calls▁begin｜>"
            for tool_call in message["tool_calls"]:
                func_name = tool_call.function.name
                args = tool_call.function.arguments
                output_str += (
                    f"<｜tool▁call▁begin｜>{func_name}<｜tool▁sep｜>{args}<｜tool▁call▁end｜>"
                )
            output_str += "<｜tool▁calls▁end｜>"

        output_tokens = self.tokenizer.encode(output_str, add_special_tokens=False)

        # Add end_of_sentence only for assistant messages with content
        # (not for empty generation prompt messages)
        if message["role"] == "assistant" and message["content"]:
            output_tokens.append(self._end_message_token)

        output: list[tinker.ModelInputChunk] = [tinker.types.EncodedTextChunk(tokens=output_tokens)]
        # Only include header if non-empty; tinker rejects empty token chunks with
        # "Chunk N has empty tokens list". This happens for system messages at idx=0.
        if header_tokens:
            return RenderedMessage(
                header=tinker.types.EncodedTextChunk(tokens=header_tokens), output=output
            )
        else:
            return RenderedMessage(output=output)

    def _get_special_token(self, name: str) -> int:
        sep = chr(65372)
        s = f"<{sep}{name}{sep}>"
        res = self.tokenizer.encode(s, add_special_tokens=False)
        assert len(res) == 1, f"Expected single token for {s}, got {res}"
        return res[0]

    @property
    def _bos_tokens(self) -> list[int]:
        return [self._get_special_token("begin▁of▁sentence")]

    @property
    def _end_message_token(self) -> int:
        return self._get_special_token("end▁of▁sentence")

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def _parse_deepseek_tool_calls(
        self, content: str
    ) -> tuple[list[ToolCall], list[UnparsedToolCall]]:
        """Parse tool calls from DeepSeek V3.1 format.

        Expected format (per HuggingFace model card and chat template):
            <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>func_name<｜tool▁sep｜>{"arg":"value"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>

        Multiple tool calls are chained directly without separators.

        References:
            - DeepSeek V3.1 Model Card: https://huggingface.co/deepseek-ai/DeepSeek-V3.1
            - Chat Template: https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/main/assets/chat_template.jinja
        """
        tool_calls: list[ToolCall] = []
        unparsed_tool_calls: list[UnparsedToolCall] = []

        calls_match = re.search(
            r"<｜tool▁calls▁begin｜>(.*?)<｜tool▁calls▁end｜>", content, re.DOTALL
        )
        if not calls_match:
            return tool_calls, unparsed_tool_calls

        for match in re.finditer(
            r"<｜tool▁call▁begin｜>(\w+)<｜tool▁sep｜>(.*?)<｜tool▁call▁end｜>",
            calls_match.group(1),
            re.DOTALL,
        ):
            raw_text = match.group(0)
            func_name, args_str = match.group(1), match.group(2).strip()

            try:
                json.loads(args_str)
                tool_calls.append(
                    ToolCall(function=ToolCall.FunctionBody(name=func_name, arguments=args_str))
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

        # Parse DeepSeek-specific tool calls
        tool_calls, unparsed_tool_calls = self._parse_deepseek_tool_calls(content)
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        if unparsed_tool_calls:
            assistant_message["unparsed_tool_calls"] = unparsed_tool_calls

        # Strip tool calls section from content (both parsed and unparsed)
        if tool_calls or unparsed_tool_calls:
            content = re.sub(
                r"\s*<｜tool▁calls▁begin｜>.*?<｜tool▁calls▁end｜>",
                "",
                content,
                flags=re.DOTALL,
            )
            content = content.strip()

        # Parse <think>...</think> blocks into ThinkingPart/TextPart list
        parts = parse_think_blocks(content)
        if parts is not None:
            assistant_message["content"] = parts
        else:
            assistant_message["content"] = content

        return assistant_message, True

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Create system message with DeepSeek V3.1 tool specifications.

        DeepSeek V3.1 tool calling requires tools to be described in the system message
        using a specific format with ### headers and inline JSON parameters.

        Note: Tool calling is supported in non-thinking mode only.

        References:
            - DeepSeek V3.1 Model Card (ToolCall section): https://huggingface.co/deepseek-ai/DeepSeek-V3.1
            - DeepSeek V3.1 Chat Template: https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/main/assets/chat_template.jinja
            - DeepSeek API Tool Calls Guide: https://api-docs.deepseek.com/guides/tool_calls
        """
        tools_text = ""
        if tools:
            # Format each tool with ### header, description, and parameters
            tool_blocks = []
            for tool in tools:
                tool_block = f"""### {tool["name"]}
Description: {tool["description"]}

Parameters: {json.dumps(tool["parameters"])}"""
                tool_blocks.append(tool_block)

            tools_text = f"""

## Tools
You have access to the following tools:

{chr(10).join(tool_blocks)}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜><｜tool▁calls▁end｜>

Where:
- `tool_call_name` must be an exact match to one of the available tools
- `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema
- For multiple tool calls, chain them directly without separators or spaces"""

        return [Message(role="system", content=system_prompt + tools_text)]


class DeepSeekV3DisableThinkingRenderer(DeepSeekV3ThinkingRenderer):
    """
    Renderer for DeepSeek V3 models in NON-THINKING mode.

    Format:
        <|begin_of_sentence|><|User|>question<|Assistant|></think>answer<|end_of_sentence|>

    The </think> prefix signals to the model to skip reasoning and respond directly.
    Any <think>...</think> blocks in the content are stripped.

    For thinking mode, use DeepSeekV3ThinkingRenderer instead.
    """

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render message in non-thinking mode.

        For assistant messages (not following tool):
        - Strip any ThinkingPart from structured content
        - Prepend </think> to signal non-thinking mode
        """
        # Check if this assistant message follows a tool response
        follows_tool = ctx.prev_message is not None and ctx.prev_message["role"] == "tool"

        if message["role"] == "assistant" and not follows_tool:
            content = message["content"]

            # Strip thinking from content
            if isinstance(content, list):
                # Remove ThinkingPart, keep only text
                text_content = "".join(p["text"] for p in content if p["type"] == "text")
            else:
                # Strip <think>...</think> blocks from string content
                text_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

            # Prepend </think> to signal non-thinking mode
            message = message.copy()
            message["content"] = "</think>" + text_content

        return super().render_message(message, ctx)

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        """Return <｜Assistant｜></think> for generation, or empty after tool messages."""
        # No suffix after tool messages - content flows directly
        if ctx.prev_message is not None and ctx.prev_message["role"] == "tool":
            return []
        # Otherwise: <｜Assistant｜></think>
        role_token = self._get_special_token("Assistant")
        think_close = self.tokenizer.encode("</think>", add_special_tokens=False)
        return [role_token] + think_close
