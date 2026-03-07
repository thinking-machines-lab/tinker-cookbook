"""
Granite 4.0 family renderers.

Includes:
- Granite4Renderer: Base Granite 4.0 with thinking enabled
- Granite4DisableThinkingRenderer: Granite 4.0 with thinking disabled

Chat format:
    <|start_of_role|>system<|end_of_role|>You are a helpful assistant.<|end_of_text|>
    <|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>
    <|start_of_role|>assistant<|end_of_role|><think>reasoning</think>answer<|end_of_text|>

Reference: https://huggingface.co/ibm-granite/granite-4.0-tiny-preview/blob/main/tokenizer_config.json
"""

import json
import re

import tinker
import torch

from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    UnparsedToolCall,
    ensure_text,
    parse_response_for_stop_token,
    parse_think_blocks,
)
from tinker_cookbook.tokenizer_utils import Tokenizer


class Granite4Renderer(Renderer):
    """
    Renderer for Granite 4.0 models with thinking enabled.

    This renderer matches HuggingFace's Granite 4.0 chat template behavior.
    The template always includes a system message (with a default prompt if none
    is provided by the user).

    Format:
        <|start_of_role|>system<|end_of_role|>You are a helpful assistant.
        Please ensure responses are professional, accurate, and safe.<|end_of_text|>
        <|start_of_role|>user<|end_of_role|>What can you help me with?<|end_of_text|>
        <|start_of_role|>assistant<|end_of_role|><think>
        [reasoning]
        </think>
        I can help you with...<|end_of_text|>

    The default strip_thinking_from_history=True removes thinking blocks from
    historical assistant messages. Set to False for multi-turn RL with the
    extension property.
    """

    # Static default system prompt. The HF template uses a dynamic date
    # (strftime_now), but we use a static version for reproducibility.
    # Users should provide their own system message for production use.
    DEFAULT_SYSTEM_PROMPT = "You are Granite, developed by IBM. You are a helpful AI assistant."

    def __init__(self, tokenizer: Tokenizer, strip_thinking_from_history: bool = True):
        super().__init__(tokenizer)
        self.strip_thinking_from_history = strip_thinking_from_history

    def _ensure_system_message(self, messages: list[Message]) -> list[Message]:
        """Ensure a system message is present, matching HF template behavior.

        The HF Granite chat template always emits a system message. If none is
        provided, it generates one with a dynamic date. We use a static default
        instead for reproducibility.
        """
        if not messages:
            return [Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)]
        if messages[0]["role"] != "system":
            return [Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)] + list(messages)
        return messages

    @property
    def has_extension_property(self) -> bool:
        return not self.strip_thinking_from_history

    def build_generation_prompt(
        self, messages: list[Message], role: str = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        messages = self._ensure_system_message(messages)
        return super().build_generation_prompt(messages, role, prefill)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        messages = self._ensure_system_message(messages)
        return super().build_supervised_example(messages, train_on_what)

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role = message["role"]
        content = message["content"]

        # Newline separator between messages (matches HF template)
        maybe_newline = "\n" if ctx.idx > 0 else ""

        # Granite uses <|start_of_role|>role<|end_of_role|> as header
        header_str = f"{maybe_newline}<|start_of_role|>{role}<|end_of_role|>"

        if role == "assistant":
            if isinstance(content, list):
                parts = content
                if (
                    self.strip_thinking_from_history
                    and not ctx.is_last
                ):
                    parts = [p for p in parts if p["type"] != "thinking"]
                rendered_parts = []
                for p in parts:
                    if p["type"] == "thinking":
                        rendered_parts.append(f"<think>{p['thinking']}</think>")
                    elif p["type"] == "text":
                        rendered_parts.append(p["text"])
                output_content = "".join(rendered_parts)
            else:
                output_content = content

            # Handle tool_calls: Granite uses <tool_call> JSON </tool_call>
            if "tool_calls" in message and message["tool_calls"]:
                tool_call_strs = []
                for tool_call in message["tool_calls"]:
                    payload = {
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments),
                    }
                    tool_call_strs.append(
                        f"<tool_call>\n{json.dumps(payload)}\n</tool_call>"
                    )
                output_content += "\n" + "\n".join(tool_call_strs)

        elif role == "tool":
            # Tool responses rendered with role="tool"
            output_content = ensure_text(content)
        else:
            # system, user
            output_content = ensure_text(content)

        output_content += "<|end_of_text|>"

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_content, add_special_tokens=False)
            )
        ]
        return RenderedMessage(header=header, output=output)

    @property
    def _end_message_token(self) -> int:
        tokens = self.tokenizer.encode("<|end_of_text|>", add_special_tokens=False)
        assert len(tokens) == 1, f"Expected single token for <|end_of_text|>, got {len(tokens)}"
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            return assistant_message, False

        assert isinstance(assistant_message["content"], str)
        content = assistant_message["content"]

        # Parse tool calls: <tool_call>JSON</tool_call>
        tool_calls: list[ToolCall] = []
        unparsed_tool_calls: list[UnparsedToolCall] = []
        for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL):
            raw_text = match.group(0)
            try:
                payload = json.loads(match.group(1))
                tool_calls.append(
                    ToolCall(
                        function=ToolCall.FunctionBody(
                            name=payload["name"],
                            arguments=json.dumps(payload["arguments"]),
                        )
                    )
                )
            except (json.JSONDecodeError, KeyError) as e:
                unparsed_tool_calls.append(
                    UnparsedToolCall(raw_text=raw_text, error=str(e))
                )

        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        if unparsed_tool_calls:
            assistant_message["unparsed_tool_calls"] = unparsed_tool_calls

        # Strip tool call sections from content
        if tool_calls or unparsed_tool_calls:
            content = re.sub(
                r"\s*<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL
            )
            content = content.strip()

        # Parse <think>...</think> blocks
        parts = parse_think_blocks(content)
        if parts is not None:
            assistant_message["content"] = parts
        else:
            assistant_message["content"] = content

        return assistant_message, True

    def to_openai_message(self, message: Message) -> dict:
        """Convert a Message to OpenAI API format with reasoning_content for thinking."""
        result: dict = {"role": message["role"]}

        content = message["content"]
        if isinstance(content, str):
            result["content"] = content
        else:
            thinking_parts = []
            text_parts = []
            for p in content:
                if p["type"] == "thinking":
                    thinking_parts.append(p["thinking"])
                elif p["type"] == "text":
                    text_parts.append(p["text"])

            result["content"] = "".join(text_parts)
            if thinking_parts:
                result["reasoning_content"] = "".join(thinking_parts)

        if "tool_calls" in message and message["tool_calls"]:
            result["tool_calls"] = [
                {
                    "type": "function",
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message["tool_calls"]
            ]

        if message["role"] == "tool":
            if "tool_call_id" in message:
                result["tool_call_id"] = message["tool_call_id"]
            if "name" in message:
                result["name"] = message["name"]

        return result

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Create system message with Granite 4.0 tool specifications.

        Granite 4.0 uses a structured tool description format in the system message,
        followed by instructions for using <tool_call> XML tags.
        """
        tools_text = ""
        if tools:
            tool_lines = "\n".join(
                json.dumps(
                    {"type": "function", "function": tool},
                    separators=(", ", ": "),
                )
                for tool in tools
            )
            tools_text = f"""

You have access to the following tools. Use them by responding with <tool_call> XML tags:

<tools>
{tool_lines}
</tools>

For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

        content = (system_prompt or self.DEFAULT_SYSTEM_PROMPT) + tools_text
        return [Message(role="system", content=content)]


class Granite4DisableThinkingRenderer(Granite4Renderer):
    """
    Renderer for Granite 4.0 models with thinking disabled.

    Adds an empty <think>\\n\\n</think>\\n\\n block to the assistant message header,
    signaling to the model to respond directly without extended reasoning.

    Granite 4.0 models also support <|think_off|> in the system prompt to disable
    thinking, but the empty think block approach is more consistent with the
    Qwen3/DeepSeek pattern used in tinker_cookbook.
    """

    @property
    def has_extension_property(self) -> bool:
        """Non-thinking mode always satisfies extension - no thinking to strip."""
        return True

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        rendered = super().render_message(message, ctx)

        # Add empty thinking block to header for last assistant message
        if message["role"] == "assistant" and ctx.is_last:
            content = message.get("content", "")
            if isinstance(content, str):
                has_think = "<think>" in content
            else:
                has_think = any(p["type"] == "thinking" for p in content)

            if not has_think:
                empty_think_tokens = self.tokenizer.encode(
                    "<think>\n\n</think>\n\n", add_special_tokens=False
                )
                old_header_tokens = list(rendered.header.tokens) if rendered.header else []
                new_header = tinker.EncodedTextChunk(tokens=old_header_tokens + empty_think_tokens)
                rendered = RenderedMessage(
                    header=new_header, output=rendered.output, stop_overlap=rendered.stop_overlap
                )

        return rendered
