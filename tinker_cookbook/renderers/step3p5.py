"""
Step-3.5-Flash renderer (stepfun-ai/Step-3.5-Flash)

Reference: https://huggingface.co/stepfun-ai/Step-3.5-Flash
"""

import json
from typing import cast

import tinker

from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    TextPart,
    ToolCall,
    ToolSpec,
    UnparsedToolCall,
    _tool_call_payload,
    parse_content_blocks,
    parse_response_for_stop_token,
)
from tinker_cookbook.tokenizer_utils import Tokenizer


class Step3p5FlashRenderer(Renderer):
    """
    Renderer for Step-3.5-Flash model.

    Format:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        What can you help me with?<|im_end|>
        <|im_start|>assistant
        <think>
        [reasoning content]
        </think>
        I can help you with...<|im_end|>

    Reference: https://huggingface.co/stepfun-ai/Step-3.5-Flash
    """

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        maybe_newline = "\n" if ctx.idx > 0 else ""

        role = message["role"]
        # Step-3.5-Flash uses same role mapping as Qwen3
        if role == "tool":
            role = "user"

        header_str = f"{maybe_newline}<|im_start|>{role}\n"

        content = message["content"]

        if isinstance(content, list):
            # Structured content - handle with list operations
            parts = content
            rendered_parts = []
            for p in parts:
                if p["type"] == "thinking":
                    rendered_parts.append(f"<think>{p['thinking']}</think>")
                elif p["type"] == "text":
                    rendered_parts.append(p["text"])
            output_content = "".join(rendered_parts)
        else:
            output_content = content

        # Handle tool response wrapping
        if message["role"] == "tool":
            output_content = f"<tool_response>\n{output_content}\n</tool_response>"

        # Handle tool_calls field
        if "tool_calls" in message:
            output_content += "\n" + "\n".join(
                [
                    f"<tool_call>\n{json.dumps(_tool_call_payload(tool_call))}\n</tool_call>"
                    for tool_call in message["tool_calls"]
                ]
            )
        output_content += "<|im_end|>"

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
        tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        assert len(tokens) == 1, f"Expected single token for <|im_end|>, got {len(tokens)}"
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            return assistant_message, False

        # Parse <think>...</think> and <tool_call>...</tool_call> blocks
        assert isinstance(assistant_message["content"], str)
        content = assistant_message["content"]

        result = parse_content_blocks(content)

        if result is not None:
            parts, tool_results = result
            assistant_message["content"] = parts

            tool_calls = [t for t in tool_results if isinstance(t, ToolCall)]
            unparsed = [t for t in tool_results if isinstance(t, UnparsedToolCall)]
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            if unparsed:
                assistant_message["unparsed_tool_calls"] = unparsed
        else:
            assistant_message["content"] = content

        return assistant_message, True

    def to_openai_message(self, message: Message) -> dict:
        """Convert a Message to OpenAI API format."""
        result: dict = {"role": message["role"]}

        content = message["content"]
        if isinstance(content, str):
            result["content"] = content
        else:
            # Extract thinking into reasoning_content, keep text in content
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

        # Handle tool_calls
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
        """Create system message with tool specifications."""
        tools_text = ""
        if tools:
            tool_lines = "\n".join(
                json.dumps(
                    {"type": "function", "function": tool},
                    separators=(", ", ": "),
                )
                for tool in tools
            )
            tools_text = f"""# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_lines}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

        if system_prompt:
            content = system_prompt + "\n\n" + tools_text
        else:
            content = tools_text

        return [Message(role="system", content=content)]
