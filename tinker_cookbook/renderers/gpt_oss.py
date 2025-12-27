"""
GptOssRenderer - OpenAI's open source model format.

Format like this (no newlines between messages, last message should end with <|return|> but
be replaced by <|end|> when continuing the convo):
    <|start|>system<|message|>You are ChatGPT...<|end|><|start|>user<|message|>How much is 1+1?<|end|><|start|>assistant<|channel|>final<|message|>2<|end|><|start|>
"""

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
    ToolSpec,
    ensure_list,
    ensure_text,
    parse_response_for_stop_token,
)
from tinker_cookbook.tokenizer_utils import Tokenizer


class GptOssRenderer(Renderer):
    """
    Format like this (no newlines between messages, last message should end with <|return|> but be replaced by <|end|> when continuing the convo):
        <|start|>system<|message|>You are ChatGPT...<|end|><|start|>user<|message|>How much is 1+1?<|end|><|start|>assistant<|channel|>final<|message|>2<|end|><|start|>
    TODO: support channels in input messages and tools
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
        assert message.get("tool_calls") is None, "TODO: support tools in gpt-oss renderer"
        # HF template maps "system" role to "developer" with special formatting
        role = message["role"]
        if role == "system":
            role = "developer"
        header_str = f"<|start|>{role}"
        output_str = ""
        if message["role"] == "assistant":
            # TODO: support commentary channel / tools

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

            # Final channel (Response Content)
            output_str += f"<|channel|>final<|message|>{text_content}"
        elif message["role"] == "system":
            # HF wraps system content as developer instructions
            output_str += f"<|message|># Instructions\n\n{ensure_text(message['content'])}\n\n"
        else:
            output_str += f"<|message|>{ensure_text(message['content'])}"

        if ctx.is_last and message["role"] == "assistant":
            # <|return|> is the stop token for assistant generation
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

    def get_stop_sequences(self) -> list[int]:
        return [self._return_token]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
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
        raise NotImplementedError("GptOssRenderer does not support tool calling")
