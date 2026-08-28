"""Renderer for Moonshot AI's Kimi K2 models."""

import json
import re
import warnings

import tinker
import torch

from tinker_cookbook.renderers.base import (
    ContentPart,
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    TextPart,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    UnparsedToolCall,
    detect_unterminated_tool_block,
    ensure_list,
    ensure_text,
    parse_response_for_stop_token,
    parse_think_blocks,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

_TOOL_CALLS_SECTION_RE = re.compile(
    r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>"
    r"|<\|tool_call_section_begin\|>(.*?)<\|tool_call_section_end\|>",
    re.DOTALL,
)
_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*([^<]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(.*?)\s*<\|tool_call_end\|>",
    re.DOTALL,
)


def _split_tool_calls_section(content: str) -> tuple[str, str | None]:
    """Split content into text before tool calls and the tool calls section.

    Args:
        content (str): Raw response content that may contain tool call sections.

    Returns:
        tuple[str, str | None]: The text content before the tool calls section, and
            the tool calls section content (or None if no tool calls found).
    """
    match = _TOOL_CALLS_SECTION_RE.search(content)
    if not match:
        return content, None
    tool_section = match.group(1) if match.group(1) is not None else match.group(2)
    return content[: match.start()], tool_section


def _detect_dangling_tool_block(content: str) -> UnparsedToolCall | None:
    """Detect a Kimi K2 tool block opened but never closed.

    Covers both section-marker spellings and an individual call inside a
    closed section. The section/call regexes above only match complete
    blocks, so without this check a dangling tool-call intent silently
    degrades to plain text even when the response terminated cleanly on
    ``<|im_end|>``.
    """
    for open_marker, close_marker in (
        ("<|tool_calls_section_begin|>", "<|tool_calls_section_end|>"),
        ("<|tool_call_section_begin|>", "<|tool_call_section_end|>"),
        ("<|tool_call_begin|>", "<|tool_call_end|>"),
    ):
        dangling = detect_unterminated_tool_block(content, open_marker, close_marker)
        if dangling is not None:
            return dangling
    return None


def _extract_tool_name(tool_id: str) -> str:
    """Extract the tool function name from a Kimi K2 tool ID string.

    Strips the ``functions.`` prefix and ``:index`` suffix from tool IDs
    like ``functions.get_weather:0``.

    Args:
        tool_id (str): The tool identifier string.

    Returns:
        str: The extracted function name, or empty string if tool_id is empty.
    """
    if not tool_id:
        return ""
    name_part = tool_id.split(":", 1)[0]
    if "." in name_part:
        _, name_part = name_part.split(".", 1)
    return name_part


def _parse_tool_calls_section(
    tool_section: str,
) -> tuple[list[ToolCall], list[UnparsedToolCall]]:
    """Parse individual tool calls from a Kimi K2 tool calls section.

    Args:
        tool_section (str): The content inside the tool calls section markers.

    Returns:
        tuple[list[ToolCall], list[UnparsedToolCall]]: Successfully parsed tool calls
            and any tool calls that failed to parse (e.g., invalid JSON arguments).
    """
    tool_calls: list[ToolCall] = []
    unparsed_tool_calls: list[UnparsedToolCall] = []

    for match in _TOOL_CALL_RE.finditer(tool_section):
        raw_text = match.group(0)
        tool_id = match.group(1).strip()
        args_str = match.group(2).strip()
        func_name = _extract_tool_name(tool_id)

        try:
            json.loads(args_str)
            tool_calls.append(
                ToolCall(
                    function=ToolCall.FunctionBody(name=func_name, arguments=args_str),
                    id=tool_id if tool_id else None,
                )
            )
        except json.JSONDecodeError as e:
            unparsed_tool_calls.append(
                UnparsedToolCall(raw_text=raw_text, error=f"Invalid JSON: {e}")
            )

    return tool_calls, unparsed_tool_calls


class KimiK2Renderer(Renderer):
    """
    Format for moonshotai/Kimi-K2-Thinking:
        <|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>
        <|im_user|>user<|im_middle|>What can you help me with?<|im_end|>
        <|im_assistant|>assistant<|im_middle|><think>reasoning</think>I can help you with...<|im_end|>

    Historical assistant messages use empty <think></think> blocks, while the assistant messages after the
    last non-tool-call assistant message preserves reasoning_content in the thinking block.

    Note: Per the HuggingFace chat template, the default system message is automatically
    prepended if no system message is provided. This ensures train-eval consistency when
    using HF's apply_chat_template for inference.
    """

    supports_streaming = True

    DEFAULT_SYSTEM_PROMPT = "You are Kimi, an AI assistant created by Moonshot AI."

    def __init__(self, tokenizer: Tokenizer, strip_thinking_from_history: bool = True):
        """Initialize the Kimi K2 renderer.

        Args:
            tokenizer (Tokenizer): The tokenizer to use for encoding.
            strip_thinking_from_history (bool): When True (default), replaces thinking
                content with empty ``<think></think>`` in historical assistant messages.
                Set to False to preserve thinking in history for multi-turn RL.
        """
        super().__init__(tokenizer)
        self.strip_thinking_from_history = strip_thinking_from_history

    def _ensure_system_message(self, messages: list[Message]) -> list[Message]:
        """Ensure a default system message is present if none exists.

        This matches the HuggingFace chat template behavior where a default system
        message is automatically added when none is provided.

        The default system message is inserted at the appropriate position:
        - If messages is empty: adds default system message
        - If starting with tool_declare: inserts default system after tool_declare (if no system message follows)
        - Otherwise: prepends default system message before first message (if first message isn't system)
        """
        if not messages:
            default_system = Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
            return [default_system]

        # Accept both system and tool_declare as valid starting messages
        first_role = messages[0]["role"]
        if first_role == "tool_declare":
            # Check if a system message already exists after tool_declare
            if len(messages) >= 2 and messages[1]["role"] == "system":
                return messages
            # No system message, insert default after tool_declare
            default_system = Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
            return [messages[0], default_system] + list(messages[1:])
        elif first_role != "system":
            default_system = Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
            return [default_system] + list(messages)

        return messages

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render a chat message into Kimi K2 ``<|im_*|>`` token format.

        Each message uses role-specific tokens (``<|im_user|>``, ``<|im_assistant|>``,
        ``<|im_system|>``) with ``<|im_middle|>`` separating the role from content.
        For assistant messages, ``ctx.in_last_assistant_turn`` controls whether thinking is
        preserved or replaced with empty ``<think></think>``.

        Args:
            message (Message): The chat message to render.
            ctx (RenderContext): Positional context including index and is_last flag.

        Returns:
            RenderedMessage: Header and output token chunks for the message.
        """
        role = message["role"]

        # Build role token based on role type
        if role == "user":
            header_str = f"<|im_user|>{role}<|im_middle|>"
        elif role == "assistant":
            header_str = f"<|im_assistant|>{role}<|im_middle|>"
        elif role == "system":
            header_str = f"<|im_system|>{role}<|im_middle|>"
        elif role == "tool_declare":
            # Tool declaration uses system token but with "tool_declare" as display name
            header_str = f"<|im_system|>{role}<|im_middle|>"
        elif role == "tool":
            # HF template uses message.name if present, otherwise role
            role_name = message.get("name")
            if not role_name:
                warnings.warn(
                    "Tool message missing 'name' field. Using 'tool' as fallback. "
                    "Consider setting 'name' to match the tool function name for better context.",
                    UserWarning,
                    stacklevel=3,
                )
                role_name = role
            header_str = f"<|im_system|>{role_name}<|im_middle|>"

            # Tool responses have special formatting - need tool_call_id to correlate with the call
            tool_call_id = message.get("tool_call_id", "")
            if not tool_call_id:
                warnings.warn(
                    "Tool message missing 'tool_call_id' field. KimiK2Renderer requires 'tool_call_id' "
                    "to render tool results correctly. The value should match ToolCall.id from the "
                    "assistant's tool_calls.",
                    UserWarning,
                    stacklevel=3,
                )
            header_str += f"## Return of {tool_call_id}\n"
        else:
            # Unknown roles default to system-style formatting
            header_str = f"<|im_system|>{role}<|im_middle|>"

        # Build output content
        content = message["content"]
        output: list[tinker.ModelInputChunk] = []
        if role == "assistant":
            output_str = ""
            # Extract thinking and text from content list
            parts = ensure_list(content)
            thinking_content = "".join(p["thinking"] for p in parts if p["type"] == "thinking")
            text_content = "".join(p["text"] for p in parts if p["type"] == "text")

            # Preserve thinking for the last assistant message, or for all messages
            # when strip_thinking_from_history is False.
            if (
                ctx.in_last_assistant_turn or not self.strip_thinking_from_history
            ) and thinking_content:
                output_str = f"<think>{thinking_content}</think>"
            else:
                output_str = "<think></think>"
            output_str += text_content

            # Handle tool calls
            if "tool_calls" in message and message["tool_calls"]:  # noqa: RUF019
                output_str += "<|tool_calls_section_begin|>"
                for idx, tool_call in enumerate(message["tool_calls"]):
                    tool_id = tool_call.id
                    if not tool_id:
                        tool_id = f"functions.{tool_call.function.name}:{idx}"
                    args = tool_call.function.arguments
                    output_str += f"<|tool_call_begin|>{tool_id}<|tool_call_argument_begin|>{args}<|tool_call_end|>"
                output_str += "<|tool_calls_section_end|>"
            output_str += "<|im_end|>"
            output.append(tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(output_str)))
        elif isinstance(content, str) or (len(content) == 1 and content[0]["type"] == "text"):
            # Single-part/text content
            output_str = ensure_text(content) + "<|im_end|>"
            output.append(tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(output_str)))
        else:
            # Mult-part content (e.g. text+image(s))
            assert isinstance(content, list), f"Expected list of content parts, got {type(content)}"
            output = self._encode_multipart_content(
                content + [TextPart(type="text", text="<|im_end|>")]
            )

        header = tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(header_str))

        return RenderedMessage(header=header, output=output)

    def _encode_multipart_content(self, content: list[ContentPart]) -> list[tinker.ModelInputChunk]:
        raise NotImplementedError(
            "Multipart/Image content encoding is not supported for Kimi K2 renderer"
        )

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        """Build a generation prompt, prepending the default system message if absent."""
        return super().build_generation_prompt(
            self._ensure_system_message(messages), role=role, prefill=prefill
        )

    def build_supervised_examples(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_TURN,
    ) -> list[tuple[tinker.ModelInput, torch.Tensor]]:
        """Build multiple supervised examples for multi-turn conversations.

        Since Kimi K2 does not satisfy the extension property (thinking is stripped
        from history), this method splits multi-turn conversations into separate
        training examples -- one per user turn -- to avoid training on incorrect
        token sequences.

        Args:
            messages (list[Message]): The full conversation messages.
            train_on_what (TrainOnWhat): Which message tokens to assign training weight.
                For LAST_ASSISTANT_MESSAGE or LAST_ASSISTANT_TURN, delegates to
                build_supervised_example directly.

        Returns:
            list[tuple[tinker.ModelInput, torch.Tensor]]: A list of (model_input, weights)
                pairs, one per training example.
        """

        if (
            train_on_what == TrainOnWhat.LAST_ASSISTANT_MESSAGE
            or train_on_what == TrainOnWhat.LAST_ASSISTANT_TURN
        ):
            return [self.build_supervised_example(messages, train_on_what=train_on_what)]

        # split the messages into turns by user messages
        user_message_idxs = [
            idx for idx, message in enumerate(messages) if message["role"] == "user"
        ]

        supervised_examples: list[tuple[tinker.ModelInput, torch.Tensor]] = []

        if train_on_what != TrainOnWhat.ALL_ASSISTANT_MESSAGES:
            warnings.warn(
                "WARNING: Using train_on_what=ALL_MESSAGES/ALL_TOKENS/ALL_USER_AND_SYSTEM_MESSAGES/CUSTOMIZED with a renderer that "
                "does not satisfy the extension property (has_extension_property=False). "
                "The behavior is we apply the same `train_on_what` to all turns. This may not be the desired behavior.",
                UserWarning,
                stacklevel=3,
            )

        # We separate the turns by user messages. The first turn is the messages before the second user message.
        for user_message_idx in [*user_message_idxs[1:], len(messages)]:
            current_messages = messages[:user_message_idx]
            if train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES:
                supervised_examples.append(
                    self.build_supervised_example(
                        current_messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN
                    )
                )
            else:
                supervised_examples.append(
                    self.build_supervised_example(current_messages, train_on_what=train_on_what)
                )

        return supervised_examples

    def _last_assistant_turn_start_index(self, messages: list[Message]) -> int:
        """The turn starts after the last assistant message that did not call a tool.

        Kimi's template keeps the reasoning of every assistant message after that point, so
        a turn spans the whole assistant/tool exchange rather than the messages after the
        last user one. The final message is excluded from the scan: it is the target, and a
        turn cannot start after itself.
        """
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx]["role"] == "assistant" and not messages[idx].get("tool_calls"):
                return idx + 1
        return 0

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """Build a supervised example, prepending the default system message if absent."""
        return super().build_supervised_example(
            self._ensure_system_message(messages), train_on_what=train_on_what
        )

    @property
    def _end_message_token(self) -> int:
        tokens = self.tokenizer.encode("<|im_end|>")
        assert len(tokens) == 1, f"Expected single token for <|im_end|>, got {len(tokens)}"
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        """Return stop sequences for Kimi K2 generation.

        Returns:
            list[int]: Single-element list containing the ``<|im_end|>`` token ID.
        """
        return [self._end_message_token]

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        """Parse sampled token IDs back into an assistant Message.

        Normalizes response tokens, strips the ``<|im_end|>`` stop token, and parses
        ``<think>...</think>`` blocks and Kimi K2 tool call sections into structured
        content and ToolCall objects.

        Args:
            response (list[int]): Raw token IDs from the sampler.

        Returns:
            tuple[Message, ParseTermination]: ``STOP_SEQUENCE`` if the
                ``<|im_end|>`` stop token was found, ``MALFORMED`` otherwise.
        """
        response = self._normalize_response_tokens(response)
        assistant_message, termination = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not termination.is_clean:
            return assistant_message, termination

        content = assistant_message["content"]
        assert isinstance(content, str)

        # Handle tool calls if present
        text_content, tool_section = _split_tool_calls_section(content)
        if tool_section is not None:
            tool_calls, unparsed_tool_calls = _parse_tool_calls_section(tool_section)
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            if unparsed_tool_calls:
                assistant_message["unparsed_tool_calls"] = unparsed_tool_calls

        dangling = _detect_dangling_tool_block(content)
        if dangling is not None:
            assistant_message["unparsed_tool_calls"] = [
                *assistant_message.get("unparsed_tool_calls", []),
                dangling,
            ]

        content_parts = parse_think_blocks(text_content)
        assistant_message["content"] = content_parts if content_parts is not None else text_content

        return assistant_message, termination

    def _parse_response_for_streaming(
        self, response: list[int]
    ) -> tuple[Message, ParseTermination]:
        """Parse response for streaming, always applying full content parsing.

        Unlike parse_response which short-circuits on missing stop token,
        this always parses think blocks and tool calls from the content.
        This matches the original KimiK2StreamingParser.finish() behavior
        where content parsing was applied regardless of stop token presence.
        """
        message, termination = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )

        content = message.get("content", "")
        if isinstance(content, str):
            text_content, tool_section = _split_tool_calls_section(content)
            if tool_section is not None:
                tool_calls, unparsed_tool_calls = _parse_tool_calls_section(tool_section)
                if tool_calls:
                    message["tool_calls"] = tool_calls
                if unparsed_tool_calls:
                    message["unparsed_tool_calls"] = unparsed_tool_calls

            dangling = _detect_dangling_tool_block(content)
            if dangling is not None:
                message["unparsed_tool_calls"] = [
                    *message.get("unparsed_tool_calls", []),
                    dangling,
                ]

            content_parts = parse_think_blocks(text_content)
            message["content"] = content_parts if content_parts is not None else text_content

        return message, termination

    def to_openai_message(self, message: Message) -> dict:
        """Convert a Message to OpenAI API format with reasoning_content for thinking.

        Kimi K2's HF template explicitly expects reasoning_content as a separate field.
        """
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
        if "tool_calls" in message and message["tool_calls"]:  # noqa: RUF019
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

        # Handle tool response fields
        if message["role"] == "tool":
            if "tool_call_id" in message:
                result["tool_call_id"] = message["tool_call_id"]
            if "name" in message:
                result["name"] = message["name"]

        return result

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Create system messages with Kimi K2 tool specifications.

        Per the HuggingFace chat template, Kimi K2 places the tool_declare message
        BEFORE the regular system message. The tool_declare payload expects the
        OpenAI-style tool schema ({"type":"function","function":{...}}).
        If no system_prompt is provided, uses the default system prompt to match
        HuggingFace chat template behavior.

        Reference: https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/main/chat_template.jinja
        """
        messages: list[Message] = []

        # Tool declaration message comes first (per HF chat template)
        if tools:
            tools_payload = [{"type": "function", "function": tool} for tool in tools]
            # Use sort_keys=True since Kimi K2 sorts keys alphabetically with its own custom apply_chat_template function
            tools_json = json.dumps(tools_payload, separators=(",", ":"), sort_keys=True)
            messages.append(Message(role="tool_declare", content=tools_json))

        # Regular system message second (use default if none provided)
        actual_system_prompt = system_prompt if system_prompt else self.DEFAULT_SYSTEM_PROMPT
        messages.append(Message(role="system", content=actual_system_prompt))

        return messages
