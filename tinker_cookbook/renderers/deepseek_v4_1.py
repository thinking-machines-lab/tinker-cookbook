"""
DeepSeek V4.1 family renderers.

Includes:
- DeepSeekV4_1Renderer: thinking mode, high reasoning effort (HF default)
- DeepSeekV4_1LowReasoningRenderer: thinking mode, low reasoning effort
- DeepSeekV4_1MaxReasoningRenderer: thinking mode, max reasoning effort
- DeepSeekV4_1DisableThinkingRenderer: chat mode (no reasoning)

Format (per the deepseek-ai/DeepSeek-V4.1-Flash chat template and ``encoding/encoding.py``):
    <｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (...)\\n\\n{system}
    <｜User｜>{question}<｜Assistant｜><think>{reasoning}</think>{answer}<｜end▁of▁sentence｜>
(shown with line breaks for readability; the actual format has none between turns)

Key format properties:
- In thinking mode every prompt opens with ``<｜System｜>`` and the reasoning effort line,
  even without a system message. Chat mode has no effort line, and the assistant header
  ends in ``</think>``.
- Tool declarations follow the system text. With no system message, the template inserts
  an empty one, so ``<｜System｜>`` appears in chat mode too.
- Tool calls are DSML blocks after the answer, one ``parameter`` per argument key.
- Tool results become ``<tool_result>...</tool_result>`` blocks in a user turn, sorted
  into the order of the preceding tool calls. Consecutive tool and user messages share
  one user turn, joined by ``\\n\\n``.
- Reasoning before the last user turn is dropped unless tools are declared.

Where the template and ``encoding.py`` disagree, this follows the template, except for
tool-call arguments given as a JSON string: the template renders them as a single
``arguments`` parameter, ``encoding.py`` renders one parameter per key, and so does this.
The ``medium`` reasoning effort exists only in the template and is not offered.

Reference: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/chat_template.jinja
"""

import contextvars
import json
import re
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import tinker
import torch
import transformers

from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.renderers.base import (
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    UnparsedToolCall,
    detect_unterminated_tool_block,
    ensure_text,
    parse_response_for_stop_token,
    parse_think_blocks,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

_EFFORT_TEMPLATE = "Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"

_TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:

<｜DSML｜ calls>
<｜DSML｜ invoke name="$TOOL_NAME">
<｜DSML｜ parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>
...
</｜DSML｜ invoke>
<｜DSML｜ invoke name="$TOOL_NAME2">
...
</｜DSML｜ invoke>
</｜DSML｜ calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""

_CALLS_OPEN = "<｜DSML｜ calls>"
_CALLS_CLOSE = "</｜DSML｜ calls>"
_CALLS_BLOCK_RE = re.compile(r"(?:\n\n)?<｜DSML｜ calls>.*?</｜DSML｜ calls>", re.DOTALL)
_INVOKE_RE = re.compile(r'<｜DSML｜ invoke name="(.*?)">\n(.*?)</｜DSML｜ invoke>', re.DOTALL)
_PARAM_RE = re.compile(
    r'<｜DSML｜ parameter name="(.*?)" string="(true|false)">(.*?)</｜DSML｜ parameter>', re.DOTALL
)

# Set by the conversation builders: assistant messages at or before this index render without
# reasoning (-1: none). The template decides this from the whole conversation -- any tool
# declaration turns dropping off, and tool results count as user turns -- while
# render_message sees one message at a time. None outside a builder; a set value also marks
# the base class's re-entry with a prefix of the normalized conversation, which must not be
# normalized twice.
_DROP_REASONING_THROUGH: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "deepseek_v4_1_drop_reasoning_through", default=None
)


def _tool_declare_specs(message: Message) -> list[dict[str, object]]:
    return [tool["function"] for tool in json.loads(ensure_text(message["content"]))]


def _render_tools(specs: list[dict[str, object]]) -> str:
    schemas = "\n".join(json.dumps(spec, ensure_ascii=False) for spec in specs)
    return _TOOLS_TEMPLATE.format(tool_schemas=schemas)


def _arguments_dict(arguments: str) -> dict[str, object]:
    parsed: object = arguments
    for _ in range(2):  # encoding.py tolerates double-encoded arguments
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except json.JSONDecodeError:
                break
    return parsed if isinstance(parsed, dict) else {"arguments": arguments}


def _render_tool_calls(tool_calls: list[ToolCall]) -> str:
    invokes = []
    for tool_call in tool_calls:
        params = [
            f'<｜DSML｜ parameter name="{key}" string="{"true" if isinstance(value, str) else "false"}">'
            f"{value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}"
            "</｜DSML｜ parameter>"
            for key, value in _arguments_dict(tool_call.function.arguments).items()
        ]
        invokes.append(
            f'<｜DSML｜ invoke name="{tool_call.function.name}">\n'
            + "\n".join(params)
            + "\n</｜DSML｜ invoke>"
        )
    return f"\n\n{_CALLS_OPEN}\n" + "\n".join(invokes) + f"\n{_CALLS_CLOSE}"


def _split_assistant_content(message: Message) -> tuple[str, str]:
    """(reasoning, answer) of an assistant message; reasoning is "" when there is none."""
    content = message["content"]
    if isinstance(content, str):
        return "", content
    reasoning, text = [], []
    for part in content:
        if part["type"] == "thinking":
            reasoning.append(part["thinking"])
        elif part["type"] == "text":
            text.append(part["text"])
        else:
            raise RendererError(
                f"DeepSeek V4.1 renderer does not support {part['type']!r} content parts"
            )
    return "".join(reasoning), "".join(text)


def _is_user_turn(message: Message) -> bool:
    return message["role"] in ("user", "tool")


def _sort_tool_results(turn: list[Message], call_order: dict[str, int]) -> list[Message]:
    """Sort a user turn's tool results into call order, in the slots tool messages hold.

    As encoding.py's ``sort_tool_results_by_call_order``: the sort is stable, and an id that
    no call has sorts as call 0.
    """
    tools = iter(
        sorted(
            (m for m in turn if m["role"] == "tool"),
            key=lambda m: call_order.get(m.get("tool_call_id") or "", 0),
        )
    )
    return [next(tools) if m["role"] == "tool" else m for m in turn]


def _normalize_messages(messages: list[Message]) -> list[Message]:
    """Reshape a cookbook conversation into the template's turns, keeping every index.

    Tool results are sorted into tool-call order within each user turn, and a ``tool_declare`` message is folded
    into the leading system message (or becomes an empty one). Consecutive tool and user
    messages share one user turn: the first carries the whole turn's text, joined by
    ``\\n\\n``, and the rest render nothing. Encoding the turn as one string keeps the
    tokenization identical to the template's, which BPE merges across ``\\n\\n`` would
    otherwise break. Indices are kept so the base class still sees which messages are user
    turns and which belong to the last assistant turn.
    """
    result = list(messages)
    for idx, message in enumerate(result):
        if message["role"] != "tool_declare":
            continue
        tools_text = _render_tools(_tool_declare_specs(message))
        if idx == 0:
            result[0] = {**message, "role": "system", "content": "\n\n" + tools_text}
        elif idx == 1 and result[0]["role"] == "system":
            system_text = ensure_text(result[0]["content"])
            result[0] = {**result[0], "content": f"{system_text}\n\n{tools_text}"}
            result[1] = {**message, "content": ""}
        else:
            raise RendererError(
                "DeepSeek V4.1 tool declarations must come first or right after the "
                "first system message; use create_conversation_prefix_with_tools"
            )

    call_order: dict[str, int] = {}
    idx = 0
    while idx < len(result):
        if tool_calls := result[idx].get("tool_calls"):
            call_order = {tc.id: order for order, tc in enumerate(tool_calls) if tc.id}
        if not _is_user_turn(result[idx]):
            idx += 1
            continue
        end = idx
        while end + 1 < len(result) and _is_user_turn(result[end + 1]):
            end += 1
        blocks = [
            f"<tool_result>{ensure_text(m['content'])}</tool_result>"
            if m["role"] == "tool"
            else ensure_text(m["content"])
            for m in _sort_tool_results(result[idx : end + 1], call_order)
        ]
        result[idx] = {**result[idx], "content": "\n\n".join(blocks)}
        for rest in range(idx + 1, end + 1):
            result[rest] = {**result[rest], "content": ""}
        idx = end + 1
    return result


def _drop_reasoning_through(messages: list[Message]) -> int:
    """The template's last user turn: user and tool messages, and later system messages."""
    return max(
        (
            idx
            for idx, m in enumerate(messages)
            if _is_user_turn(m) or (m["role"] == "system" and idx > 0)
        ),
        default=-1,
    )


class DeepSeekV4_1Renderer(Renderer):
    """
    Renderer for DeepSeek V4.1 in thinking mode with high reasoning effort.

    This matches the HF chat template's defaults (``thinking_mode`` thinking,
    ``reasoning_effort`` high, ``drop_thinking`` true). See the module docstring for the
    format.
    """

    reasoning_effort: int | None = 75
    """Reasoning effort budget (1-100), or None for chat mode."""

    supports_streaming = True

    def __init__(self, tokenizer: Tokenizer, strip_thinking_from_history: bool = True):
        """Initialize the DeepSeek V4.1 renderer.

        Args:
            tokenizer (Tokenizer): The tokenizer to use for encoding.
            strip_thinking_from_history (bool): When True (default), drops reasoning from
                assistant messages before the last user turn, unless tools are declared,
                as the template's ``drop_thinking`` does. Set to False for multi-turn RL
                to preserve the extension property.
        """
        super().__init__(tokenizer)
        self.strip_thinking_from_history = strip_thinking_from_history

        if transformers.__version__ == "5.3.0":
            warnings.warn(
                "transformers 5.3.0 has a known bug with the DeepSeek tokenizer that "
                "strips spaces during decode, which will produce incorrect outputs. "
                "Please upgrade to transformers>=5.3.1 or downgrade to transformers<5.3.0. "
                "See https://github.com/huggingface/transformers/pull/44801",
                stacklevel=2,
            )

    @property
    def _thinking(self) -> bool:
        return self.reasoning_effort is not None

    @property
    def has_extension_property(self) -> bool:
        return not (self._thinking and self.strip_thinking_from_history)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _encode_after(self, prefix: str, text: str) -> tuple[list[int], list[int]]:
        """Encode ``prefix + text`` as one string and split it into (prefix, text) tokens.

        BPE can merge across the boundary (the effort line's ``\\n\\n`` and a tools-only
        system message's leading ``\\n\\n`` become one token); a merged token goes with the
        prefix.
        """
        tokens = self._encode(prefix + text)
        split = 0
        while split < len(tokens) and len(self.tokenizer.decode(tokens[:split])) < len(prefix):
            split += 1
        return tokens[:split], tokens[split:]

    @property
    def _bos_tokens(self) -> list[int]:
        return self._encode("<｜begin▁of▁sentence｜>")

    @property
    def _end_message_token(self) -> int:
        (token,) = self._encode("<｜end▁of▁sentence｜>")
        return token

    def get_stop_sequences(self) -> list[int]:
        """Return stop sequences for DeepSeek V4.1 generation.

        Returns:
            list[int]: Single-element list containing the end-of-sentence token ID.
        """
        return [self._end_message_token]

    def _opening(self, role: Role) -> str:
        """Text before the first message: the system marker and, in thinking mode, the effort."""
        if self._thinking:
            return "<｜System｜>" + _EFFORT_TEMPLATE.format(budget=self.reasoning_effort)
        return "<｜System｜>" if role == "system" else ""

    def _last_assistant_turn_start_index(self, messages: list[Message]) -> int:
        """A later system message counts as a user turn, as in the template."""
        last = max(
            (
                idx
                for idx, m in enumerate(messages)
                if m["role"] == "user" or (m["role"] == "system" and idx > 0)
            ),
            default=-1,
        )
        return last + 1

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render one message of a conversation already shaped by ``_normalize_messages``.

        Args:
            message: The message to render.
            ctx: Context about the message's position. ``idx`` against the conversation's
                last user turn decides whether an assistant message keeps its reasoning.
        """
        role = message["role"]
        opening = self._opening(role) if ctx.idx == 0 else ""

        if role == "system":
            prefix = opening if ctx.idx == 0 else "<｜System｜>"
            header, output = self._encode_after(prefix, ensure_text(message["content"]))
            return RenderedMessage(
                header=tinker.EncodedTextChunk(tokens=header),
                output=[tinker.EncodedTextChunk(tokens=output)] if output else [],
            )

        if role == "tool_declare":  # already folded into the system message
            return RenderedMessage(output=[])

        if _is_user_turn(message):
            if ctx.prev_message is not None and _is_user_turn(ctx.prev_message):
                return RenderedMessage(output=[])  # rendered by the turn's first message
            text = self._encode(ensure_text(message["content"]))
            return RenderedMessage(
                header=tinker.EncodedTextChunk(tokens=self._encode(opening + "<｜User｜>")),
                output=[tinker.EncodedTextChunk(tokens=text)] if text else [],
            )

        if role != "assistant":
            raise RendererError(f"Unsupported role for DeepSeek V4.1: {role}")

        drop_through = _DROP_REASONING_THROUGH.get()
        keeps_reasoning = self._thinking and ctx.idx > (
            -1 if drop_through is None else drop_through
        )
        # The template ends a user turn with the assistant marker; with no such turn before
        # this message (only the first system message, or another assistant), there is none.
        prev = ctx.prev_message
        follows_user_turn = prev is not None and (
            _is_user_turn(prev) or (prev["role"] == "system" and ctx.idx > 1)
        )
        marker = "<｜Assistant｜>" + ("<think>" if keeps_reasoning else "</think>")
        header_text = opening + (marker if follows_user_turn else "")
        header = tinker.EncodedTextChunk(tokens=self._encode(header_text)) if header_text else None
        reasoning, answer = _split_assistant_content(message)
        body = (reasoning + "</think>" if keeps_reasoning else "") + answer
        if tool_calls := message.get("tool_calls"):
            body += _render_tool_calls(tool_calls)
        output = self._encode(body) + [self._end_message_token]
        return RenderedMessage(header=header, output=[tinker.EncodedTextChunk(tokens=output)])

    @contextmanager
    def _conversation(self, messages: list[Message]) -> Iterator[list[Message]]:
        if _DROP_REASONING_THROUGH.get() is not None:
            yield messages
            return
        normalized = _normalize_messages(messages)
        drops = self.strip_thinking_from_history and not any(
            m["role"] == "tool_declare" for m in messages
        )
        token = _DROP_REASONING_THROUGH.set(_drop_reasoning_through(normalized) if drops else -1)
        try:
            yield normalized
        finally:
            _DROP_REASONING_THROUGH.reset(token)

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        """Build a sampling prompt in the template's turn structure.

        See :meth:`Renderer.build_generation_prompt` and the module docstring.
        """
        with self._conversation(messages) as normalized:
            return super().build_generation_prompt(normalized, role, prefill)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """Build a supervised example in the template's turn structure.

        See :meth:`Renderer.build_supervised_example`. Tool and user messages that share a
        user turn are trained or masked together, as one message.
        """
        with self._conversation(messages) as normalized:
            return super().build_supervised_example(normalized, train_on_what)

    def _normalize_response_tokens(self, response: list[int]) -> list[int]:
        """Restore the prefilled ``<think>`` before parsing a thinking-mode response."""
        if not self._thinking:
            return response
        (think_open,) = self._encode("<think>")
        (think_close,) = self._encode("</think>")
        if think_close in response and response[0] != think_open:
            return [think_open, *response]
        return response

    def _parse_tool_calls(self, block: str) -> tuple[list[ToolCall], list[UnparsedToolCall]]:
        body = block.removeprefix("\n\n").removeprefix(_CALLS_OPEN).removesuffix(_CALLS_CLOSE)
        invokes = list(_INVOKE_RE.finditer(body))
        if not invokes or _INVOKE_RE.sub("", body).strip():
            return [], [UnparsedToolCall(raw_text=block, error="Invalid DSML: malformed invoke")]
        tool_calls: list[ToolCall] = []
        unparsed: list[UnparsedToolCall] = []
        for invoke in invokes:
            name, params_text = invoke.group(1), invoke.group(2)
            arguments: dict[str, object] = {}
            try:
                if not name.strip():
                    raise ValueError("empty function name")
                if _PARAM_RE.sub("", params_text).strip():
                    raise ValueError("text outside parameters")
                for param in _PARAM_RE.finditer(params_text):
                    key, is_string, value = param.groups()
                    if not key.strip():
                        raise ValueError("empty parameter name")
                    if key in arguments:
                        raise ValueError(f"duplicate parameter {key!r}")
                    arguments[key] = value if is_string == "true" else json.loads(value)
            except (json.JSONDecodeError, ValueError) as e:
                unparsed.append(
                    UnparsedToolCall(raw_text=invoke.group(0), error=f"Invalid DSML: {e}")
                )
                continue
            tool_calls.append(
                ToolCall(function=ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments)))
            )
        return tool_calls, unparsed

    def _parse_response_content(
        self, response: list[int], *, allow_missing_stop: bool = False
    ) -> tuple[Message, ParseTermination]:
        message, termination = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not termination.is_clean and not allow_missing_stop:
            return message, termination

        content = message["content"]
        assert isinstance(content, str)
        block = _CALLS_BLOCK_RE.search(content)
        tool_calls, unparsed = self._parse_tool_calls(block.group(0)) if block else ([], [])
        if block:
            # The format ends the response with one calls block; anything after it is malformed.
            after = content[block.end() :]
            if after.strip():
                unparsed.append(
                    UnparsedToolCall(raw_text=after, error="Invalid DSML: output after calls block")
                )
            content = content[: block.start()]
        dangling = detect_unterminated_tool_block(content, _CALLS_OPEN, _CALLS_CLOSE)
        if dangling is None and "｜DSML｜" in content:
            # The model writes this token only for tool calls, so any left over is a malformed call.
            dangling = UnparsedToolCall(
                raw_text=content, error="Invalid DSML: marker outside a calls block"
            )
        if dangling is not None:
            unparsed.append(dangling)
        if tool_calls:
            message["tool_calls"] = tool_calls
        if unparsed:
            message["unparsed_tool_calls"] = unparsed

        parts = parse_think_blocks(content)
        message["content"] = parts if parts is not None else content
        return message, termination

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        """Parse sampled token IDs back into an assistant Message.

        Restores the prefilled ``<think>``, strips the end-of-sentence stop token, and parses
        reasoning and DSML tool calls into structured content.

        Args:
            response (list[int]): Raw token IDs from the sampler.

        Returns:
            tuple[Message, ParseTermination]: ``STOP_SEQUENCE`` if the end-of-sentence
                token was found, ``MALFORMED`` otherwise.
        """
        return self._parse_response_content(self._normalize_response_tokens(response))

    def _parse_response_for_streaming(
        self, response: list[int]
    ) -> tuple[Message, ParseTermination]:
        """Parse response for streaming, always applying full content parsing.

        Unlike parse_response which short-circuits on missing stop token, this always parses
        think blocks and tool calls so the final Message emitted by streaming is complete
        even for truncated responses.
        """
        return self._parse_response_content(response, allow_missing_stop=True)

    def to_openai_message(self, message: Message) -> dict:
        """Convert a Message to OpenAI format, with reasoning_content and dict arguments.

        The HF template iterates ``tool_calls[].function.arguments.items()``, so arguments
        are passed as a dict.
        """
        result: dict = {"role": message["role"]}
        if message["role"] == "assistant":
            reasoning, answer = _split_assistant_content(message)
            result["content"] = answer
            if reasoning:
                result["reasoning_content"] = reasoning
        else:
            result["content"] = ensure_text(message["content"])
        if tool_calls := message.get("tool_calls"):
            result["tool_calls"] = [
                {
                    "type": "function",
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": _arguments_dict(tc.function.arguments),
                    },
                }
                for tc in tool_calls
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
        """Create the system message and tool declaration for DeepSeek V4.1.

        The tools render after the system prompt in the same ``<｜System｜>`` turn. They are
        kept as a separate ``tool_declare`` message so the renderer can tell that tools are
        declared, which keeps history reasoning (see the module docstring).
        """
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        if tools:
            payload = [{"type": "function", "function": dict(tool)} for tool in tools]
            messages.append(Message(role="tool_declare", content=json.dumps(payload)))
        return messages


class DeepSeekV4_1LowReasoningRenderer(DeepSeekV4_1Renderer):
    """DeepSeek V4.1 in thinking mode with low reasoning effort (``reasoning_effort='low'``)."""

    reasoning_effort = 50


class DeepSeekV4_1MaxReasoningRenderer(DeepSeekV4_1Renderer):
    """DeepSeek V4.1 in thinking mode with max reasoning effort (``reasoning_effort='max'``)."""

    reasoning_effort = 100


class DeepSeekV4_1DisableThinkingRenderer(DeepSeekV4_1Renderer):
    """DeepSeek V4.1 in chat mode (``thinking_mode='chat'``): no reasoning, no effort line."""

    reasoning_effort = None
