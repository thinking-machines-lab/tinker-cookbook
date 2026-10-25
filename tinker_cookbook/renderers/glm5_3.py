"""
GLM-5.3 family renderers.

Includes:
- Glm5_3Renderer: GLM-5.3 with max reasoning effort (HF default)
- Glm5_3HighReasoningRenderer: GLM-5.3 with high reasoning effort
- Glm5_3LowReasoningRenderer: GLM-5.3 with low reasoning effort

Format (per the zai-org/GLM-5.3 HF chat template):

    [gMASK]<sop><|system|>Reasoning Effort: Max<|system|>{system prompt}
    <|user|>{question}<|assistant|><think>{reasoning}</think>{answer}

(shown with line breaks for readability; the actual format has none between messages)

Key format properties:

- The prompt starts with ``[gMASK]<sop>``, always followed by a
  ``<|system|>Reasoning Effort: ...`` segment. The template accepts
  ``reasoning_effort`` of ``low``/``high``/``max`` and renders anything else
  (including unset) as ``Max``. Unlike GLM-5.2 there is no ``enable_thinking``
  knob: thinking cannot be disabled at the template level, and generation
  prompts always end with an open ``<think>`` tag. Low reasoning effort is the
  closest replacement for a non-thinking mode.
- There is no per-message end token. An assistant turn is terminated by the
  next role token: ``<|user|>`` after a normal reply, ``<|observation|>`` after
  tool calls, or ``<|endoftext|>``. These are exactly the model's eos tokens in
  generation_config.json. Each assistant message owns its terminator: it is
  rendered as that message's output (so it carries the message's loss weight)
  and the following message drops the matching role token from its header. The
  last assistant message has no follower, so it gets one via
  ``RenderedMessage.stop_overlap`` instead. For every message but the last this
  changes only the chunk boundary: the token was already there as the next
  message's role token, so the sequence is exactly what the HF template
  produces. For the last message it appends one token the HF template does not
  emit -- the template stops after the assistant content -- which the model
  must nonetheless learn to produce.
- Assistant messages always begin with a ``<think>...</think>`` block. The HF
  template's ``clear_thinking`` defaults to False, so thinking content is
  preserved for ALL assistant messages by default (this renderer's
  ``strip_thinking_from_history=False`` default). Pass
  ``strip_thinking_from_history=True`` (HF ``clear_thinking=True``) to replace
  thinking with an empty ``<think></think>`` for assistant messages at or
  before the last user message. (GLM-5.2 had the opposite default.)
- The visible text of assistant messages is stripped of surrounding whitespace
  (the HF template applies ``.strip()``).
- Tool calls use an XML-ish key/value format, with string values rendered raw
  and non-string values rendered as JSON:

    <tool_call>{name}<arg_key>{key}</arg_key><arg_value>{value}</arg_value>...</tool_call>

- Consecutive tool response messages share a single ``<|observation|>`` role
  token, with each response wrapped in ``<tool_response>...</tool_response>``.
- Consecutive tool responses are re-sorted to match the order of the preceding
  assistant message's ``tool_calls`` when the match is unambiguous: every tool
  call and every tool response in the block must carry a unique id, and every
  response id must correspond to a tool call. Any ambiguity (missing ids,
  duplicates, or unknown ids) falls back to as-given message order. GLM-5.2
  always rendered tool responses in message order.

Generation prompts always end with ``<|assistant|><think>``. Because sampling
supplies that prefill at every turn, the same tokens are placed in the
zero-weight header of every assistant message rather than being trained on. For
the last assistant message this additionally makes the observation part of a
supervised example match ``build_generation_prompt`` exactly.

Reference: https://huggingface.co/zai-org/GLM-5.3/blob/main/chat_template.jinja
"""

import json
import re
from collections.abc import Iterator

import tinker
import torch

from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.renderers.base import (
    Content,
    Message,
    MessageDelta,
    ParseTermination,
    ReasoningStreamingParser,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    UnparsedToolCall,
    parse_content_blocks,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

# Matches a full GLM tool call block: <tool_call>{name}<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
_GLM_TOOL_CALL_RE = re.compile(
    r"^\s*<tool_call>\s*(?P<name>[\w\-\.]+)\s*(?P<body>.*?)\s*</tool_call>\s*$",
    re.DOTALL,
)
_GLM_ARG_PAIR_RE = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*<arg_value>(?P<value>.*?)</arg_value>",
    re.DOTALL,
)

# Tokens appended after <|assistant|> in the generation prompt. The GLM-5.3
# template always opens a think block (there is no non-thinking mode).
_THINK_PREFILL = "<think>"

# Tool declaration system message, matching the HF chat template's tools block.
# The leading newline is part of the message content: the template renders
# "<|system|>\n# Tools ..." while regular system messages render "<|system|>{content}".
_TOOLS_SYSTEM_TEMPLATE = """
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_lines}</tools>

For each function call, output the function name and arguments within the following XML format:
<tool_call>{{function-name}}<arg_key>{{arg-key-1}}</arg_key><arg_value>{{arg-value-1}}</arg_value><arg_key>{{arg-key-2}}</arg_key><arg_value>{{arg-value-2}}</arg_value>...</tool_call>"""


def _tool_declaration_json(tool: dict[str, object]) -> str:
    """Serialize one tool for the <tools> declaration block.

    Mirrors the template's ``tool_to_json`` macro: the ``strict`` and
    ``defer_loading`` keys are filtered out; the remaining keys are JSON-encoded
    in insertion order (json.dumps' default separators match Jinja's tojson).
    """
    filtered = {k: v for k, v in tool.items() if k not in ("defer_loading", "strict")}
    return json.dumps(filtered, ensure_ascii=False)


def _format_glm_arg_value(value: object) -> str:
    """Format a tool call argument value per the GLM template.

    String values are rendered raw; everything else is rendered as JSON
    (matching the template's ``v | tojson if v is not string else v``).
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _format_glm_tool_call(tool_call: ToolCall) -> str:
    """Format a single tool call in GLM's <arg_key>/<arg_value> format."""
    arguments = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
    parts = [f"<tool_call>{tool_call.function.name}"]
    for key, value in arguments.items():
        parts.append(
            f"<arg_key>{key}</arg_key><arg_value>{_format_glm_arg_value(value)}</arg_value>"
        )
    parts.append("</tool_call>")
    return "".join(parts)


def _parse_glm_tool_call(raw_text: str) -> ToolCall | UnparsedToolCall:
    """Parse a GLM-format tool call from a raw ``<tool_call>...</tool_call>`` block.

    Argument values are parsed as JSON when possible, falling back to raw
    strings otherwise. This mirrors how the template renders values (strings
    raw, non-strings as JSON); like other key/value formats, a string that
    happens to be valid JSON (e.g. ``"3"``) is indistinguishable from the
    typed value and parses as the typed value.
    """
    match = _GLM_TOOL_CALL_RE.match(raw_text)
    if not match:
        return UnparsedToolCall(raw_text=raw_text, error="Malformed GLM tool call")

    function_name = match.group("name")
    body = match.group("body")

    arguments: dict[str, object] = {}
    pos = 0
    for pair in _GLM_ARG_PAIR_RE.finditer(body):
        if body[pos : pair.start()].strip():
            return UnparsedToolCall(
                raw_text=raw_text,
                error="Unexpected content between tool call arguments",
            )
        key = pair.group("key").strip()
        if not key:
            return UnparsedToolCall(raw_text=raw_text, error="Empty tool call argument key")
        value_text = pair.group("value")
        try:
            value: object = json.loads(value_text)
        except json.JSONDecodeError:
            value = value_text
        arguments[key] = value
        pos = pair.end()

    if body[pos:].strip():
        return UnparsedToolCall(
            raw_text=raw_text,
            error="Unexpected trailing content inside tool call",
        )

    return ToolCall(
        function=ToolCall.FunctionBody(name=function_name, arguments=json.dumps(arguments))
    )


def _sort_tool_block(block: list[Message], prev_message: Message | None) -> list[Message]:
    """Sort one block of consecutive tool responses into tool-call order.

    Mirrors the HF template's ``can_sort`` machinery: the block is reordered to
    match the preceding assistant message's ``tool_calls`` order only when the
    mapping is unambiguous — every tool call has a unique non-empty id, every
    tool response has a unique non-empty ``tool_call_id``, and every response id
    matches one of the tool calls. Responses may cover a subset of the tool
    calls. On any ambiguity the block is returned in as-given order.
    """
    if prev_message is None or prev_message["role"] != "assistant":
        return block
    tool_calls = prev_message.get("tool_calls")
    if not tool_calls:
        return block

    call_ids = [tool_call.id for tool_call in tool_calls]
    if any(not call_id for call_id in call_ids) or len(set(call_ids)) != len(call_ids):
        return block

    response_ids = [message.get("tool_call_id") for message in block]
    if any(not response_id for response_id in response_ids) or len(set(response_ids)) != len(
        response_ids
    ):
        return block
    if not set(response_ids) <= set(call_ids):
        return block

    response_by_id = dict(zip(response_ids, block))
    return [response_by_id[call_id] for call_id in call_ids if call_id in response_by_id]


def _reorder_tool_results(messages: list[Message]) -> list[Message]:
    """Reorder consecutive tool responses to match tool-call order, per block.

    Returns a new message list; the input is not modified. See
    ``_sort_tool_block`` for the conditions under which a block is reordered.
    """
    result: list[Message] = []
    idx = 0
    while idx < len(messages):
        if messages[idx]["role"] != "tool":
            result.append(messages[idx])
            idx += 1
            continue
        block_end = idx
        while block_end + 1 < len(messages) and messages[block_end + 1]["role"] == "tool":
            block_end += 1
        prev_message = messages[idx - 1] if idx > 0 else None
        result.extend(_sort_tool_block(messages[idx : block_end + 1], prev_message))
        idx = block_end + 1
    return result


class Glm5_3Renderer(Renderer):
    """
    Renderer for GLM-5.3 with max reasoning effort.

    This matches HuggingFace's GLM-5.3 chat template default behavior
    (``reasoning_effort`` unset -> ``Max``, ``clear_thinking`` unset -> False).
    See the module docstring for format details.

    The default strip_thinking_from_history=False matches the HF template's
    ``clear_thinking=False`` default: thinking is preserved for all assistant
    messages, so the renderer has the extension property out of the box. Use
    strip_thinking_from_history=True (HF ``clear_thinking=True``) to replace
    thinking with empty ``<think></think>`` blocks for assistant messages at or
    before the last user message.
    """

    supports_streaming = True

    # Display string for the "Reasoning Effort" system segment, always emitted.
    # The template accepts low/high/max (anything else renders as Max).
    # Overridden by subclasses.
    reasoning_effort: str = "Max"

    def __init__(self, tokenizer: Tokenizer, strip_thinking_from_history: bool = False):
        """
        Args:
            tokenizer: The tokenizer to use for encoding.
            strip_thinking_from_history: When False (default), thinking is
                preserved in all assistant messages, matching the HF template's
                ``clear_thinking=False`` default. Set to True (HF
                ``clear_thinking=True``) to replace thinking with empty
                ``<think></think>`` blocks in assistant messages at or before
                the last user message.
        """
        super().__init__(tokenizer)
        self.strip_thinking_from_history = strip_thinking_from_history
        self._system_token = self._get_special_token("<|system|>")
        self._user_token = self._get_special_token("<|user|>")
        self._assistant_token = self._get_special_token("<|assistant|>")
        self._observation_token = self._get_special_token("<|observation|>")
        self._endoftext_token = self._get_special_token("<|endoftext|>")
        self._think_open_token = self._get_special_token("<think>")
        self._think_close_token = self._get_special_token("</think>")

    def _get_special_token(self, s: str) -> int:
        tokens = self.tokenizer.encode(s, add_special_tokens=False)
        assert len(tokens) == 1, f"Expected single token for {s}, got {tokens}"
        return tokens[0]

    @property
    def has_extension_property(self) -> bool:
        """Extension property depends on strip_thinking_from_history setting.

        When strip_thinking_from_history=False (default), thinking is preserved
        in history, so each successive observation is a prefix extension of the
        previous.

        When strip_thinking_from_history=True, thinking is replaced with empty
        ``<think></think>`` blocks once a new user message arrives, breaking
        the extension property.

        Note: the HF template strips surrounding whitespace from assistant text, so
        even with strip_thinking_from_history=False the extension property requires
        assistant text content without leading/trailing whitespace.
        """
        return not self.strip_thinking_from_history

    @property
    def _bos_tokens(self) -> list[int]:
        # The Reasoning Effort segment is always emitted (unlike GLM-5.2, which
        # dropped it when thinking was disabled).
        prefix = f"[gMASK]<sop><|system|>Reasoning Effort: {self.reasoning_effort}"
        return self.tokenizer.encode(prefix, add_special_tokens=False)

    @property
    def _stop_token_ids(self) -> tuple[int, int, int]:
        return (self._user_token, self._observation_token, self._endoftext_token)

    def _visible_text(self, content: Content) -> str:
        """Extract the visible text of a message, mirroring the template's visible_text macro."""
        if isinstance(content, str):
            return content
        texts = []
        for part in content:
            if part["type"] == "text":
                texts.append(part["text"])
            else:
                raise RendererError(
                    f"GLM-5.3 renderer does not support {part['type']!r} content parts here"
                )
        return "".join(texts)

    def _split_reasoning_and_text(self, message: Message) -> tuple[str | None, str]:
        """Split assistant content into (reasoning, visible text).

        Structured content: ThinkingPart -> reasoning, TextPart -> text.
        String content: split on embedded ``<think>...</think>`` tags exactly like
        the HF template does. Returns reasoning=None when the message has no
        thinking content.
        """
        content = message["content"]
        if isinstance(content, list):
            reasoning_parts = []
            text_parts = []
            for part in content:
                if part["type"] == "thinking":
                    reasoning_parts.append(part["thinking"])
                elif part["type"] == "text":
                    text_parts.append(part["text"])
                else:
                    raise RendererError(
                        f"GLM-5.3 renderer does not support {part['type']!r} content parts"
                    )
            reasoning = "".join(reasoning_parts) if reasoning_parts else None
            return reasoning, "".join(text_parts)
        if "</think>" in content:
            reasoning = content.split("</think>")[0].split("<think>")[-1]
            return reasoning, content.split("</think>")[-1]
        return None, content

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        """Build a sampling prompt, re-sorting tool responses like the HF template.

        See :meth:`Renderer.build_generation_prompt`. Consecutive tool
        responses are reordered to match the preceding assistant message's
        ``tool_calls`` order when ids allow an unambiguous match (see the
        module docstring).
        """
        return super().build_generation_prompt(_reorder_tool_results(messages), role, prefill)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """Build a supervised example, re-sorting tool responses like the HF template.

        See :meth:`Renderer.build_supervised_example`. Consecutive tool
        responses are reordered to match the preceding assistant message's
        ``tool_calls`` order when ids allow an unambiguous match (see the
        module docstring).
        """
        return super().build_supervised_example(_reorder_tool_results(messages), train_on_what)

    def _turn_terminator_token(self, message: Message, next_message: Message | None) -> int | None:
        """The token the model must emit to end this assistant turn.

        ``<|observation|>`` when handing off to tools, ``<|user|>`` otherwise.

        For a message that is followed by another, this is only meaningful when
        the next message's role token is the one the model would have had to
        produce. A mid-conversation ``system`` message, or a tool response after
        a turn that made no tool calls, is not something the model ends its turn
        with, so ``None`` is returned and the token stays where it was -- on the
        next message's header. That keeps the token sequence identical in every
        case; only which chunk owns the token changes.

        Args:
            message (Message): The assistant message being terminated.
            next_message (Message | None): The message that follows it, or None
                when it is the last one.

        Returns:
            int | None: The terminator token id, or None when the next message
                does not consume one.
        """
        expected = self._observation_token if message.get("tool_calls") else self._user_token
        if next_message is None:
            return expected
        next_header = {"user": self._user_token, "tool": self._observation_token}.get(
            next_message["role"]
        )
        return expected if next_header == expected else None

    def _header_supplied_by_previous_turn(self, message: Message, ctx: RenderContext) -> bool:
        """Whether the preceding assistant turn already emitted this message's role token.

        Mirrors :meth:`_turn_terminator_token` exactly, so the token is written
        once and only once.
        """
        prev = ctx.prev_message
        if prev is None or prev["role"] != "assistant":
            return False
        return self._turn_terminator_token(prev, message) is not None

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render a chat message into GLM-5.3 role-token chunks.

        GLM-5.3 has no per-message end token: an assistant turn ends when the
        next role token appears. That token is part of what the model must
        generate, so it is emitted as the assistant message's own output rather
        than as the next message's (zero-weight) header -- otherwise only the
        final turn of a multi-turn supervised example would ever be trained to
        stop. The following message then suppresses its header, leaving the
        token sequence unchanged.

        For the same reason the ``<think>`` prefill is moved into the zero-weight
        header of *every* assistant message, not just the last: sampling always
        supplies it, at every turn.

        Args:
            message (Message): The chat message to render.
            ctx (RenderContext): Positional context. ``last_user_index`` controls
                thinking preservation, ``prev_message`` controls ``<|observation|>``
                grouping for consecutive tool responses and header suppression,
                and ``next_message`` selects the turn terminator.

        Returns:
            RenderedMessage: Header, output, and (for the last assistant message)
                stop_overlap token chunks.
        """
        role = message["role"]
        stop_overlap: tinker.types.EncodedTextChunk | None = None
        trailing_tokens: list[int] = []

        if role == "system":
            header_tokens = [self._system_token]
            output_str = self._visible_text(message["content"])
        elif role == "user":
            header_tokens = (
                [] if self._header_supplied_by_previous_turn(message, ctx) else [self._user_token]
            )
            output_str = self._visible_text(message["content"])
        elif role == "assistant":
            reasoning, text = self._split_reasoning_and_text(message)
            keep_thinking = not self.strip_thinking_from_history or ctx.idx > ctx.last_user_index
            if keep_thinking and reasoning is not None:
                think_block = f"<think>{reasoning}</think>"
            else:
                think_block = "<think></think>"
            output_str = think_block + text.strip()

            tool_calls = message.get("tool_calls")
            if tool_calls:
                output_str += "".join(_format_glm_tool_call(tool_call) for tool_call in tool_calls)

            header_tokens = [self._assistant_token]
            # Sampling always prefills <think>, at every turn, so move it into the
            # zero-weight header rather than training the model to produce a token
            # it is always given. For the last message this also makes the
            # observation part of a supervised example match build_generation_prompt.
            if output_str.startswith(_THINK_PREFILL):
                header_tokens += self.tokenizer.encode(_THINK_PREFILL, add_special_tokens=False)
                output_str = output_str[len(_THINK_PREFILL) :]

            terminator = self._turn_terminator_token(message, ctx.next_message)
            if ctx.is_last:
                # No following message to carry the terminator; build_supervised_example
                # appends it via stop_overlap.
                stop_overlap = (
                    tinker.types.EncodedTextChunk(tokens=[terminator])
                    if terminator is not None
                    else None
                )
            elif terminator is not None:
                # Own the terminator so it carries this message's loss weight; the
                # next message drops the matching header.
                trailing_tokens = [terminator]
        elif role == "tool":
            # Consecutive tool responses share a single <|observation|> role token,
            # as does a tool response following the turn that called it.
            follows_tool = ctx.prev_message is not None and ctx.prev_message["role"] == "tool"
            if follows_tool or self._header_supplied_by_previous_turn(message, ctx):
                header_tokens = []
            else:
                header_tokens = [self._observation_token]
            output_str = (
                "<tool_response>" + self._visible_text(message["content"]) + "</tool_response>"
            )
        else:
            raise RendererError(f"Unsupported role: {role}")

        output_tokens = self.tokenizer.encode(output_str, add_special_tokens=False)
        output_tokens += trailing_tokens
        output: list[tinker.ModelInputChunk] = (
            [tinker.types.EncodedTextChunk(tokens=output_tokens)] if output_tokens else []
        )
        header = tinker.types.EncodedTextChunk(tokens=header_tokens) if header_tokens else None
        return RenderedMessage(header=header, output=output, stop_overlap=stop_overlap)

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        """Return the generation suffix: the role token plus the thinking prefill.

        For assistant generation this is ``<|assistant|><think>``, matching the
        HF template's add_generation_prompt behavior (GLM-5.3 always opens a
        think block). A custom prefill passed to build_generation_prompt is
        appended after this suffix.
        """
        role_tokens = {
            "system": self._system_token,
            "user": self._user_token,
            "assistant": self._assistant_token,
            "tool": self._observation_token,
        }
        if role not in role_tokens:
            raise RendererError(f"Unsupported generation role: {role}")
        suffix = [role_tokens[role]]
        if role == "assistant":
            suffix += self.tokenizer.encode(_THINK_PREFILL, add_special_tokens=False)
        return suffix

    def get_stop_sequences(self) -> list[int]:
        """Return stop sequences for GLM-5.3 generation.

        Returns:
            list[int]: The ``<|user|>``, ``<|observation|>``, and ``<|endoftext|>``
                token IDs — the model's eos tokens per generation_config.json.
        """
        return [self._user_token, self._observation_token, self._endoftext_token]

    def _normalize_response_tokens(self, response: list[int]) -> list[int]:
        """Restore the prefilled ``<think>`` token before parsing sampled tokens.

        The generation prompt ends with an open ``<think>`` tag, so sampled tokens
        look like ``reasoning</think>answer``. Prepend ``<think>`` so the parser
        sees a complete think block.
        """
        if (
            response
            and response[0] != self._think_open_token
            and self._think_close_token in response
        ):
            return [self._think_open_token, *response]
        return response

    def _parse_response_content(
        self, response: list[int], *, allow_missing_stop: bool
    ) -> tuple[Message, ParseTermination]:
        """Shared parsing logic for both batch and streaming paths.

        Callers are responsible for normalization — this method does NOT call
        ``_normalize_response_tokens``.
        """
        stop_ids = self._stop_token_ids
        stop_positions = [i for i, token in enumerate(response) if token in stop_ids]
        if len(stop_positions) == 0:
            content = str(self.tokenizer.decode(response))
            termination = ParseTermination.MALFORMED
            if not allow_missing_stop:
                return Message(role="assistant", content=content), termination
        elif len(stop_positions) == 1:
            stop_index = stop_positions[0]
            content = str(self.tokenizer.decode(response[:stop_index]))
            # <|endoftext|> is the model's generic EOS; <|user|>/<|observation|>
            # are the expected end-of-turn signals.
            termination = (
                ParseTermination.EOS
                if response[stop_index] == self._endoftext_token
                else ParseTermination.STOP_SEQUENCE
            )
        else:
            raise RendererError(
                f"When parsing response, expected to split into 1 or 2 pieces using stop tokens, but got {len(stop_positions)}. "
                "You probably are using the wrong stop tokens when sampling"
            )

        message = Message(role="assistant", content=content)

        # Parse <think>...</think> and <tool_call>...</tool_call> blocks together.
        # parse_content_blocks assumes JSON tool call payloads, so GLM-format tool
        # calls come back as UnparsedToolCall and are re-parsed here.
        result = parse_content_blocks(content)
        if result is not None:
            parts, tool_results = result
            message["content"] = parts

            tool_calls = [t for t in tool_results if isinstance(t, ToolCall)]
            unparsed: list[UnparsedToolCall] = []
            for tool_result in tool_results:
                if isinstance(tool_result, UnparsedToolCall):
                    parsed = _parse_glm_tool_call(tool_result.raw_text)
                    if isinstance(parsed, ToolCall):
                        tool_calls.append(parsed)
                    else:
                        unparsed.append(parsed)
            if tool_calls:
                message["tool_calls"] = tool_calls
            if unparsed:
                message["unparsed_tool_calls"] = unparsed

        return message, termination

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        """Parse sampled token IDs back into an assistant Message.

        Restores the ``<think>`` prefill, strips the terminating stop token, and
        parses ``<think>...</think>`` blocks into ThinkingPart and GLM-format
        ``<tool_call>...</tool_call>`` blocks into ToolCall objects.

        Args:
            response (list[int]): Raw token IDs from the sampler.

        Returns:
            tuple[Message, ParseTermination]: ``STOP_SEQUENCE`` when terminated by
                ``<|user|>`` or ``<|observation|>``, ``EOS`` when terminated by
                ``<|endoftext|>``, ``MALFORMED`` when no stop token was found.
        """
        response = self._normalize_response_tokens(response)
        return self._parse_response_content(response, allow_missing_stop=False)

    def _parse_response_for_streaming(
        self, response: list[int]
    ) -> tuple[Message, ParseTermination]:
        """Parse response for streaming, always applying full content parsing.

        Unlike parse_response which short-circuits on missing stop token, this
        always parses think blocks and tool calls so the final Message emitted
        by streaming is complete even for truncated responses.
        """
        return self._parse_response_content(response, allow_missing_stop=True)

    def parse_response_streaming(self, response: list[int]) -> Iterator[MessageDelta]:
        """Parse response tokens with streaming, yielding incremental deltas.

        GLM-5.3 has multiple stop tokens (``<|user|>``, ``<|observation|>``,
        ``<|endoftext|>``), so this overrides the single-stop-token default with
        whichever stop token actually appears in the response.
        """
        response = self._normalize_response_tokens(response)
        end_token = next(
            (token for token in response if token in self._stop_token_ids), self._user_token
        )
        parser = ReasoningStreamingParser(
            tokenizer=self.tokenizer,
            end_message_token=end_token,
            parse_final_response=self._parse_response_for_streaming,
        )
        for token in response:
            yield from parser.feed(token)
        yield from parser.finish()

    def to_openai_message(self, message: Message) -> dict:
        """Convert a Message to OpenAI API format with reasoning_content for thinking.

        The GLM-5.3 HF template reads ``reasoning_content`` for thinking and
        iterates ``tool_calls[].function.arguments.items()``, so thinking is
        emitted as a separate field and tool call arguments as a dict.
        """
        result: dict = {"role": message["role"]}

        content = message["content"]
        if isinstance(content, str):
            result["content"] = content
        else:
            thinking_parts = []
            text_parts = []
            for part in content:
                if part["type"] == "thinking":
                    thinking_parts.append(part["thinking"])
                elif part["type"] == "text":
                    text_parts.append(part["text"])
                else:
                    raise RendererError(
                        f"GLM-5.3 renderer does not support {part['type']!r} content parts"
                    )
            result["content"] = "".join(text_parts)
            if thinking_parts:
                result["reasoning_content"] = "".join(thinking_parts)

        if "tool_calls" in message and message["tool_calls"]:  # noqa: RUF019
            result["tool_calls"] = [
                {
                    "type": "function",
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        # The GLM template iterates arguments as a mapping.
                        "arguments": json.loads(tc.function.arguments)
                        if tc.function.arguments
                        else {},
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
        """Create system messages with GLM-5.3 tool specifications.

        The GLM-5.3 HF template renders tool declarations in a dedicated
        ``<|system|>`` segment that comes BEFORE the conversation's regular
        system message, so this returns up to two system messages: the tools
        declaration first, then the system prompt.

        Reference: https://huggingface.co/zai-org/GLM-5.3/blob/main/chat_template.jinja
        """
        messages: list[Message] = []
        if tools:
            # One JSON object per line. Per the template, tools with a truthy
            # defer_loading are skipped entirely (but the tools block is still
            # rendered), and the strict/defer_loading keys are never serialized.
            included = [t for t in (dict(tool) for tool in tools) if not t.get("defer_loading")]
            tool_lines = "".join(_tool_declaration_json(tool) + "\n" for tool in included)
            messages.append(
                Message(role="system", content=_TOOLS_SYSTEM_TEMPLATE.format(tool_lines=tool_lines))
            )
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        return messages


class Glm5_3HighReasoningRenderer(Glm5_3Renderer):
    """
    Renderer for GLM-5.3 with HIGH reasoning effort.

    Matches the HF template with ``reasoning_effort='high'``: the prompt prefix
    contains ``<|system|>Reasoning Effort: High`` instead of the default
    ``Reasoning Effort: Max``.
    """

    reasoning_effort = "High"


class Glm5_3LowReasoningRenderer(Glm5_3Renderer):
    """
    Renderer for GLM-5.3 with LOW reasoning effort.

    Matches the HF template with ``reasoning_effort='low'``: the prompt prefix
    contains ``<|system|>Reasoning Effort: Low`` instead of the default
    ``Reasoning Effort: Max``. GLM-5.3 has no template-level way to disable
    thinking (generation prompts always open a ``<think>`` block); low effort
    is the closest replacement for GLM-5.2's disable-thinking mode.
    """

    reasoning_effort = "Low"
