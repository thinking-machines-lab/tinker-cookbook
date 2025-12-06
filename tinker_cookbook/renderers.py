"""
Use viz_sft_dataset to visualize the output of different renderers. E.g.,
    python -m tinker_cookbook.supervised.viz_sft_dataset dataset_path=Tulu3Builder renderer_name=role_colon
"""

import json
import logging
import re
from datetime import datetime
from enum import StrEnum
from typing import Callable, Literal, NotRequired, TypedDict

import tinker
import torch
import pydantic

from tinker_cookbook.tokenizer_utils import Tokenizer

logger = logging.getLogger(__name__)

# Tool types are based on kosong (https://github.com/MoonshotAI/kosong).


class StrictBase(pydantic.BaseModel):
    """
    Pydantic base class that's immutable and doesn't silently ignore extra fields.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    def __str__(self) -> str:
        return repr(self)


class ToolCall(StrictBase):
    """
    Structured tool invocation following OpenAI/kosong format.

    This represents a request to invoke a tool/function. The structure follows
    the OpenAI function calling format for compatibility with various LLM APIs.

    Example:
        tool_call = ToolCall(
            function=ToolCall.FunctionBody(
                name="search",
                arguments='{"query_list": ["python async", "pydantic validation"]}'
            ),
            id="call_abc123"
        )
    """

    class FunctionBody(pydantic.BaseModel):
        """
        Tool call function body containing the tool name and arguments.

        The arguments field must be a valid JSON string that will be parsed
        by the tool implementation.
        """

        name: str
        """The name of the tool to be called."""
        arguments: str
        """Arguments of the tool call in JSON string format."""

    type: Literal["function"] = "function"
    """Tool call type, must be 'function' for compatibility."""

    id: str | None = None
    """Optional unique identifier for tracking this specific tool call."""

    function: FunctionBody
    """The function body containing tool name and arguments."""


class ToolOk(StrictBase):
    """
    Successful tool execution result.

    Used to indicate that a tool call completed successfully, with
    the main output and optional metadata fields.
    """

    output: str
    """The main output/result from the tool execution."""

    message: str = ""
    """Optional human-readable message about the execution."""

    brief: str = ""
    """Optional brief summary of the result for logging."""


class ToolError(StrictBase):
    """
    Tool execution error result.

    Used to indicate that a tool call failed or encountered an error,
    with details about what went wrong.
    """

    output: str = ""
    """Any partial output that was generated before the error."""

    message: str = ""
    """Error message describing what went wrong."""

    brief: str = ""
    """Brief error summary for logging."""


ToolReturnType = ToolOk | ToolError
"""Union type for tool execution results - either success or error."""


class ToolResult(StrictBase):
    """
    Complete tool execution result with tracking ID.

    Wraps the actual result (ToolOk or ToolError) with the corresponding
    tool call ID for correlation in multi-tool scenarios.

    Note: This class is defined for future use in handling multiple
    concurrent tool calls with result correlation.
    """

    tool_call_id: str | None
    """ID of the tool call this result corresponds to."""

    result: ToolReturnType
    """The actual execution result (success or error)."""


# NOTE: we use a broad type definition for the role to be flexible
# Common roles are "user", "assistant", "system", "tool"
Role = str


class Message(TypedDict):
    role: Role
    content: str
    tool_calls: NotRequired[list[ToolCall]]
    thinking: NotRequired[str]
    trainable: NotRequired[bool]
    tool_call_id: NotRequired[str]
    name: NotRequired[str]


def _tool_call_payload(tool_call: ToolCall) -> dict[str, object]:
    """Minimal JSON payload for embedding in <tool_call> blocks."""
    # Convert from nested structure to flat format for compatibility
    return {
        "name": tool_call.function.name,
        "args": json.loads(tool_call.function.arguments),
    }


class TrainOnWhat(StrEnum):
    LAST_ASSISTANT_MESSAGE = "last_assistant_message"
    ALL_ASSISTANT_MESSAGES = "all_assistant_messages"
    ALL_MESSAGES = "all_messages"
    ALL_TOKENS = "all_tokens"
    ALL_USER_AND_SYSTEM_MESSAGES = "all_user_and_system_messages"
    CUSTOMIZED = "customized"


class Renderer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        raise NotImplementedError

    def get_stop_sequences(self) -> list[str] | list[int]:
        raise NotImplementedError

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        raise NotImplementedError


def tokens_weights_from_strings_weights(
    strings_weights: list[tuple[str, float]],
    tokenizer: Tokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    strings, weights = zip(*strings_weights, strict=True)
    token_chunks = [tokenizer.encode(s, add_special_tokens=i == 0) for i, s in enumerate(strings)]
    weights = torch.cat(
        [torch.full((len(chunk),), w) for chunk, w in zip(token_chunks, weights, strict=True)]
    )
    tokens = torch.cat([torch.tensor(chunk) for chunk in token_chunks])
    assert tokens.dtype == torch.int64
    return tokens, weights


def build_supervised_example(
    start_tokens: list[int],
    render_message: Callable[[int, Message], tuple[list[int], list[int], list[int]]],
    messages: list[Message],
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates tokens and weights (for SFT) in the most standard way; by concatenating
    together tokens and weights for each message.

    Args:
        start_tokens: a list of tokens that are added at the beginning of the sequence.
        render_message: a function that takes an index and a message and returns a tuple of three lists of tokens:
            - ob_part: tokens for the observation part of the message
            - action_part: tokens for the action part of the message
            - action_tail: tokens that are generated by the assistant in this message, which are also
                part of the ob part of the next message. (Only relevant for some renderers, such as RoleColonRenderer)
        train_on_what: an enum that controls how the weights are assigned to the tokens.
            - TrainOnWhat.LAST_ASSISTANT_MESSAGE: only the last assistant message is used for training
            - TrainOnWhat.ALL_ASSISTANT_MESSAGES: all assistant messages are used for training
            - TrainOnWhat.ALL_MESSAGES: all messages are used for training
            - TrainOnWhat.ALL_TOKENS: all tokens are used for training
            - TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES: all user and system messages are used for training
            - TrainOnWhat.CUSTOMIZED: each message has a trainable field, and the weights are assigned based on the trainable field
        messages: a list of messages to render.

    Returns:
        A tuple of two tensors:
            - tokens: a tensor of tokens
            - weights: a tensor of weights
    """
    tokens_weights = [(token, 0) for token in start_tokens]
    for idx, message in enumerate(messages):
        if train_on_what == TrainOnWhat.CUSTOMIZED:
            assert "trainable" in message, (
                "When using CUSTOMIZED train_on_what, each message must have a trainable field: True if loss is applied on this message, False otherwise"
            )
        else:
            assert "trainable" not in message, (
                "When using non-CUSTOMIZED train_on_what, each message must not have a trainable field. Either change train_on_what to CUSTOMIZED or remove the trainable field from the message"
            )

        is_last_message = idx == len(messages) - 1
        is_assistant = message["role"] == "assistant"
        is_user_or_system = message["role"] in ["user", "system"]

        # only apply weight to observation part if train_on_what is ALL_TOKENS
        ob_part, action_part, action_tail = render_message(idx, message)
        ob_weight = int(train_on_what == TrainOnWhat.ALL_TOKENS)
        tokens_weights += [(token, ob_weight) for token in ob_part]

        action_tokens = action_part
        # action tail is effectively the stop_token and the start token for the next turn
        # e.g. \n\nUser:
        if is_last_message:
            action_tokens += action_tail

        match train_on_what:
            case TrainOnWhat.LAST_ASSISTANT_MESSAGE:
                action_has_weight = is_last_message and is_assistant
            case TrainOnWhat.ALL_ASSISTANT_MESSAGES:
                action_has_weight = is_assistant
            case TrainOnWhat.ALL_MESSAGES:
                action_has_weight = True
            case TrainOnWhat.ALL_TOKENS:
                action_has_weight = True
            case TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES:
                action_has_weight = is_user_or_system
            case TrainOnWhat.CUSTOMIZED:
                action_has_weight = message.get("trainable", False)
            case _:
                raise ValueError(f"Unknown train_on_what: {train_on_what}")

        tokens_weights += [(token, int(action_has_weight)) for token in action_tokens]

    tokens, weights = zip(*tokens_weights, strict=True)
    return torch.tensor(tokens), torch.tensor(weights)


def parse_response_for_stop_token(
    response: list[int], tokenizer: Tokenizer, stop_token: int
) -> tuple[Message, bool]:
    """Parse response for a single stop token.

    We expect a properly rendered response to have exactly one stop token; but it may have zero if e.g. the model
    ran out of tokens when sampling, which will incur a format error. If there are > 1, there is likely a bug in the
    sampler and we should error.
    """
    emt_count = response.count(stop_token)
    if emt_count == 0:
        str_response = tokenizer.decode(response)
        logger.debug(f"Response is not a valid assistant response: {str_response}")
        return Message(role="assistant", content=str_response), False
    elif emt_count == 1:
        str_response = tokenizer.decode(response[: response.index(stop_token)])
        return Message(role="assistant", content=str_response), True
    else:
        raise ValueError(
            f"When parsing response, expected to split into 1 or 2 pieces using stop tokens, but got {emt_count}. "
            "You probably are using the wrong stop tokens when sampling"
        )


class RoleColonRenderer(Renderer):
    """
    format like this:
        User: <content>

        Assistant: <content>

    This is basically the format used by DeepSeek, and similar to the format used by Anthropic,
    except that they use "Human" instead of "User".
    """

    def _render_message(self, message: Message) -> tuple[list[int], list[int], list[int]]:
        assert message.get("thinking") is None, "Thinking tokens not supported in RoleColonRenderer"
        ob_str = message["role"].capitalize() + ":"
        # Observation (prompt) part
        ac_str = " " + message["content"] + "\n\n"
        # Action part
        ac_tail_str = "User:" if message["role"] == "assistant" else "<UNUSED>"
        # Action part that's only included in the last message in SFT
        return (
            self.tokenizer.encode(ob_str, add_special_tokens=False),
            self.tokenizer.encode(ac_str, add_special_tokens=False),
            self.tokenizer.encode(ac_tail_str, add_special_tokens=False),
        )

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        tokens: list[int] = []
        tokens.extend(self._bos_tokens)
        for message in messages:
            ob_part, action_part, action_tail = self._render_message(message)
            tokens.extend(ob_part)
            tokens.extend(action_part)
        new_partial_message = Message(role=role, content="")
        ob_part, _action_part, _action_tail = self._render_message(new_partial_message)
        tokens.extend(ob_part)
        tokens.extend(self.tokenizer.encode(prefill or "", add_special_tokens=False))
        return tinker.ModelInput.from_ints(tokens)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get tokens and weights for action corresponding to final message
        """
        return build_supervised_example(
            self._bos_tokens,
            lambda _idx, message: self._render_message(message),
            messages,
            train_on_what,
        )

    def get_stop_sequences(self) -> list[str]:
        return ["\n\nUser:"]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        str_response = self.tokenizer.decode(response)
        splitted = str_response.split("\n\nUser:")
        if len(splitted) == 1:
            logger.debug(f"Response is not a valid assistant response: {str_response}")
            return Message(role="assistant", content=str_response.strip()), False
        elif len(splitted) == 2:
            before, _after = splitted
            return Message(role="assistant", content=before.strip()), True
        else:
            raise ValueError(
                f"When parsing response, expected to split into 1 or 2 pieces using stop tokens, but got {len(splitted)}. "
                "You probably are using the wrong stop tokens when sampling"
            )

    @property
    def _bos_tokens(self) -> list[int]:
        bos_token_str = self.tokenizer.bos_token
        if bos_token_str is None:
            return []
        assert isinstance(bos_token_str, str)
        return self.tokenizer.encode(bos_token_str, add_special_tokens=False)


class Llama3Renderer(Renderer):
    """
    Format like this:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>

        What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """

    def _render_message(self, message: Message) -> tuple[list[int], list[int], list[int]]:
        assert message.get("thinking") is None, "CoT tokens not supported in Llama3"
        ob_str = f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
        # Observation (prompt) part
        ac_str = f"{message['content']}<|eot_id|>"
        # Action part
        ac_tail_str = ""  # No action tail needed for Llama3 format
        # Action part that's only included in the last message in SFT
        return (
            self.tokenizer.encode(ob_str, add_special_tokens=False),
            self.tokenizer.encode(ac_str, add_special_tokens=False),
            self.tokenizer.encode(ac_tail_str, add_special_tokens=False),
        )

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        tokens: list[int] = []
        tokens.extend(self._bos_tokens)
        for message in messages:
            ob_part, action_part, action_tail = self._render_message(message)
            tokens.extend(ob_part)
            tokens.extend(action_part)
        new_partial_message = Message(role=role, content="")
        ob_part, _action_part, _action_tail = self._render_message(new_partial_message)
        tokens.extend(ob_part)
        tokens.extend(self.tokenizer.encode(prefill or "", add_special_tokens=False))
        return tinker.ModelInput.from_ints(tokens)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get tokens and weights for action corresponding to final message
        """
        return build_supervised_example(
            self._bos_tokens,
            lambda _idx, message: self._render_message(message),
            messages,
            train_on_what,
        )

    @property
    def _bos_tokens(self) -> list[int]:
        return self.tokenizer.encode("<|begin_of_text|>", add_special_tokens=False)

    @property
    def _end_message_token(self) -> int:
        (token,) = self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)
        return token

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        return parse_response_for_stop_token(response, self.tokenizer, self._end_message_token)


class Qwen3Renderer(Renderer):
    """
    Format like this:
        <|im_start|>system
        You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
        <|im_start|>user
        What can you help me with?<|im_end|>
        <|im_start|>assistant
        <think>

        </think>
        I can help you with...<|im_end|>
    """

    def __init__(self, tokenizer: Tokenizer, strip_thinking_from_history: bool = True):
        """
        Args:
            tokenizer: The tokenizer to use for encoding.
            strip_thinking_from_history: When True (default), strips <think>...</think> blocks
                from assistant messages in multi-turn history. This matches how Qwen3 models
                were trained - they only see their own thinking during the current turn, not
                from previous turns. Set to False to preserve thinking in history (useful for
                certain RL scenarios where you want the extension property for efficiency).

        See https://tinker-docs.thinkingmachines.ai/rl/sequence-extension for details on
        how this option affects multi-turn RL compute efficiency.
        """
        super().__init__(tokenizer)
        self.strip_thinking_from_history = strip_thinking_from_history

    def _render_message(self, idx: int, message: Message) -> tuple[list[int], list[int], list[int]]:
        assert message.get("thinking") is None, "TODO: support CoT in Qwen3 renderer"
        maybe_newline = "\n" if idx > 0 else ""
        ob_str = f"{maybe_newline}<|im_start|>{message['role']}\n"
        ac_content = message["content"]
        if (
            self.strip_thinking_from_history
            and message["role"] == "assistant"
            and "</think>" in ac_content
        ):
            # Multi-turn conversation, we remove the thinking section from the assistant message.
            # This matches how Qwen3 models were trained - they only see their own thinking
            # during the current turn, not from previous turns.
            ac_content = ac_content.split("</think>")[1].lstrip()
        elif message["role"] == "assistant" and "<think>" not in ac_content:
            # Matching the paper, we force the assistant to start with <think>. Some SFT datasets include
            # <think> in the assistant messages, we so don't need to re-add it in those cases.
            ob_str += "<think>\n"
        # Observation (prompt) part
        if "tool_calls" in message:
            ac_content += "\n".join(
                [
                    f"<tool_call>\n{json.dumps(_tool_call_payload(tool_call))}\n</tool_call>"
                    for tool_call in message["tool_calls"]
                ]
            )
        ac_content += "<|im_end|>"
        # Action part
        ac_tail_str = ""  # No action tail needed for Qwen format
        # Action part that's only included in the last message in SFT
        return (
            self.tokenizer.encode(ob_str, add_special_tokens=False),
            self.tokenizer.encode(ac_content, add_special_tokens=False),
            self.tokenizer.encode(ac_tail_str, add_special_tokens=False),
        )

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        tokens: list[int] = []  # No BOS token for Qwen
        for idx, message in enumerate(messages):
            ob_part, action_part, _ = self._render_message(idx, message)
            tokens.extend(ob_part)
            tokens.extend(action_part)
        # Add generation prompt
        new_partial_message = Message(role=role, content="")
        ob_part, _, _ = self._render_message(len(messages), new_partial_message)
        tokens.extend(ob_part)
        tokens.extend(self.tokenizer.encode(prefill or "", add_special_tokens=False))
        return tinker.ModelInput.from_ints(tokens)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get tokens and weights for action corresponding to final message.
        """
        return build_supervised_example([], self._render_message, messages, train_on_what)

    @property
    def _end_message_token(self) -> int:
        tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        assert len(tokens) == 1, f"Expected single token for <|im_end|>, got {len(tokens)}"
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def _parse_tool_call(self, tool_call_str: str) -> list[ToolCall] | None:
        try:
            tool_call = json.loads(tool_call_str)
        except json.JSONDecodeError:
            return None

        if not isinstance(tool_call, dict):
            return None
        name = tool_call.get("name")
        args = tool_call.get("args")
        tool_id = tool_call.get("id")
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        if tool_id is not None and not isinstance(tool_id, str):
            tool_id = None
        # Convert to nested structure with arguments as JSON string
        return [
            ToolCall(
                function=ToolCall.FunctionBody(name=name, arguments=json.dumps(args)),
                id=tool_id,
            )
        ]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            return assistant_message, False

        # Follow Qwen docs and Qwen-Agent's tool calling prompt to use <tool_call>...</tool_call> tags to wrap the tool call.
        # - https://qwen.readthedocs.io/en/latest/getting_started/concepts.html#tool-calling
        # - https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/llm/fncall_prompts/nous_fncall_prompt.py#L279-L282
        match = re.search(r"<tool_call>(.*?)</tool_call>", assistant_message["content"], re.DOTALL)
        if match:
            tool_calls = self._parse_tool_call(match.group(1))
            if tool_calls is None:
                return assistant_message, False
            else:
                assistant_message["tool_calls"] = tool_calls
                return assistant_message, True
        return assistant_message, True


class Qwen3DisableThinkingRenderer(Qwen3Renderer):
    """
    Renderer that disables thinking for hybrid-mode Qwen3 models
    """

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        prefill = "\n</think>\n\n" + (prefill or "")
        # XXX this causes inefficiency in RL, because the observations don't grow by appending to the end.
        # Maybe we should just insert this empty thinking block in every message?
        return super().build_generation_prompt(messages, role, prefill)


class Qwen3InstructRenderer(Qwen3Renderer):
    """
    Renderer for Qwen3 instruct 2507 models. Unlike the earlier Qwen3 models, these models do not
    use the <think> tag at all.
    """

    def _render_message(self, idx: int, message: Message) -> tuple[list[int], list[int], list[int]]:
        assert message.get("thinking") is None, "CoT tokens not supported in Qwen3 instruct 2507"
        maybe_newline = "\n" if idx > 0 else ""
        ob_str = f"{maybe_newline}<|im_start|>{message['role']}\n"
        ac_content = message["content"]
        # Observation (prompt) part
        if "tool_calls" in message:
            ac_content += "\n".join(
                [
                    f"<tool_call>\n{json.dumps(_tool_call_payload(tool_call))}\n</tool_call>"
                    for tool_call in message["tool_calls"]
                ]
            )
        ac_content += "<|im_end|>"
        # Action part
        ac_tail_str = ""  # No action tail needed for Qwen format
        # Action part that's only included in the last message in SFT
        return (
            self.tokenizer.encode(ob_str, add_special_tokens=False),
            self.tokenizer.encode(ac_content, add_special_tokens=False),
            self.tokenizer.encode(ac_tail_str, add_special_tokens=False),
        )


class DeepSeekV3Renderer(Renderer):
    """
    Format like this (no newlines between messages):
        <|begin_of_sentence|><|User|>What can you help me with?<|Assistant|><think>Thinking...</think>I can help you with...<|end_of_sentence|>
    For no-think, just use <|Assistant|></think>
    Deepseek renderer does not support the system role out of the box. You can set system_role_as_user to True to automatically convert the system role to the user role.
    """

    def __init__(self, tokenizer: Tokenizer, system_role_as_user: bool = False):
        super().__init__(tokenizer)
        self.system_role_as_user = system_role_as_user

    def _render_message(self, message: Message) -> tuple[list[int], list[int], list[int]]:
        assert message.get("thinking") is None, "TODO: support CoT in DsV3 renderer"
        if message["role"] == "user" or (self.system_role_as_user and message["role"] == "system"):
            role_token = self._get_special_token("User")
        elif message["role"] == "assistant":
            role_token = self._get_special_token("Assistant")
        else:
            raise ValueError(f"Unsupported role: {message['role']}")
        ob = [role_token]
        ac = self.tokenizer.encode(message["content"], add_special_tokens=False)

        if message["role"] == "assistant":  # end_of_message only for assistant in dsv3
            ac.append(self._end_message_token)
        # Action part that's only included in the last message in SFT
        ac_tail = []  # No action tail needed for DsV3 format
        return (ob, ac, ac_tail)

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        tokens: list[int] = []
        tokens.extend(self._bos_tokens)
        for message in messages:
            ob_part, action_part, action_tail = self._render_message(message)
            tokens.extend(ob_part)
            tokens.extend(action_part)
        new_partial_message = Message(role=role, content="")
        ob_part, _action_part, _action_tail = self._render_message(new_partial_message)
        tokens.extend(ob_part)
        tokens.extend(self.tokenizer.encode(prefill or "", add_special_tokens=False))
        return tinker.ModelInput.from_ints(tokens)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get tokens and weights for action corresponding to final message
        """
        return build_supervised_example(
            self._bos_tokens,
            lambda _idx, message: self._render_message(message),
            messages,
            train_on_what,
        )

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

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        return parse_response_for_stop_token(response, self.tokenizer, self._end_message_token)


class DeepSeekV3DisableThinkingRenderer(DeepSeekV3Renderer):
    """
    Renderer that disables thinking for DsV3 models
    """

    def _render_message(self, message: Message) -> tuple[list[int], list[int], list[int]]:
        if (
            message["role"] == "assistant"
            and not message["content"].startswith("<think>")
            and not message["content"].startswith("</think>")
        ):
            message["content"] = "</think>" + message["content"]
        return super()._render_message(message)

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        prefill = "</think>" + (prefill or "")
        return super().build_generation_prompt(messages, role, prefill)


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

    def _render_message(
        self, message: Message, is_last: bool = False
    ) -> tuple[list[int], list[int], list[int]]:
        assert message.get("tool_calls") is None, "TODO: support tools in gpt-oss renderer"
        # Observation (prompt) part
        ob_str = f"<|start|>{message['role']}"
        # Action part
        ac_str = ""
        if message["role"] == "assistant":
            # TODO: support commentary channel / tools

            # Assistant channels. See https://cookbook.openai.com/articles/openai-harmony
            thinking = message.get("thinking")
            content = message.get("content", "")

            # Analysis channel (CoT)
            if thinking:
                if is_last:
                    # Analysis channel only included in the last message. See https://cookbook.openai.com/articles/gpt-oss/handle-raw-cot
                    ac_str += f"<|channel|>analysis<|message|>{thinking}<|end|><|start|>assistant"

            # Final channel (Response Content)
            ac_str += f"<|channel|>final<|message|>{content}"
        else:
            assert message.get("thinking") is None, (
                "Thinking is only allowed for assistant messages"
            )
            ac_str += f"<|message|>{message['content']}"

        if not is_last:
            ac_str += "<|end|>"
        else:
            # <|return|> ends the last-message in harmony (but should be replaced by <|end|> when continuing the convo)
            ac_str += "<|return|>"

        # Action part that's only included in the last message in SFT
        ac_tail_str = ""  # No action tail needed for gpt-oss format
        return (
            self.tokenizer.encode(ob_str, add_special_tokens=False),
            self.tokenizer.encode(ac_str, add_special_tokens=False),
            self.tokenizer.encode(ac_tail_str, add_special_tokens=False),
        )

    def _build_system_prompt(self) -> str:
        current_date = (
            self.current_date
            if self.current_date is not None
            else datetime.now().strftime("%Y-%m-%d")
        )
        return self.system_prompt.format(
            current_date=current_date, reasoning_effort=self.reasoning_effort
        )

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        tokens: list[int] = []
        tokens.extend(self._bos_tokens)
        if self.use_system_prompt:
            tokens.extend(
                self.tokenizer.encode(self._build_system_prompt(), add_special_tokens=False)
            )
        for message in messages:
            ob_part, action_part, action_tail = self._render_message(message)
            tokens.extend(ob_part)
            tokens.extend(action_part)
        new_partial_message = Message(role=role, content="")
        ob_part, _action_part, _action_tail = self._render_message(new_partial_message)
        tokens.extend(ob_part)
        tokens.extend(self.tokenizer.encode(prefill or "", add_special_tokens=False))
        return tinker.ModelInput.from_ints(tokens)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get tokens and weights for action corresponding to final message
        """
        start_tokens = self._bos_tokens
        if self.use_system_prompt:
            start_tokens.extend(
                self.tokenizer.encode(self._build_system_prompt(), add_special_tokens=False)
            )
        return build_supervised_example(
            start_tokens,
            lambda _idx, message: self._render_message(message, is_last=_idx == len(messages) - 1),
            messages,
            train_on_what,
        )

    @property
    def _bos_tokens(self) -> list[int]:
        return []

    @property
    def _return_token(self) -> int:
        res = self.tokenizer.encode("<|return|>", add_special_tokens=False)
        assert len(res) == 1, f"Expected single token for <|return|>, got {len(res)}"
        return res[0]

    def get_stop_sequences(self) -> list[int]:
        return [self._return_token]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        return parse_response_for_stop_token(response, self.tokenizer, self._return_token)


def get_renderer(name: str, tokenizer: Tokenizer) -> Renderer:
    if name == "role_colon":
        return RoleColonRenderer(tokenizer)
    elif name == "llama3":
        return Llama3Renderer(tokenizer)
    elif name == "qwen3":
        return Qwen3Renderer(tokenizer)
    elif name == "qwen3_disable_thinking":
        return Qwen3DisableThinkingRenderer(tokenizer)
    elif name == "qwen3_instruct":
        return Qwen3InstructRenderer(tokenizer)
    elif name == "deepseekv3":
        return DeepSeekV3Renderer(tokenizer)
    elif name == "deepseekv3_disable_thinking":
        return DeepSeekV3DisableThinkingRenderer(tokenizer)
    elif name == "gpt_oss_no_sysprompt":
        return GptOssRenderer(tokenizer, use_system_prompt=False)
    elif name == "gpt_oss_low_reasoning":
        return GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="low")
    elif name == "gpt_oss_medium_reasoning":
        return GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")
    elif name == "gpt_oss_high_reasoning":
        return GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="high")
    else:
        raise ValueError(f"Unknown renderer: {name}")
