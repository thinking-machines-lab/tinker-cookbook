"""
Tests for DeepSeek-V4.1 renderers.

Tests verify that the DeepSeek-V4.1 renderers produce correct output, in both thinking and
chat mode:
1. <｜Assistant｜> follows only a user turn (the template writes it at the end of the turn),
   so a conversation that starts with a system or assistant message matches HF
2. HF template compatibility for build_generation_prompt and build_supervised_example
   (exact): thinking before the last user message is dropped unless tools are declared,
   whichever assistant messages are trained
3. Text that spells a control marker encodes as that marker's token, as HF does
4. Tool results are sorted into call order across the whole merged user turn, and ids
   that are missing, duplicated, or unknown order the results as HF does
5. Tool declarations fold into the system message (an empty one when there is none) and
   match HF exactly; a declaration after a user message raises
6. The last assistant turn trains the tool call and the answer, not the tool results
7. parse_response reads DSML tool calls by their string flag and reports malformed DSML
   as unparsed tool calls; streaming yields deltas and a complete final message
8. deepseekv4_1_preserve_thinking keeps earlier thinking, as HF's drop_thinking=False does
9. An image in an assistant message raises RendererError
"""

import json
from collections.abc import Callable

import pytest

from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.renderers import (
    ImagePart,
    Message,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    get_renderer,
)
from tinker_cookbook.renderers.base import (
    StreamingMessageHeader,
    StreamingTextDelta,
    StreamingThinkingDelta,
)
from tinker_cookbook.renderers.testing_utils import extract_token_ids
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
EOS = "<｜end▁of▁sentence｜>"


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL)


_HF_MODES = pytest.mark.parametrize(
    "renderer_name,hf_kwargs",
    [("deepseekv4_1", {}), ("deepseekv4_1_disable_thinking", {"thinking_mode": "chat"})],
    ids=["thinking", "chat"],
)


def _hf_generation_tokens(tokenizer, hf_messages, tools=None, **hf_kwargs) -> list[int]:
    """Run HF apply_chat_template with generation prompt and return token list."""
    return extract_token_ids(
        tokenizer.apply_chat_template(
            hf_messages, tools=tools, add_generation_prompt=True, tokenize=True, **hf_kwargs
        )
    )


def _hf_supervised_tokens(tokenizer, hf_messages, tools=None, **hf_kwargs) -> list[int]:
    """Run HF apply_chat_template without generation prompt and return token list."""
    return extract_token_ids(
        tokenizer.apply_chat_template(hf_messages, tools=tools, tokenize=True, **hf_kwargs)
    )


# =============================================================================
# Test Conversations
# =============================================================================

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Get the weather",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["location"],
    },
)
HF_WEATHER_TOOLS = [{"type": "function", "function": WEATHER_TOOL}]


def _weather_call(arguments: dict, call_id: str | None = "c1") -> ToolCall:
    return ToolCall(
        function=ToolCall.FunctionBody(name="get_weather", arguments=json.dumps(arguments)),
        id=call_id,
    )


def get_thinking_conversation_for_supervised() -> list[Message]:
    """Conversation with thinking content, ending with assistant."""
    return [
        Message(role="user", content="Solve 2+2."),
        Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking="2 plus 2 equals 4."),
                TextPart(type="text", text="The answer is 4."),
            ],
        ),
    ]


def get_multiturn_thinking_conversation() -> list[Message]:
    """Multi-turn with thinking in both assistant messages."""
    return [
        Message(role="user", content="First question."),
        Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking="First turn reasoning."),
                TextPart(type="text", text="First answer."),
            ],
        ),
        Message(role="user", content="Second question."),
        Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking="Second turn reasoning."),
                TextPart(type="text", text="Second answer."),
            ],
        ),
    ]


def get_tool_call_conversation() -> list[Message]:
    """A tool call (with a non-string argument), its result, and the answer, each assistant
    message with thinking."""
    return [
        Message(role="user", content="Weather in Paris?"),
        Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking="Look it up."),
                TextPart(type="text", text="Checking."),
            ],
            tool_calls=[_weather_call({"location": "Paris", "days": 2})],
        ),
        Message(role="tool", content="sunny", tool_call_id="c1"),
        Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking="The tool says sunny."),
                TextPart(type="text", text="It is sunny."),
            ],
        ),
    ]


def get_parallel_tool_call_conversation(
    call_ids: tuple[str | None, ...] = ("c1", "c2"),
    result_ids: tuple[str | None, ...] = ("c1", "c2"),
) -> list[Message]:
    """Two tool calls and consecutive tool results, ending with the results.

    The ids are parameterizable for the ordering tests. A result id of None omits the
    tool_call_id field; fewer result ids than calls gives a partial set of results.
    """
    calls = [
        _weather_call({"location": "Paris"}, call_ids[0]),
        _weather_call({"location": "Rome", "days": 3}, call_ids[1]),
    ]
    results: list[Message] = []
    for result_id, content in zip(result_ids, ("sun", "rain")):
        result = Message(role="tool", content=content)
        if result_id is not None:
            result["tool_call_id"] = result_id
        results.append(result)
    return [
        Message(role="user", content="Weather in Paris and Rome?"),
        Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking="Check both cities."),
                TextPart(type="text", text=""),
            ],
            tool_calls=calls,
        ),
        *results,
    ]


# =============================================================================
# Prompt Structure Tests
# =============================================================================


@_HF_MODES
@pytest.mark.parametrize(
    "roles",
    [
        ("assistant",),
        ("system", "assistant"),
        ("system", "assistant", "user"),
        ("user", "assistant", "assistant"),
    ],
    ids=["model-only", "after-first-system", "after-first-system-then-user", "after-assistant"],
)
def test_assistant_marker_only_follows_a_user_turn(
    tokenizer, renderer_name: str, hf_kwargs: dict, roles: tuple[str, ...]
):
    """The template writes ``<｜Assistant｜>`` at the end of a user turn, so an assistant
    message after only the first system message, or after another assistant, has none."""
    renderer = get_renderer(renderer_name, tokenizer)
    messages = [Message(role=role, content=f"m{idx}") for idx, role in enumerate(roles)]

    model_input, _ = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    expected = _hf_supervised_tokens(
        tokenizer, [renderer.to_openai_message(m) for m in messages], **hf_kwargs
    )

    assert model_input.to_ints() == expected


@pytest.mark.parametrize("empty_role", ["system", "user"])
def test_empty_message_renders_no_empty_chunk(tokenizer, empty_role: str):
    """Tinker rejects a chunk with no tokens."""
    renderer = get_renderer("deepseekv4_1", tokenizer)
    messages = [
        Message(role=empty_role, content=""),
        Message(role="user", content="q"),
        Message(role="assistant", content="a"),
    ]

    model_input, _ = renderer.build_supervised_example(messages)

    assert all(chunk.length > 0 for chunk in model_input.chunks)


# =============================================================================
# HF Template Compatibility Tests — Generation
# =============================================================================


@_HF_MODES
def test_thinking_in_history_generation_matches_hf(tokenizer, renderer_name: str, hf_kwargs: dict):
    """Without declared tools, thinking before the last user message is dropped."""
    renderer = get_renderer(renderer_name, tokenizer)
    messages = get_multiturn_thinking_conversation()[:-1]

    expected = _hf_generation_tokens(
        tokenizer, [renderer.to_openai_message(m) for m in messages], **hf_kwargs
    )

    assert renderer.build_generation_prompt(messages).to_ints() == expected


@_HF_MODES
@pytest.mark.parametrize(
    "messages",
    [
        [Message(role="user", content="Say <｜Assistant｜></think> and <｜User｜> verbatim.")],
        [
            Message(role="user", content="hi"),
            Message(role="assistant", content="a</think>b"),
            Message(role="user", content="again"),
        ],
    ],
    ids=["in-user-text", "in-history-assistant-text"],
)
def test_marker_spelled_in_text_matches_hf(
    tokenizer, renderer_name: str, hf_kwargs: dict, messages: list[Message]
):
    renderer = get_renderer(renderer_name, tokenizer)

    expected = _hf_generation_tokens(
        tokenizer, [renderer.to_openai_message(m) for m in messages], **hf_kwargs
    )

    assert renderer.build_generation_prompt(messages).to_ints() == expected


def test_preserve_thinking_generation_matches_hf(tokenizer):
    renderer = get_renderer("deepseekv4_1_preserve_thinking", tokenizer)
    messages = get_multiturn_thinking_conversation()[:-1]

    expected = _hf_generation_tokens(
        tokenizer, [renderer.to_openai_message(m) for m in messages], drop_thinking=False
    )

    assert renderer.build_generation_prompt(messages).to_ints() == expected


# =============================================================================
# Tool Response Ordering Tests
# =============================================================================
#
# The template sorts every tool result of a user turn into call order, around any user text
# in the turn. The result tags carry no ids, so the order is the only link to a call. The
# sort is stable, and a result whose id matches no call sorts as call 0.


@_HF_MODES
@pytest.mark.parametrize("interleaved", [False, True], ids=["consecutive", "with-user-text"])
def test_out_of_order_tool_results_match_hf(
    tokenizer, renderer_name: str, hf_kwargs: dict, interleaved: bool
):
    renderer = get_renderer(renderer_name, tokenizer)
    messages = get_parallel_tool_call_conversation(result_ids=("c2", "c1"))
    if interleaved:
        messages.insert(3, Message(role="user", content="Also pack an umbrella?"))

    expected = _hf_generation_tokens(
        tokenizer, [renderer.to_openai_message(m) for m in messages], **hf_kwargs
    )

    assert renderer.build_generation_prompt(messages).to_ints() == expected


@_HF_MODES
def test_out_of_order_tool_results_supervised_matches_hf(
    tokenizer, renderer_name: str, hf_kwargs: dict
):
    renderer = get_renderer(renderer_name, tokenizer)
    messages = get_parallel_tool_call_conversation(result_ids=("c2", "c1"))
    messages.append(Message(role="assistant", content="Paris is drier."))

    model_input, _ = renderer.build_supervised_example(messages)
    expected = _hf_supervised_tokens(
        tokenizer, [renderer.to_openai_message(m) for m in messages], **hf_kwargs
    )

    assert model_input.to_ints() == expected


@_HF_MODES
@pytest.mark.parametrize(
    "call_ids,result_ids",
    [
        (("c1", "c2"), ("c2",)),
        (("c1", "c2"), ("c2", None)),
        (("c1", "c2"), ("c2", "c2")),
        (("c1", "c2"), ("c2", "c_unknown")),
        ((None, None), ("c2", "c1")),
        (("c1", "c1"), ("c1", "c1")),
    ],
    ids=[
        "partial-results",
        "missing-result-id",
        "dup-result-id",
        "unknown-result-id",
        "no-call-ids",
        "dup-call-ids",
    ],
)
def test_ambiguous_tool_result_ids_match_hf(
    tokenizer,
    renderer_name: str,
    hf_kwargs: dict,
    call_ids: tuple[str | None, ...],
    result_ids: tuple[str | None, ...],
):
    renderer = get_renderer(renderer_name, tokenizer)
    messages = get_parallel_tool_call_conversation(call_ids=call_ids, result_ids=result_ids)

    expected = _hf_generation_tokens(
        tokenizer, [renderer.to_openai_message(m) for m in messages], **hf_kwargs
    )

    assert renderer.build_generation_prompt(messages).to_ints() == expected


# =============================================================================
# HF Template Compatibility Tests — Supervised
# =============================================================================


@_HF_MODES
@pytest.mark.parametrize(
    "train_on_what",
    [TrainOnWhat.LAST_ASSISTANT_MESSAGE, TrainOnWhat.ALL_ASSISTANT_MESSAGES],
    ids=["last", "all"],
)
@pytest.mark.parametrize(
    "conversation_fn",
    [get_thinking_conversation_for_supervised, get_multiturn_thinking_conversation],
    ids=["thinking-then-text", "thinking-in-history-and-current"],
)
def test_thinking_supervised_matches_hf(
    tokenizer,
    renderer_name: str,
    hf_kwargs: dict,
    train_on_what: TrainOnWhat,
    conversation_fn: Callable[[], list[Message]],
):
    """The last turn keeps its thinking and earlier turns lose theirs, whichever turns are
    trained."""
    renderer = get_renderer(renderer_name, tokenizer)
    messages = conversation_fn()

    model_input, _ = renderer.build_supervised_example(messages, train_on_what=train_on_what)
    expected = _hf_supervised_tokens(
        tokenizer, [renderer.to_openai_message(m) for m in messages], **hf_kwargs
    )

    assert model_input.to_ints() == expected


@_HF_MODES
@pytest.mark.parametrize("declared", [False, True], ids=["undeclared", "declared"])
def test_tool_call_supervised_matches_hf(
    tokenizer, renderer_name: str, hf_kwargs: dict, declared: bool
):
    """Declaring tools keeps the tool call's thinking, which the template otherwise drops
    because the tool result counts as a user turn."""
    renderer = get_renderer(renderer_name, tokenizer)
    conversation = get_tool_call_conversation()
    prefix = renderer.create_conversation_prefix_with_tools([WEATHER_TOOL]) if declared else []

    model_input, _ = renderer.build_supervised_example(prefix + conversation)
    expected = _hf_supervised_tokens(
        tokenizer,
        [renderer.to_openai_message(m) for m in conversation],
        tools=HF_WEATHER_TOOLS if declared else None,
        **hf_kwargs,
    )

    assert model_input.to_ints() == expected


@_HF_MODES
def test_empty_assistant_message_supervised_matches_hf(
    tokenizer, renderer_name: str, hf_kwargs: dict
):
    """An empty answer still trains its terminator."""
    renderer = get_renderer(renderer_name, tokenizer)
    messages = [Message(role="user", content="q"), Message(role="assistant", content="")]

    model_input, weights = renderer.build_supervised_example(messages)
    expected = _hf_supervised_tokens(
        tokenizer, [renderer.to_openai_message(m) for m in messages], **hf_kwargs
    )

    assert model_input.to_ints() == expected
    assert model_input.to_ints()[-1] == expected[-1] and weights[-1] > 0


def test_preserve_thinking_supervised_matches_hf(tokenizer):
    renderer = get_renderer("deepseekv4_1_preserve_thinking", tokenizer)
    messages = get_multiturn_thinking_conversation()

    model_input, _ = renderer.build_supervised_example(messages)
    expected = _hf_supervised_tokens(
        tokenizer, [renderer.to_openai_message(m) for m in messages], drop_thinking=False
    )

    assert model_input.to_ints() == expected


# =============================================================================
# Tool Declaration Tests
# =============================================================================


@_HF_MODES
@pytest.mark.parametrize("system", ["", "Be brief."], ids=["no-system", "system"])
def test_tool_declaration_matches_hf(tokenizer, renderer_name: str, hf_kwargs: dict, system: str):
    """Covers the tools text, its fold into the system message (an empty one when there is
    none, so chat mode opens with ``<｜System｜>`` too), and history thinking kept because
    tools are declared."""
    renderer = get_renderer(renderer_name, tokenizer)
    conversation = [*get_tool_call_conversation(), Message(role="user", content="And tomorrow?")]
    messages = renderer.create_conversation_prefix_with_tools([WEATHER_TOOL], system)
    hf_messages = [{"role": "system", "content": system}] if system else []
    hf_messages += [renderer.to_openai_message(m) for m in conversation]

    expected = _hf_generation_tokens(tokenizer, hf_messages, tools=HF_WEATHER_TOOLS, **hf_kwargs)

    assert renderer.build_generation_prompt(messages + conversation).to_ints() == expected


def test_tool_declaration_after_a_user_message_raises(tokenizer):
    renderer = get_renderer("deepseekv4_1", tokenizer)
    messages = [Message(role="user", content="q")]
    messages += renderer.create_conversation_prefix_with_tools([WEATHER_TOOL])

    with pytest.raises(RendererError, match="tool declarations must come first"):
        renderer.build_generation_prompt(messages)


@pytest.mark.parametrize("system", ["", "Be brief."], ids=["no-system", "system"])
def test_tool_declaration_keeps_trainable_flags(tokenizer, system: str):
    """Folding the declaration into the system message keeps the fields CUSTOMIZED reads."""
    renderer = get_renderer("deepseekv4_1", tokenizer)
    messages = renderer.create_conversation_prefix_with_tools([WEATHER_TOOL], system)
    messages += [Message(role="user", content="q"), Message(role="assistant", content="a")]
    for message in messages:
        message["trainable"] = message["role"] == "assistant"

    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.CUSTOMIZED
    )
    tokens = model_input.to_ints()

    assert (
        tokenizer.decode([t for t, w in zip(tokens, weights.tolist()) if w > 0])
        == f"</think>a{EOS}"
    )


# =============================================================================
# Loss Weighting Tests
# =============================================================================


def test_last_assistant_turn_trains_the_tool_call_and_the_answer(tokenizer):
    """Tool results share a user turn with any user text that follows; the merge must not
    move the start of the last assistant turn."""
    renderer = get_renderer("deepseekv4_1", tokenizer)

    model_input, weights = renderer.build_supervised_example(
        get_tool_call_conversation(), train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN
    )
    tokens = model_input.to_ints()
    trained = tokenizer.decode([t for t, w in zip(tokens, weights.tolist()) if w > 0])

    assert trained.startswith("Checking.\n\n<｜DSML｜ calls>")
    assert trained.endswith(f"{EOS}The tool says sunny.</think>It is sunny.{EOS}")
    assert "<tool_result>" not in trained


# =============================================================================
# Parse Response Tests
# =============================================================================

_INVOKE = (
    '<｜DSML｜ invoke name="get_weather">\n'
    '<｜DSML｜ parameter name="days" string="false">2</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>"
)


def _dsml_response(params: str, name: str = "get_weather") -> str:
    return (
        f'Checking.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="{name}">\n'
        f"{params}\n</｜DSML｜ invoke>\n</｜DSML｜ calls>{EOS}"
    )


def test_parse_response_reads_string_flag(tokenizer):
    """``string="true"`` values stay raw text; all others are JSON."""
    renderer = get_renderer("deepseekv4_1", tokenizer)
    response = "plan</think>" + _dsml_response(
        '<｜DSML｜ parameter name="location" string="true">[1, 2]</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="days" string="false">2</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="units" string="false">{"temp": "C", "wind": null}'
        "</｜DSML｜ parameter>"
    )

    message, termination = renderer.parse_response(
        tokenizer.encode(response, add_special_tokens=False)
    )

    assert termination.is_clean
    assert message["content"] == [
        {"type": "thinking", "thinking": "plan"},
        TextPart(type="text", text="Checking."),
    ]
    (call,) = message.get("tool_calls", [])
    assert json.loads(call.function.arguments) == {
        "location": "[1, 2]",
        "days": 2,
        "units": {"temp": "C", "wind": None},
    }


@pytest.mark.parametrize(
    "name,params",
    [
        (
            "get_weather",
            '<｜DSML｜ parameter name="days" string="false">2</｜DSML｜ parameter>\n'
            '<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>',
        ),
        ("get_weather", '<｜DSML｜ parameter name="days" string="false">two</｜DSML｜ parameter>'),
        ("get_weather", '<｜DSML｜ parameter name="days" strin="false">2</｜DSML｜ parameter>'),
        ("", '<｜DSML｜ parameter name="days" string="false">2</｜DSML｜ parameter>'),
        (" ", '<｜DSML｜ parameter name="days" string="false">2</｜DSML｜ parameter>'),
        ("get_weather", '<｜DSML｜ parameter name="" string="true">x</｜DSML｜ parameter>'),
    ],
    ids=[
        "duplicate-key",
        "invalid-json",
        "unmatched-parameter",
        "empty-function-name",
        "blank-function-name",
        "empty-parameter-name",
    ],
)
def test_parse_response_keeps_invalid_calls_as_unparsed(tokenizer, name: str, params: str):
    renderer = get_renderer("deepseekv4_1_disable_thinking", tokenizer)

    message, termination = renderer.parse_response(
        tokenizer.encode(_dsml_response(params, name), add_special_tokens=False)
    )

    assert termination.is_clean
    assert "tool_calls" not in message
    assert len(message.get("unparsed_tool_calls", [])) == 1
    assert message["content"] == "Checking."


@pytest.mark.parametrize(
    "response",
    [
        'Checking.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather"></｜DSML｜ invoke>\n'
        f"</｜DSML｜ calls>{EOS}",
        f"Checking.\n\n<｜DSML｜ calls>\n{_INVOKE}\n</｜DSML｜ calls>"
        f"\n\n<｜DSML｜ calls>\n{_INVOKE}\n</｜DSML｜ calls>{EOS}",
        f"Checking.\n\n<｜DSML｜ calls>\n{_INVOKE}\n</｜DSML｜ calls>\nDone.{EOS}",
        f"Checking.\n\n<｜DSML｜ calls>\n<｜DSML｜ calls>\n{_INVOKE}\n</｜DSML｜ calls>{EOS}",
        f"Checking.\n\n<｜DSML｜ calls>\n{_INVOKE}\n<｜DSML｜ calls>\n</｜DSML｜ calls>{EOS}",
    ],
    ids=[
        "invoke-without-newline",
        "second-calls-block",
        "text-after-calls-block",
        "duplicate-calls-open",
        "calls-open-after-invoke",
    ],
)
def test_parse_response_reports_output_outside_the_format(tokenizer, response: str):
    """A response ends with one calls block of invokes; anything else is reported."""
    renderer = get_renderer("deepseekv4_1_disable_thinking", tokenizer)

    message, termination = renderer.parse_response(
        tokenizer.encode(response, add_special_tokens=False)
    )

    assert termination.is_clean
    assert message["content"] == "Checking."
    assert len(message.get("unparsed_tool_calls", [])) == 1


@pytest.mark.parametrize(
    "response",
    [f"Checking.\n\n{_INVOKE}{EOS}", f"Checking.</｜DSML｜ calls>{EOS}"],
    ids=["invoke-without-calls-block", "stray-calls-close"],
)
def test_parse_response_reports_dsml_outside_a_calls_block(tokenizer, response: str):
    renderer = get_renderer("deepseekv4_1_disable_thinking", tokenizer)

    message, termination = renderer.parse_response(
        tokenizer.encode(response, add_special_tokens=False)
    )

    assert termination.is_clean
    assert "tool_calls" not in message
    assert len(message.get("unparsed_tool_calls", [])) == 1


# =============================================================================
# Streaming Tests
# =============================================================================


def test_parse_response_streaming(tokenizer):
    """Streaming yields thinking deltas, text deltas, and a complete final Message."""
    renderer = get_renderer("deepseekv4_1", tokenizer)
    tokens = tokenizer.encode(
        f"Let me think.</think>The answer is 42.{EOS}", add_special_tokens=False
    )

    deltas = list(renderer.parse_response_streaming(tokens))

    assert isinstance(deltas[0], StreamingMessageHeader)
    thinking = "".join(d.thinking for d in deltas if isinstance(d, StreamingThinkingDelta))
    text = "".join(d.text for d in deltas if isinstance(d, StreamingTextDelta))
    assert thinking == "Let me think."
    assert text == "The answer is 42."
    final_message = deltas[-1]
    assert isinstance(final_message, dict)
    assert final_message["content"] == [
        {"type": "thinking", "thinking": "Let me think."},
        {"type": "text", "text": "The answer is 42."},
    ]


@pytest.mark.parametrize("stop", [EOS, ""], ids=["stopped", "truncated"])
def test_parse_response_streaming_parses_tool_calls(tokenizer, stop: str):
    """The final Message carries the tool calls even when the stop token is missing."""
    renderer = get_renderer("deepseekv4_1_disable_thinking", tokenizer)
    response = _dsml_response(
        '<｜DSML｜ parameter name="days" string="false">2</｜DSML｜ parameter>'
    ).removesuffix(EOS)

    deltas = list(
        renderer.parse_response_streaming(
            tokenizer.encode(response + stop, add_special_tokens=False)
        )
    )

    final_message = deltas[-1]
    assert isinstance(final_message, dict)
    (call,) = final_message.get("tool_calls", [])
    assert call.function.name == "get_weather"


# =============================================================================
# Unsupported Input Tests
# =============================================================================


def test_assistant_image_content_raises(tokenizer):
    renderer = get_renderer("deepseekv4_1", tokenizer)
    image = ImagePart(type="image", image="https://example.com/cat.png")
    messages = [Message(role="user", content="q"), Message(role="assistant", content=[image])]

    with pytest.raises(RendererError):
        renderer.build_supervised_example(messages, train_on_what=TrainOnWhat.ALL_MESSAGES)
