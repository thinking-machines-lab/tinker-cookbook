"""
Tests for GLM-5.3 renderers.

Tests verify that the GLM-5.3 renderers produce correct output:
1. Prompts start with [gMASK]<sop> plus the always-present Reasoning Effort
   system segment (Max by default, Low/High for the variants)
2. Generation prompts always end with <|assistant|><think> (GLM-5.3 has no
   template-level way to disable thinking)
3. HF template compatibility for build_generation_prompt (exact) and
   build_supervised_example (exact plus the trailing turn terminator, which the
   HF template does not emit but the model must learn to generate)
4. Thinking is preserved for ALL assistant messages by default (HF
   clear_thinking=False default); strip_thinking_from_history=True matches HF
   clear_thinking=True
5. Tool declarations and <arg_key>/<arg_value> tool calls match HF exactly
6. Consecutive tool responses are re-sorted to match the assistant's tool_calls
   order when ids allow an unambiguous match, falling back to message order
   otherwise — matching HF in both cases
7. parse_response handles GLM's multiple stop tokens (<|user|>, <|observation|>,
   <|endoftext|>) and GLM-format tool calls
"""

import json
from typing import cast

import pytest

from tinker_cookbook.renderers import (
    Message,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    get_renderer,
    get_text_content,
)
from tinker_cookbook.renderers.base import (
    StreamingMessageHeader,
    StreamingTextDelta,
    StreamingThinkingDelta,
    ensure_list,
)
from tinker_cookbook.renderers.glm5_3 import (
    Glm5_3HighReasoningRenderer,
    Glm5_3LowReasoningRenderer,
    Glm5_3Renderer,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

GLM5_3_MODEL = "zai-org/GLM-5.3"


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def glm_tokenizer():
    return get_tokenizer(GLM5_3_MODEL)


@pytest.fixture(scope="module")
def glm_renderer(glm_tokenizer):
    return get_renderer("glm5_3_max_reasoning", glm_tokenizer)


@pytest.fixture(scope="module")
def glm_renderer_low_reasoning(glm_tokenizer):
    return get_renderer("glm5_3_low_reasoning", glm_tokenizer)


@pytest.fixture(scope="module")
def glm_renderer_high_reasoning(glm_tokenizer):
    return get_renderer("glm5_3_high_reasoning", glm_tokenizer)


def _hf_generation_tokens(tokenizer, hf_messages, tools=None, **hf_kwargs):
    """Run HF apply_chat_template with generation prompt and return token list."""
    result = tokenizer.apply_chat_template(
        hf_messages, tools=tools, add_generation_prompt=True, tokenize=True, **hf_kwargs
    )
    if hasattr(result, "input_ids"):
        return list(result["input_ids"])
    return list(result)


def _hf_supervised_tokens(tokenizer, hf_messages, terminator="<|user|>", tools=None, **hf_kwargs):
    """Run HF apply_chat_template without generation prompt and append the turn terminator.

    GLM-5.3 has no per-message end token; the model terminates an assistant turn
    by emitting the next role token (<|user|> after a reply, <|observation|>
    after tool calls). The cookbook renderer appends that terminator to the
    supervised target so the model learns to emit it, while the HF template does
    not include it — so it is appended here for comparison.
    """
    result = tokenizer.apply_chat_template(
        hf_messages, tools=tools, add_generation_prompt=False, tokenize=False, **hf_kwargs
    )
    assert isinstance(result, str)
    return tokenizer.encode(result + terminator, add_special_tokens=False)


# =============================================================================
# Test Conversations
# =============================================================================


def get_basic_conversation_for_generation() -> list[Message]:
    """3-turn conversation ending with user message (for generation)."""
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello, how are you?"),
        Message(role="assistant", content="I'm fine, thank you!"),
        Message(role="user", content="What is the capital of France?"),
    ]


def get_basic_conversation_for_supervised() -> list[Message]:
    """2-turn conversation ending with assistant (for supervised)."""
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello, how are you?"),
        Message(role="assistant", content="I'm fine, thank you!"),
    ]


def get_thinking_conversation_for_supervised() -> list[Message]:
    """Conversation with thinking content, ending with assistant."""
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Solve 2+2."),
        Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "2 plus 2 equals 4."},
                {"type": "text", "text": "The answer is 4."},
            ],
        ),
    ]


def get_multiturn_thinking_conversation() -> list[Message]:
    """Multi-turn with thinking in both assistant messages."""
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="First question."),
        Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "First turn reasoning."},
                {"type": "text", "text": "First answer."},
            ],
        ),
        Message(role="user", content="Second question."),
        Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "Second turn reasoning."},
                {"type": "text", "text": "Second answer."},
            ],
        ),
    ]


def get_embedded_think_string_conversation() -> list[Message]:
    """Multi-turn conversation with <think> tags embedded in string content.

    The GLM-5.3 HF template extracts reasoning from embedded <think>...</think>
    tags in string content and strips the remaining text.
    """
    return [
        Message(role="user", content="First question."),
        Message(role="assistant", content="<think>first reasoning</think>\n\nFirst answer."),
        Message(role="user", content="Second question."),
        Message(role="assistant", content="<think>second reasoning</think>\n\nSecond answer."),
    ]


def get_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    )


def get_tool_call_conversation_for_supervised() -> tuple[list[Message], list[ToolSpec]]:
    tools = [get_tool_spec()]
    tool_call = ToolCall(
        id="call_abc123",
        function=ToolCall.FunctionBody(
            name="get_weather",
            arguments='{"location": "New York, NY"}',
        ),
    )
    messages: list[Message] = [
        Message(role="user", content="What's the weather in NYC?"),
        Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "I need to check the weather in NYC."},
                {"type": "text", "text": ""},
            ],
            tool_calls=[tool_call],
        ),
        Message(
            role="tool",
            name="get_weather",
            tool_call_id="call_abc123",
            content='{"temperature": 72, "condition": "sunny"}',
        ),
        Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "The weather is 72F and sunny."},
                {"type": "text", "text": "The weather in NYC is 72°F and sunny."},
            ],
        ),
    ]
    return messages, tools


def get_tool_call_conversation_for_generation() -> tuple[list[Message], list[ToolSpec]]:
    messages, tools = get_tool_call_conversation_for_supervised()
    return messages[:-1], tools


def get_parallel_tool_call_conversation(
    call_ids: tuple[str | None, str | None] = ("call_1", "call_2"),
    result_ids: tuple[str | None, str | None] = ("call_1", "call_2"),
) -> tuple[list[Message], list[ToolSpec]]:
    """Two tool calls (one with typed argument values) and consecutive tool responses.

    The tool call / tool response ids are parameterizable so the tool-response
    ordering tests can construct out-of-order, duplicate, unknown, and missing
    id scenarios. A result id of None omits the tool_call_id field entirely.
    """
    tools = [get_tool_spec()]
    tool_calls = [
        ToolCall(
            id=call_ids[0],
            function=ToolCall.FunctionBody(name="get_weather", arguments='{"location": "NYC"}'),
        ),
        ToolCall(
            id=call_ids[1],
            function=ToolCall.FunctionBody(
                name="get_weather",
                arguments='{"location": "SF", "days": 3, "options": {"units": "F"}}',
            ),
        ),
    ]
    tool_results = []
    for result_id, content in zip(result_ids, ('{"temp": 72}', '{"temp": 65}')):
        result = Message(role="tool", name="get_weather", content=content)
        if result_id is not None:
            result["tool_call_id"] = result_id
        tool_results.append(result)
    messages: list[Message] = [
        Message(role="user", content="Compare the weather in NYC and SF."),
        Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "I should check both cities."},
                {"type": "text", "text": ""},
            ],
            tool_calls=tool_calls,
        ),
        *tool_results,
    ]
    return messages, tools


# =============================================================================
# Prompt Structure Tests
# =============================================================================


def test_generation_prompt_prefix_and_think(glm_tokenizer, glm_renderer):
    """Default prompts start with [gMASK]<sop> + Reasoning Effort: Max and end with <think>."""
    messages = get_basic_conversation_for_generation()
    decoded = glm_tokenizer.decode(glm_renderer.build_generation_prompt(messages).to_ints())
    assert decoded.startswith("[gMASK]<sop><|system|>Reasoning Effort: Max")
    assert decoded.endswith("<|assistant|><think>")
    assert not decoded.endswith("<think></think>")


@pytest.mark.parametrize(
    "renderer_name,effort",
    [
        ("glm5_3_max_reasoning", "Max"),
        ("glm5_3_low_reasoning", "Low"),
        ("glm5_3_high_reasoning", "High"),
    ],
)
def test_reasoning_effort_segment_always_emitted(renderer_name, effort, glm_tokenizer):
    """Every variant emits the Reasoning Effort segment and prefills an open <think>.

    GLM-5.3 has no template-level way to disable thinking: the effort segment is
    always present (unlike GLM-5.2, which dropped it when thinking was disabled)
    and the generation prompt always ends with an open <think> tag.
    """
    renderer = get_renderer(renderer_name, glm_tokenizer)
    messages = get_basic_conversation_for_generation()
    decoded = glm_tokenizer.decode(renderer.build_generation_prompt(messages).to_ints())
    assert decoded.startswith(f"[gMASK]<sop><|system|>Reasoning Effort: {effort}")
    assert decoded.endswith("<|assistant|><think>")
    assert not decoded.endswith("<think></think>")


def test_custom_prefill_appends_after_think(glm_tokenizer, glm_renderer):
    """A custom prefill continues after the <think> prefill (thinking is always on)."""
    messages = get_basic_conversation_for_generation()
    decoded = glm_tokenizer.decode(
        glm_renderer.build_generation_prompt(messages, prefill="Sure, ").to_ints()
    )
    assert decoded.endswith("<|assistant|><think>Sure, ")


def test_historical_thinking_preserved_by_default(glm_tokenizer, glm_renderer):
    """Thinking in assistant messages before the last user message is preserved.

    The HF template's clear_thinking defaults to False in GLM-5.3 (flipped from
    GLM-5.2), so past-turn reasoning stays in the rendered context by default.
    """
    messages = get_multiturn_thinking_conversation()
    decoded = glm_tokenizer.decode(glm_renderer.build_supervised_example(messages)[0].to_ints())
    assert "<think>First turn reasoning.</think>First answer." in decoded
    assert "<think>Second turn reasoning.</think>Second answer." in decoded


def test_strip_thinking_from_history_replaces_with_empty_block(glm_tokenizer):
    """strip_thinking_from_history=True strips thinking before the last user message."""
    renderer = Glm5_3Renderer(glm_tokenizer, strip_thinking_from_history=True)
    messages = get_multiturn_thinking_conversation()
    decoded = glm_tokenizer.decode(renderer.build_supervised_example(messages)[0].to_ints())
    assert "First turn reasoning" not in decoded
    assert "<|assistant|><think></think>First answer." in decoded
    assert "<think>Second turn reasoning.</think>Second answer." in decoded


def test_get_stop_sequences(glm_tokenizer, glm_renderer):
    """Stop sequences are the model's eos tokens: <|user|>, <|observation|>, <|endoftext|>."""
    expected = [
        glm_tokenizer.encode(t, add_special_tokens=False)[0]
        for t in ("<|user|>", "<|observation|>", "<|endoftext|>")
    ]
    assert glm_renderer.get_stop_sequences() == expected


# =============================================================================
# HF Template Compatibility Tests — Generation
# =============================================================================


def test_basic_conversation_generation_matches_hf(glm_tokenizer, glm_renderer):
    messages = get_basic_conversation_for_generation()
    cookbook = glm_renderer.build_generation_prompt(messages).to_ints()
    hf = _hf_generation_tokens(glm_tokenizer, [glm_renderer.to_openai_message(m) for m in messages])
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )


def test_low_reasoning_generation_matches_hf(glm_tokenizer, glm_renderer_low_reasoning):
    messages = get_basic_conversation_for_generation()
    r = glm_renderer_low_reasoning
    cookbook = r.build_generation_prompt(messages).to_ints()
    hf = _hf_generation_tokens(
        glm_tokenizer, [r.to_openai_message(m) for m in messages], reasoning_effort="low"
    )
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )


def test_high_reasoning_generation_matches_hf(glm_tokenizer, glm_renderer_high_reasoning):
    messages = get_basic_conversation_for_generation()
    r = glm_renderer_high_reasoning
    cookbook = r.build_generation_prompt(messages).to_ints()
    hf = _hf_generation_tokens(
        glm_tokenizer, [r.to_openai_message(m) for m in messages], reasoning_effort="high"
    )
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )


def test_unrecognized_reasoning_effort_renders_as_max(glm_tokenizer, glm_renderer):
    """The HF template maps anything outside low/high to Max — the default renderer."""
    messages = get_basic_conversation_for_generation()
    cookbook = glm_renderer.build_generation_prompt(messages).to_ints()
    hf = _hf_generation_tokens(
        glm_tokenizer,
        [glm_renderer.to_openai_message(m) for m in messages],
        reasoning_effort="medium",
    )
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )


def test_multiturn_thinking_generation_matches_hf(glm_tokenizer, glm_renderer):
    """Historical thinking preservation in generation prompts matches HF's default."""
    messages = get_multiturn_thinking_conversation()[:-1]
    cookbook = glm_renderer.build_generation_prompt(messages).to_ints()
    hf = _hf_generation_tokens(glm_tokenizer, [glm_renderer.to_openai_message(m) for m in messages])
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )
    assert "First turn reasoning" in glm_tokenizer.decode(cookbook)


def test_embedded_think_string_generation_matches_hf(glm_tokenizer, glm_renderer):
    """String content with embedded <think> tags matches HF's split-and-strip handling."""
    messages = get_embedded_think_string_conversation()
    cookbook = glm_renderer.build_generation_prompt(messages).to_ints()
    hf = _hf_generation_tokens(glm_tokenizer, [glm_renderer.to_openai_message(m) for m in messages])
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )
    decoded = glm_tokenizer.decode(cookbook)
    # Reasoning extracted from the string is preserved (clear_thinking=False
    # default); the text after </think> is stripped of surrounding whitespace.
    assert "<|assistant|><think>first reasoning</think>First answer." in decoded


def test_embedded_think_string_stripped_with_clear_thinking(glm_tokenizer):
    """strip_thinking_from_history=True matches HF clear_thinking=True for string content."""
    renderer = Glm5_3Renderer(glm_tokenizer, strip_thinking_from_history=True)
    messages = get_embedded_think_string_conversation()
    cookbook = renderer.build_generation_prompt(messages).to_ints()
    hf = _hf_generation_tokens(
        glm_tokenizer,
        [renderer.to_openai_message(m) for m in messages],
        clear_thinking=True,
    )
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )
    decoded = glm_tokenizer.decode(cookbook)
    assert "first reasoning" not in decoded
    assert "<|assistant|><think></think>First answer." in decoded


def test_post_tool_generation_matches_hf(glm_tokenizer, glm_renderer):
    """Generation prompt right after a tool response still prefills <think>."""
    messages, tools = get_tool_call_conversation_for_generation()
    openai_tools = [{"type": "function", "function": tool} for tool in tools]

    prefix = glm_renderer.create_conversation_prefix_with_tools(
        tools, system_prompt="You are a helpful assistant."
    )
    cookbook = glm_renderer.build_generation_prompt(prefix + messages).to_ints()

    hf_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        *[glm_renderer.to_openai_message(m) for m in messages],
    ]
    hf = _hf_generation_tokens(glm_tokenizer, hf_messages, tools=openai_tools)
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )
    assert glm_tokenizer.decode(cookbook).endswith("</tool_response><|assistant|><think>")


def test_parallel_tool_calls_generation_matches_hf(glm_tokenizer, glm_renderer):
    """Multiple tool calls with typed values and consecutive tool responses match HF."""
    messages, tools = get_parallel_tool_call_conversation()
    openai_tools = [{"type": "function", "function": tool} for tool in tools]

    prefix = glm_renderer.create_conversation_prefix_with_tools(tools)
    cookbook = glm_renderer.build_generation_prompt(prefix + messages).to_ints()

    hf_messages = [glm_renderer.to_openai_message(m) for m in messages]
    hf = _hf_generation_tokens(glm_tokenizer, hf_messages, tools=openai_tools)
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )

    decoded = glm_tokenizer.decode(cookbook)
    # String values render raw, non-string values as JSON
    assert "<arg_key>location</arg_key><arg_value>SF</arg_value>" in decoded
    assert "<arg_key>days</arg_key><arg_value>3</arg_value>" in decoded
    assert '<arg_key>options</arg_key><arg_value>{"units": "F"}</arg_value>' in decoded
    # Consecutive tool responses share a single <|observation|> token
    assert decoded.count("<|observation|>") == 1
    assert "</tool_response><tool_response>" in decoded


# =============================================================================
# Tool Response Ordering Tests
# =============================================================================
#
# GLM-5.3 re-sorts consecutive tool responses to match the assistant's
# tool_calls order when every id is unique and accounted for, and falls back to
# as-given message order on any ambiguity. Each test asserts both the expected
# cookbook ordering and exact agreement with the HF template.


def _tool_ordering_tokens(glm_renderer, messages, tools):
    prefix = glm_renderer.create_conversation_prefix_with_tools(tools)
    return glm_renderer.build_generation_prompt(prefix + messages).to_ints()


def _tool_ordering_hf_tokens(glm_tokenizer, glm_renderer, messages, tools):
    openai_tools = [{"type": "function", "function": tool} for tool in tools]
    hf_messages = [glm_renderer.to_openai_message(m) for m in messages]
    return _hf_generation_tokens(glm_tokenizer, hf_messages, tools=openai_tools)


def test_out_of_order_tool_results_sorted_by_tool_call_order(glm_tokenizer, glm_renderer):
    """Results arriving out of order are re-sorted to tool-call order, matching HF."""
    messages, tools = get_parallel_tool_call_conversation(result_ids=("call_2", "call_1"))
    cookbook = _tool_ordering_tokens(glm_renderer, messages, tools)
    hf = _tool_ordering_hf_tokens(glm_tokenizer, glm_renderer, messages, tools)
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )
    decoded = glm_tokenizer.decode(cookbook)
    # call_1's result ({"temp": 65} was given first as call_2's) renders first
    assert (
        '<tool_response>{"temp": 65}</tool_response><tool_response>{"temp": 72}</tool_response>'
        in decoded
    )
    assert decoded.count("<|observation|>") == 1


def test_out_of_order_tool_results_sorted_in_supervised_example(glm_tokenizer, glm_renderer):
    """The re-sort also applies to supervised examples."""
    messages, tools = get_parallel_tool_call_conversation(result_ids=("call_2", "call_1"))
    messages.append(Message(role="assistant", content="NYC is warmer."))
    openai_tools = [{"type": "function", "function": tool} for tool in tools]

    prefix = glm_renderer.create_conversation_prefix_with_tools(tools)
    cookbook = glm_renderer.build_supervised_example(prefix + messages)[0].to_ints()

    hf_messages = [glm_renderer.to_openai_message(m) for m in messages]
    hf = _hf_supervised_tokens(glm_tokenizer, hf_messages, tools=openai_tools)
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )
    assert (
        '<tool_response>{"temp": 65}</tool_response><tool_response>{"temp": 72}</tool_response>'
        in glm_tokenizer.decode(cookbook)
    )


def test_partial_tool_results_sorted_by_tool_call_order(glm_tokenizer, glm_renderer):
    """A lone result for the second call renders alone (results may be a subset of calls)."""
    messages, tools = get_parallel_tool_call_conversation()
    del messages[2]  # drop call_1's result, keeping only call_2's
    cookbook = _tool_ordering_tokens(glm_renderer, messages, tools)
    hf = _tool_ordering_hf_tokens(glm_tokenizer, glm_renderer, messages, tools)
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )
    decoded = glm_tokenizer.decode(cookbook)
    assert '<tool_response>{"temp": 65}</tool_response>' in decoded
    assert '{"temp": 72}' not in decoded


@pytest.mark.parametrize(
    "call_ids,result_ids",
    [
        # A result without a tool_call_id
        (("call_1", "call_2"), ("call_2", None)),
        # Duplicate result ids
        (("call_1", "call_2"), ("call_2", "call_2")),
        # A result id that matches no tool call
        (("call_1", "call_2"), ("call_2", "call_unknown")),
        # Tool calls without ids
        ((None, None), ("call_2", "call_1")),
    ],
    ids=["missing-result-id", "dup-result-id", "unknown-result-id", "no-call-ids"],
)
def test_ambiguous_tool_result_ids_keep_message_order(
    call_ids, result_ids, glm_tokenizer, glm_renderer
):
    """Any id ambiguity falls back to as-given message order, matching HF."""
    messages, tools = get_parallel_tool_call_conversation(call_ids=call_ids, result_ids=result_ids)
    cookbook = _tool_ordering_tokens(glm_renderer, messages, tools)
    hf = _tool_ordering_hf_tokens(glm_tokenizer, glm_renderer, messages, tools)
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )
    # Message order preserved: {"temp": 72} was given first
    assert (
        '<tool_response>{"temp": 72}</tool_response><tool_response>{"temp": 65}</tool_response>'
        in glm_tokenizer.decode(cookbook)
    )


def test_duplicate_tool_call_ids_keep_message_order(glm_tokenizer, glm_renderer):
    """Duplicate ids among the tool_calls themselves also disable the re-sort.

    With a single result matching a duplicated call id, a (buggy) sorted render
    would emit the response once per matching tool call, so exact HF agreement
    here confirms both sides take the fallback path.
    """
    messages, tools = get_parallel_tool_call_conversation(
        call_ids=("call_1", "call_1"), result_ids=("call_1", "call_1")
    )
    del messages[3]  # keep a single result so all result-side checks pass
    cookbook = _tool_ordering_tokens(glm_renderer, messages, tools)
    hf = _tool_ordering_hf_tokens(glm_tokenizer, glm_renderer, messages, tools)
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )
    assert glm_tokenizer.decode(cookbook).count('<tool_response>{"temp": 72}</tool_response>') == 1


def test_tool_results_without_preceding_tool_calls_keep_message_order(glm_tokenizer, glm_renderer):
    """A tool block whose preceding assistant message has no tool_calls is not re-sorted."""
    messages, tools = get_parallel_tool_call_conversation(result_ids=("call_2", "call_1"))
    del messages[1]["tool_calls"]
    cookbook = _tool_ordering_tokens(glm_renderer, messages, tools)
    hf = _tool_ordering_hf_tokens(glm_tokenizer, glm_renderer, messages, tools)
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )
    assert (
        '<tool_response>{"temp": 72}</tool_response><tool_response>{"temp": 65}</tool_response>'
        in glm_tokenizer.decode(cookbook)
    )


# =============================================================================
# HF Template Compatibility Tests — Supervised
# =============================================================================


def test_basic_conversation_supervised_matches_hf(glm_tokenizer, glm_renderer):
    """Supervised example matches HF plus the trailing <|user|> terminator."""
    messages = get_basic_conversation_for_supervised()
    cookbook = glm_renderer.build_supervised_example(messages)[0].to_ints()
    hf = _hf_supervised_tokens(glm_tokenizer, [glm_renderer.to_openai_message(m) for m in messages])
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )


def test_thinking_conversation_supervised_matches_hf(glm_tokenizer, glm_renderer):
    messages = get_thinking_conversation_for_supervised()
    cookbook = glm_renderer.build_supervised_example(messages)[0].to_ints()
    hf = _hf_supervised_tokens(glm_tokenizer, [glm_renderer.to_openai_message(m) for m in messages])
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )


def test_multiturn_thinking_supervised_matches_hf(glm_tokenizer, glm_renderer):
    """The default preserves all thinking, matching HF's clear_thinking=False default."""
    messages = get_multiturn_thinking_conversation()
    cookbook = glm_renderer.build_supervised_example(messages)[0].to_ints()
    hf = _hf_supervised_tokens(glm_tokenizer, [glm_renderer.to_openai_message(m) for m in messages])
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )
    decoded = glm_tokenizer.decode(cookbook)
    assert "First turn reasoning" in decoded
    assert "Second turn reasoning" in decoded


def test_strip_thinking_supervised_matches_hf_clear_thinking_true(glm_tokenizer):
    """strip_thinking_from_history=True matches HF's clear_thinking=True."""
    renderer = Glm5_3Renderer(glm_tokenizer, strip_thinking_from_history=True)
    messages = get_multiturn_thinking_conversation()
    cookbook = renderer.build_supervised_example(messages)[0].to_ints()
    hf = _hf_supervised_tokens(
        glm_tokenizer,
        [renderer.to_openai_message(m) for m in messages],
        clear_thinking=True,
    )
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )
    decoded = glm_tokenizer.decode(cookbook)
    assert "First turn reasoning" not in decoded
    assert "Second turn reasoning" in decoded


def test_low_reasoning_supervised_matches_hf(glm_tokenizer, glm_renderer_low_reasoning):
    """Low-reasoning supervised example matches HF with reasoning_effort='low'."""
    messages = get_basic_conversation_for_supervised()
    r = glm_renderer_low_reasoning
    cookbook = r.build_supervised_example(messages)[0].to_ints()
    hf = _hf_supervised_tokens(
        glm_tokenizer, [r.to_openai_message(m) for m in messages], reasoning_effort="low"
    )
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )


def test_tool_call_conversation_supervised_matches_hf(glm_tokenizer, glm_renderer):
    messages, tools = get_tool_call_conversation_for_supervised()
    openai_tools = [{"type": "function", "function": tool} for tool in tools]
    system_prompt = "You are a helpful assistant."

    prefix = glm_renderer.create_conversation_prefix_with_tools(tools, system_prompt=system_prompt)
    cookbook = glm_renderer.build_supervised_example(prefix + messages)[0].to_ints()

    hf_messages = [
        {"role": "system", "content": system_prompt},
        *[glm_renderer.to_openai_message(m) for m in messages],
    ]
    hf = _hf_supervised_tokens(glm_tokenizer, hf_messages, tools=openai_tools)
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )


def test_supervised_ending_with_tool_call_uses_observation_terminator(glm_tokenizer, glm_renderer):
    """A supervised target that issues tool calls ends with <|observation|>, not <|user|>."""
    messages, tools = get_tool_call_conversation_for_supervised()
    openai_tools = [{"type": "function", "function": tool} for tool in tools]

    # Truncate right after the tool-calling assistant message
    messages = messages[:2]
    prefix = glm_renderer.create_conversation_prefix_with_tools(tools)
    cookbook = glm_renderer.build_supervised_example(prefix + messages)[0].to_ints()

    hf_messages = [glm_renderer.to_openai_message(m) for m in messages]
    hf = _hf_supervised_tokens(
        glm_tokenizer, hf_messages, terminator="<|observation|>", tools=openai_tools
    )
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )
    assert glm_tokenizer.decode(cookbook).endswith("</tool_call><|observation|>")


# =============================================================================
# Turn Terminator Weighting Tests
# =============================================================================


def _weighted_tokens(renderer, tokenizer, messages, train_on_what):
    """Decode only the tokens that carry loss."""
    model_input, weights = renderer.build_supervised_example(messages, train_on_what=train_on_what)
    tokens = model_input.to_ints()
    return [t for t, w in zip(tokens, weights.tolist()) if w > 0]


def test_every_assistant_turn_trains_its_terminator(glm_tokenizer, glm_renderer):
    """GLM has no per-message end token, so the terminator is the next role token.

    It must still carry loss on every trained turn -- otherwise a multi-turn
    supervised example only ever teaches the model to stop once, at the end.
    """
    messages = [
        Message(role="user", content="q1"),
        Message(role="assistant", content="<think>t1</think>a1"),
        Message(role="user", content="q2"),
        Message(role="assistant", content="<think>t2</think>a2"),
        Message(role="user", content="q3"),
        Message(role="assistant", content="<think>t3</think>a3"),
    ]
    user_token = glm_tokenizer.encode("<|user|>", add_special_tokens=False)[0]
    trained = _weighted_tokens(
        glm_renderer, glm_tokenizer, messages, TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    assert trained.count(user_token) == 3

    # LAST_ASSISTANT_MESSAGE trains exactly the one turn it targets.
    trained_last = _weighted_tokens(
        glm_renderer, glm_tokenizer, messages, TrainOnWhat.LAST_ASSISTANT_MESSAGE
    )
    assert trained_last.count(user_token) == 1


def test_tool_calling_turn_trains_observation_terminator(glm_tokenizer, glm_renderer):
    """A turn that hands off to tools is terminated by <|observation|>."""
    messages = [
        Message(role="user", content="weather?"),
        Message(
            role="assistant",
            content="<think>t1</think>",
            tool_calls=[
                ToolCall(
                    id="c1",
                    function=ToolCall.FunctionBody(
                        name="get_weather", arguments=json.dumps({"city": "SF"})
                    ),
                )
            ],
        ),
        Message(role="tool", content="sunny", tool_call_id="c1"),
        Message(role="assistant", content="<think>t2</think>It is sunny."),
    ]
    observation_token = glm_tokenizer.encode("<|observation|>", add_special_tokens=False)[0]
    user_token = glm_tokenizer.encode("<|user|>", add_special_tokens=False)[0]
    trained = _weighted_tokens(
        glm_renderer, glm_tokenizer, messages, TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    assert trained.count(observation_token) == 1
    assert trained.count(user_token) == 1


def test_think_prefill_never_trained(glm_tokenizer, glm_renderer):
    """Sampling always supplies <think>, at every turn, so it must never carry loss."""
    messages = [
        Message(role="user", content="q1"),
        Message(role="assistant", content="<think>t1</think>a1"),
        Message(role="user", content="q2"),
        Message(role="assistant", content="<think>t2</think>a2"),
    ]
    think_token = glm_tokenizer.encode("<think>", add_special_tokens=False)[0]
    for train_on_what in (
        TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ):
        trained = _weighted_tokens(glm_renderer, glm_tokenizer, messages, train_on_what)
        assert think_token not in trained, train_on_what


def test_terminator_left_alone_when_next_message_does_not_consume_it(glm_tokenizer, glm_renderer):
    """A mid-conversation system message is not something the model ends a turn with.

    The assistant keeps no terminator, the system message keeps its own header,
    and the token sequence is unchanged.
    """
    messages = [
        Message(role="user", content="q1"),
        Message(role="assistant", content="<think>t1</think>a1"),
        Message(role="system", content="new rules"),
        Message(role="user", content="q2"),
        Message(role="assistant", content="<think>t2</think>a2"),
    ]
    user_token = glm_tokenizer.encode("<|user|>", add_special_tokens=False)[0]
    trained = _weighted_tokens(
        glm_renderer, glm_tokenizer, messages, TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    # Only the final turn's terminator; the first turn hands off to <|system|>.
    assert trained.count(user_token) == 1
    rendered = glm_tokenizer.decode(
        glm_renderer.build_supervised_example(
            messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )[0].to_ints()
    )
    assert "a1<|system|>new rules<|user|>q2" in rendered


@pytest.mark.parametrize(
    "train_on_what",
    [TrainOnWhat.ALL_ASSISTANT_MESSAGES, TrainOnWhat.LAST_ASSISTANT_MESSAGE],
)
def test_terminator_ownership_does_not_change_tokens(glm_tokenizer, glm_renderer, train_on_what):
    """Moving the terminator between chunks must leave the rendering byte-identical to HF."""
    messages = get_multiturn_thinking_conversation()
    cookbook = glm_renderer.build_supervised_example(messages, train_on_what=train_on_what)[
        0
    ].to_ints()
    hf_messages = [glm_renderer.to_openai_message(m) for m in messages]
    hf = _hf_supervised_tokens(glm_tokenizer, hf_messages)
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )


# =============================================================================
# Tool Declaration Tests
# =============================================================================


def test_create_conversation_prefix_tools_before_system(glm_renderer):
    """The tools declaration is a separate system message BEFORE the system prompt."""
    tools = [get_tool_spec()]
    prefix = glm_renderer.create_conversation_prefix_with_tools(
        tools, system_prompt="You are helpful."
    )
    assert len(prefix) == 2
    assert prefix[0]["role"] == "system"
    assert prefix[1]["role"] == "system"
    tools_content = prefix[0]["content"]
    assert isinstance(tools_content, str)
    assert tools_content.startswith("\n# Tools")
    assert "<tools>" in tools_content
    assert '"name": "get_weather"' in tools_content
    assert prefix[1]["content"] == "You are helpful."


def test_create_conversation_prefix_without_system_prompt(glm_renderer):
    """Without a system prompt, only the tools declaration message is returned."""
    prefix = glm_renderer.create_conversation_prefix_with_tools([get_tool_spec()])
    assert len(prefix) == 1
    content = prefix[0]["content"]
    assert isinstance(content, str)
    assert "# Tools" in content


def test_create_conversation_prefix_no_tools(glm_renderer):
    """No tools: returns just the system prompt message."""
    prefix = glm_renderer.create_conversation_prefix_with_tools([], system_prompt="Be brief.")
    assert prefix == [Message(role="system", content="Be brief.")]


def _tool_spec_with(**extra) -> ToolSpec:
    """A get_weather ToolSpec extended with optional metadata keys (strict, defer_loading)."""
    tool = dict(get_tool_spec())
    tool.update(extra)
    return cast(ToolSpec, tool)


def _assert_tools_generation_matches_hf(glm_tokenizer, glm_renderer, tools: list[ToolSpec]):
    """Assert cookbook and HF render identical generation prompts for the given tools."""
    openai_tools = [{"type": "function", "function": tool} for tool in tools]
    prefix = glm_renderer.create_conversation_prefix_with_tools(tools, system_prompt="Sys.")
    messages = prefix + [Message(role="user", content="Hi")]
    cookbook = glm_renderer.build_generation_prompt(messages).to_ints()
    hf_messages = [
        {"role": "system", "content": "Sys."},
        {"role": "user", "content": "Hi"},
    ]
    hf = _hf_generation_tokens(glm_tokenizer, hf_messages, tools=openai_tools)
    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )
    return prefix


def test_tool_declaration_filters_strict_key(glm_tokenizer, glm_renderer):
    """The template's tool_to_json macro drops the strict key from declarations."""
    prefix = _assert_tools_generation_matches_hf(
        glm_tokenizer, glm_renderer, [_tool_spec_with(strict=True)]
    )
    content = prefix[0]["content"]
    assert isinstance(content, str)
    assert '"name": "get_weather"' in content
    assert "strict" not in content


def test_deferred_tool_skipped_in_declaration(glm_tokenizer, glm_renderer):
    """Tools with defer_loading=True are omitted from the <tools> block entirely."""
    deferred = _tool_spec_with(defer_loading=True)
    kept = ToolSpec(name="get_time", description="Get the current time", parameters={})
    prefix = _assert_tools_generation_matches_hf(glm_tokenizer, glm_renderer, [deferred, kept])
    content = prefix[0]["content"]
    assert isinstance(content, str)
    assert "get_weather" not in content
    assert '"name": "get_time"' in content
    assert "defer_loading" not in content


def test_defer_loading_false_tool_kept(glm_tokenizer, glm_renderer):
    """defer_loading=False keeps the tool but drops the key itself."""
    prefix = _assert_tools_generation_matches_hf(
        glm_tokenizer, glm_renderer, [_tool_spec_with(defer_loading=False)]
    )
    content = prefix[0]["content"]
    assert isinstance(content, str)
    assert '"name": "get_weather"' in content
    assert "defer_loading" not in content


def test_all_deferred_tools_render_empty_tools_block(glm_tokenizer, glm_renderer):
    """When every tool is deferred, the tools system block still renders, with an empty list."""
    prefix = _assert_tools_generation_matches_hf(
        glm_tokenizer, glm_renderer, [_tool_spec_with(defer_loading=True)]
    )
    content = prefix[0]["content"]
    assert isinstance(content, str)
    assert "<tools>\n</tools>" in content
    assert "get_weather" not in content


@pytest.mark.parametrize("build_mode", ["generation", "supervised"])
def test_tool_declaration_matches_hf(build_mode: str, glm_tokenizer, glm_renderer):
    """Tool declarations match HF template output exactly."""
    tools = [get_tool_spec()]
    openai_tools = [{"type": "function", "function": tool} for tool in tools]
    system_prompt = "You are a helpful assistant."

    prefix = glm_renderer.create_conversation_prefix_with_tools(tools, system_prompt=system_prompt)
    user_msg = Message(role="user", content="What's the weather in NYC?")

    hf_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What's the weather in NYC?"},
    ]

    if build_mode == "generation":
        cookbook = glm_renderer.build_generation_prompt(prefix + [user_msg]).to_ints()
        hf = _hf_generation_tokens(glm_tokenizer, hf_messages, tools=openai_tools)
    else:
        assistant_msg = Message(role="assistant", content="Let me check that for you.")
        cookbook = glm_renderer.build_supervised_example(prefix + [user_msg, assistant_msg])[
            0
        ].to_ints()
        hf_messages.append({"role": "assistant", "content": "Let me check that for you."})
        hf = _hf_supervised_tokens(glm_tokenizer, hf_messages, tools=openai_tools)

    assert cookbook == hf, (
        f"Cookbook: {glm_tokenizer.decode(cookbook)}\nHF: {glm_tokenizer.decode(hf)}"
    )


# =============================================================================
# Parse Response Tests
# =============================================================================


def test_parse_response_plain_text(glm_tokenizer, glm_renderer):
    """A plain response terminated by <|user|> parses cleanly."""
    tokens = glm_tokenizer.encode("The answer is 42.<|user|>", add_special_tokens=False)
    message, termination = glm_renderer.parse_response(tokens)
    assert termination.is_stop_sequence
    assert message["content"] == "The answer is 42."


def test_parse_response_with_thinking(glm_tokenizer, glm_renderer):
    """The <think> prefill is restored so reasoning parses into a ThinkingPart."""
    # Simulates what the model generates after the <|assistant|><think> prefill
    response_text = "I should reason carefully.</think>The answer is 42.<|user|>"
    tokens = glm_tokenizer.encode(response_text, add_special_tokens=False)
    message, termination = glm_renderer.parse_response(tokens)

    assert termination.is_stop_sequence
    content = message["content"]
    assert isinstance(content, list)
    assert content == [
        {"type": "thinking", "thinking": "I should reason carefully."},
        {"type": "text", "text": "The answer is 42."},
    ]


def test_parse_response_endoftext_is_eos(glm_tokenizer, glm_renderer):
    """<|endoftext|> is a clean EOS termination (but not the expected stop sequence)."""
    tokens = glm_tokenizer.encode("Done.<|endoftext|>", add_special_tokens=False)
    message, termination = glm_renderer.parse_response(tokens)
    assert termination.is_clean
    assert not termination.is_stop_sequence
    assert message["content"] == "Done."


def test_parse_response_missing_stop_is_malformed(glm_tokenizer, glm_renderer):
    """A truncated response (no stop token) is MALFORMED but still returns content."""
    tokens = glm_tokenizer.encode("Truncated answ", add_special_tokens=False)
    message, termination = glm_renderer.parse_response(tokens)
    assert not termination.is_clean
    assert message["content"] == "Truncated answ"


def test_parse_response_double_stop_raises(glm_tokenizer, glm_renderer):
    """Multiple stop tokens indicate a sampler misconfiguration."""
    tokens = glm_tokenizer.encode("Hi<|user|>Bye<|user|>", add_special_tokens=False)
    with pytest.raises(ValueError, match=r"expected .* 1"):
        glm_renderer.parse_response(tokens)


def test_parse_response_tool_call(glm_tokenizer, glm_renderer):
    """Tool calls in GLM's <arg_key>/<arg_value> format parse into ToolCall objects."""
    response_text = (
        "I should check the weather.</think>"
        "<tool_call>get_weather"
        "<arg_key>location</arg_key><arg_value>New York, NY</arg_value>"
        "<arg_key>days</arg_key><arg_value>3</arg_value>"
        '<arg_key>options</arg_key><arg_value>{"units": "F"}</arg_value>'
        "</tool_call><|observation|>"
    )
    tokens = glm_tokenizer.encode(response_text, add_special_tokens=False)
    message, termination = glm_renderer.parse_response(tokens)

    assert termination.is_stop_sequence
    tool_calls = message.get("tool_calls", [])
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    args = json.loads(tool_calls[0].function.arguments)
    # String values are raw, non-string values are JSON-typed
    assert args == {"location": "New York, NY", "days": 3, "options": {"units": "F"}}
    assert "<tool_call>" not in get_text_content(message)


def test_parse_response_multiple_tool_calls(glm_tokenizer, glm_renderer):
    """Multiple back-to-back tool calls all parse."""
    response_text = (
        "check both</think>"
        "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>"
        "<tool_call>get_weather<arg_key>location</arg_key><arg_value>LA</arg_value></tool_call>"
        "<|observation|>"
    )
    tokens = glm_tokenizer.encode(response_text, add_special_tokens=False)
    message, termination = glm_renderer.parse_response(tokens)

    assert termination.is_stop_sequence
    tool_calls = message.get("tool_calls", [])
    assert len(tool_calls) == 2
    assert json.loads(tool_calls[0].function.arguments) == {"location": "NYC"}
    assert json.loads(tool_calls[1].function.arguments) == {"location": "LA"}


def test_parse_response_malformed_tool_call(glm_tokenizer, glm_renderer):
    """A malformed tool call block is captured as unparsed_tool_calls."""
    response_text = "</think><tool_call>{not glm format}</tool_call><|user|>"
    tokens = glm_tokenizer.encode(response_text, add_special_tokens=False)
    message, termination = glm_renderer.parse_response(tokens)

    assert termination.is_clean
    assert not message.get("tool_calls")
    unparsed = message.get("unparsed_tool_calls", [])
    assert len(unparsed) == 1
    assert "<tool_call>" in unparsed[0].raw_text


def test_parse_response_roundtrips_through_render(glm_tokenizer, glm_renderer):
    """render(parse(tokens)) reproduces the sampled tool call tokens."""
    response_text = (
        "check</think>Checking now."
        "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>"
        "<|observation|>"
    )
    tokens = glm_tokenizer.encode(response_text, add_special_tokens=False)
    message, _ = glm_renderer.parse_response(tokens)

    messages = [Message(role="user", content="Weather in NYC?"), message]
    rendered = glm_renderer.build_supervised_example(messages)[0].to_ints()
    decoded = glm_tokenizer.decode(rendered)
    assert decoded.endswith(
        "<|assistant|><think>check</think>Checking now."
        "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>"
        "<|observation|>"
    )


# =============================================================================
# Supervised / Generation / Parse Consistency
# =============================================================================


@pytest.mark.parametrize(
    "renderer_name", ["glm5_3_max_reasoning", "glm5_3_low_reasoning", "glm5_3_high_reasoning"]
)
def test_observation_matches_generation_prompt(renderer_name, glm_tokenizer):
    """The weight-0 prefix of a supervised example equals the generation prompt.

    The thinking prefill lives in the zero-weight header of the supervised
    target, so the observation/action split is consistent with sampling.
    """
    renderer = get_renderer(renderer_name, glm_tokenizer)
    messages = get_thinking_conversation_for_supervised()

    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
    )
    tokens = model_input.to_ints()
    weights_list = weights.tolist()
    first_weighted = weights_list.index(1.0)
    observation, action = tokens[:first_weighted], tokens[first_weighted:]

    gen_tokens = renderer.build_generation_prompt(messages[:-1]).to_ints()
    assert observation == gen_tokens, (
        f"Observation: {glm_tokenizer.decode(observation)!r}\n"
        f"Generation prompt: {glm_tokenizer.decode(gen_tokens)!r}"
    )

    parsed, termination = renderer.parse_response(action)
    assert termination.is_stop_sequence
    assert ensure_list(parsed["content"]) == ensure_list(messages[-1]["content"])


# =============================================================================
# Streaming Tests
# =============================================================================


def test_parse_response_streaming(glm_tokenizer, glm_renderer):
    """Streaming yields thinking deltas, text deltas, and a complete final Message."""
    response_text = "Let me think.</think>The answer is 42.<|user|>"
    tokens = glm_tokenizer.encode(response_text, add_special_tokens=False)

    deltas = list(glm_renderer.parse_response_streaming(tokens))

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


def test_parse_response_streaming_observation_stop(glm_tokenizer, glm_renderer):
    """Streaming handles <|observation|>-terminated (tool call) responses."""
    response_text = (
        "check</think><tool_call>get_weather<arg_key>location</arg_key>"
        "<arg_value>NYC</arg_value></tool_call><|observation|>"
    )
    tokens = glm_tokenizer.encode(response_text, add_special_tokens=False)

    deltas = list(glm_renderer.parse_response_streaming(tokens))
    final_message = deltas[-1]
    assert isinstance(final_message, dict)
    tool_calls = final_message.get("tool_calls", [])
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"


# =============================================================================
# Renderer Identity Tests
# =============================================================================


def test_renderer_types(glm_renderer, glm_renderer_low_reasoning, glm_renderer_high_reasoning):
    assert isinstance(glm_renderer, Glm5_3Renderer)
    assert isinstance(glm_renderer_low_reasoning, Glm5_3LowReasoningRenderer)
    assert isinstance(glm_renderer_high_reasoning, Glm5_3HighReasoningRenderer)


def test_extension_property_flags(glm_tokenizer):
    # The default (strip_thinking_from_history=False, matching HF's
    # clear_thinking=False default) preserves thinking, so the extension
    # property holds out of the box.
    assert Glm5_3Renderer(glm_tokenizer).has_extension_property
    assert not Glm5_3Renderer(
        glm_tokenizer, strip_thinking_from_history=True
    ).has_extension_property
