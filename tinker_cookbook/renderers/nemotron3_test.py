"""
Tests for Nemotron-3 renderer.

Tests verify that the Nemotron3Renderer produces correct output:
1. Generation prompt ends with <|im_start|>assistant\n<think>\n (thinking enabled)
2. Disable-thinking variant ends with <|im_start|>assistant\n<think></think>
3. Tool declarations use Nemotron-3's structured XML format
4. System prompt comes BEFORE tools in the system message
5. <think></think> is prepended to ALL assistant messages without thinking (not just last)
6. HF template compatibility for both build_generation_prompt and build_supervised_example
7. Low-thinking variant appends {reasoning effort: low} to last user message
"""

import json

import pytest

from tinker_cookbook.renderers import Message, ToolCall, ToolSpec, get_renderer
from tinker_cookbook.renderers.nemotron3 import (
    Nemotron3DisableThinkingRenderer,
    Nemotron3LowThinkingRenderer,
    Nemotron3PreserveThinkingRenderer,
    Nemotron3Renderer,
    Nemotron3UltraDisableThinkingRenderer,
    Nemotron3UltraMediumThinkingRenderer,
    Nemotron3UltraPreserveThinkingRenderer,
    Nemotron3UltraRenderer,
    _format_nemotron3_tool_declaration,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

NEMOTRON3_NANO_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
NEMOTRON3_SUPER_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
NEMOTRON3_ULTRA_MODEL = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
NEMOTRON3_TOKENIZER_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def nemotron_tokenizer():
    return get_tokenizer(NEMOTRON3_TOKENIZER_PATH)


@pytest.fixture(scope="module")
def nemotron_renderer(nemotron_tokenizer):
    return get_renderer("nemotron3", nemotron_tokenizer)


@pytest.fixture(scope="module")
def nemotron_renderer_disable_thinking(nemotron_tokenizer):
    return get_renderer("nemotron3_disable_thinking", nemotron_tokenizer)


def _hf_generation_tokens(
    tokenizer,
    hf_messages,
    tools=None,
    enable_thinking: bool = True,
    low_effort: bool = False,
    medium_effort: bool = False,
    truncate_history_thinking: bool = True,
):
    """Run HF apply_chat_template with generation prompt and return token list."""
    kwargs = {
        "add_generation_prompt": True,
        "tokenize": True,
        "enable_thinking": enable_thinking,
        "truncate_history_thinking": truncate_history_thinking,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if low_effort:
        kwargs["low_effort"] = True
    if medium_effort:
        kwargs["medium_effort"] = True
    result = tokenizer.apply_chat_template(hf_messages, **kwargs)
    # apply_chat_template may return BatchEncoding (dict-like) when tools are provided.
    if hasattr(result, "input_ids"):
        return list(result.input_ids)
    return list(result)


def _hf_supervised_tokens(
    tokenizer,
    hf_messages,
    tools=None,
    enable_thinking: bool = True,
    low_effort: bool = False,
    medium_effort: bool = False,
    truncate_history_thinking: bool = True,
):
    """Run HF apply_chat_template without generation prompt, strip trailing newline, re-encode."""
    kwargs = {
        "add_generation_prompt": False,
        "tokenize": False,
        "enable_thinking": enable_thinking,
        "truncate_history_thinking": truncate_history_thinking,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if low_effort:
        kwargs["low_effort"] = True
    if medium_effort:
        kwargs["medium_effort"] = True
    result = tokenizer.apply_chat_template(hf_messages, **kwargs)
    # apply_chat_template with tokenize=False may return BatchEncoding when tools are provided.
    hf_str = result.input_ids if hasattr(result, "input_ids") else result
    assert isinstance(hf_str, str)
    return tokenizer.encode(hf_str.rstrip("\n"), add_special_tokens=False)


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


def get_rich_tool_spec() -> ToolSpec:
    """Tool spec with extra JSON Schema fields beyond name/type/description/enum."""
    return ToolSpec(
        name="search",
        description="Search for items",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                    "default": "*",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "minimum": 1,
                    "maximum": 100,
                },
                "tags": {
                    "type": "array",
                    "description": "Filter tags",
                    "items": {"type": "string"},
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )


def get_tool_call_conversation_for_generation() -> tuple[list[Message], list[ToolSpec]]:
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
    ]
    return messages, tools


def get_historical_tool_call_with_nonempty_text_conversation() -> tuple[
    list[Message], list[ToolSpec]
]:
    """Conversation where a historical assistant message has thinking + non-empty text + tool_calls.

    This is an edge case where the HF Jinja template's tool_calls branch applies
    ``| trim`` to the content *before* concatenation with ``<think></think>``,
    stripping the leading ``\\n`` that would otherwise be preserved in the
    non-tool_calls branch. The result is ``<think></think>text`` (no newline)
    for the historical message.

    The first assistant message becomes historical because a later user message
    follows the tool response + second assistant exchange.
    """
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
        # This assistant message has thinking + non-empty text + tool_calls
        # and will be historical (before the last user message).
        Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "I need to check the weather."},
                {"type": "text", "text": "Let me check that for you."},
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
                {"type": "text", "text": "It's 72°F and sunny in NYC."},
            ],
        ),
        Message(role="user", content="Thanks!"),
    ]
    return messages, tools


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


# =============================================================================
# Tool Declaration Format Tests (no tokenizer required)
# =============================================================================


def test_tool_declaration_xml_format():
    """Tool declarations use Nemotron-3's structured XML format."""
    tool = get_tool_spec()
    declaration = _format_nemotron3_tool_declaration(tool)

    assert "<function>" in declaration
    assert "<name>get_weather</name>" in declaration
    assert "<description>Get the current weather for a location</description>" in declaration
    assert "<parameters>" in declaration
    assert "<parameter>" in declaration
    assert "<name>location</name>" in declaration
    assert "<type>string</type>" in declaration
    assert "<name>unit</name>" in declaration
    assert "<enum>" in declaration
    assert '"celsius"' in declaration
    assert "<required>" in declaration
    assert '"location"' in declaration
    assert "</function>" in declaration


def test_tool_declaration_not_json_per_line():
    """Tool declarations should NOT use Qwen3.5's JSON-per-line format."""
    tool = get_tool_spec()
    declaration = _format_nemotron3_tool_declaration(tool)
    assert not declaration.strip().startswith("{")
    assert '"name": "get_weather"' not in declaration


def test_tool_declaration_minimal_tool():
    """Tool with no description and no parameters."""
    tool = ToolSpec(name="ping", description="", parameters={})
    declaration = _format_nemotron3_tool_declaration(tool)
    assert "<name>ping</name>" in declaration
    assert "<description>" not in declaration
    assert "<parameter>" not in declaration
    assert "<required>" not in declaration


def test_tool_declaration_extra_schema_keys_match_hf(nemotron_tokenizer, nemotron_renderer):
    """Tool with extra JSON Schema fields (default, minimum, items, etc.) matches HF.

    The HF Jinja template has a render_extra_keys macro that outputs additional
    parameter fields beyond name/type/description/enum. This test verifies our
    renderer handles those extra keys the same way.
    """
    tools = [get_rich_tool_spec()]
    openai_tools = [{"type": "function", "function": tool} for tool in tools]
    system_prompt = "You are a helpful assistant."

    prefix = nemotron_renderer.create_conversation_prefix_with_tools(
        tools, system_prompt=system_prompt
    )
    messages = prefix + [Message(role="user", content="Search for cats")]
    cookbook = nemotron_renderer.build_generation_prompt(messages).to_ints()

    hf_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Search for cats"},
    ]
    hf = _hf_generation_tokens(nemotron_tokenizer, hf_messages, tools=openai_tools)

    assert cookbook == hf, (
        f"Cookbook: {nemotron_tokenizer.decode(cookbook)}\nHF: {nemotron_tokenizer.decode(hf)}"
    )


def test_create_conversation_prefix_system_before_tools(nemotron_renderer):
    """System prompt should appear BEFORE tools block."""
    tools = [get_tool_spec()]
    system_prompt = "You are a helpful assistant."
    prefix = nemotron_renderer.create_conversation_prefix_with_tools(tools, system_prompt)

    assert len(prefix) == 1
    assert prefix[0]["role"] == "system"
    content = prefix[0]["content"]
    assert isinstance(content, str)

    sysprompt_idx = content.index(system_prompt)
    tools_idx = content.index("# Tools")
    assert sysprompt_idx < tools_idx


def test_create_conversation_prefix_without_system_prompt(nemotron_renderer):
    """Without system_prompt, content starts directly with # Tools."""
    tools = [get_tool_spec()]
    prefix = nemotron_renderer.create_conversation_prefix_with_tools(tools)
    content = prefix[0]["content"]
    assert isinstance(content, str)
    assert content.startswith("# Tools")


def test_create_conversation_prefix_xml_tool_format(nemotron_renderer):
    """Tool declarations in prefix use XML format, not JSON."""
    tools = [get_tool_spec()]
    prefix = nemotron_renderer.create_conversation_prefix_with_tools(tools)
    content = prefix[0]["content"]
    assert "<tools>" in content
    assert "<function>" in content
    assert "<name>get_weather</name>" in content
    assert "<parameter>" in content
    assert "</tools>" in content
    assert '{"name": "get_weather"' not in content


def test_create_conversation_prefix_no_tools(nemotron_renderer):
    """No tools: returns just the system_prompt."""
    prefix = nemotron_renderer.create_conversation_prefix_with_tools(
        [], system_prompt="You are helpful."
    )
    assert prefix[0]["content"] == "You are helpful."


# =============================================================================
# Generation Prompt Tests
# =============================================================================


def test_generation_prompt_ends_with_think(nemotron_tokenizer, nemotron_renderer):
    """Nemotron3Renderer prefills with <think>\\n."""
    messages = get_basic_conversation_for_generation()
    decoded = nemotron_tokenizer.decode(
        nemotron_renderer.build_generation_prompt(messages).to_ints()
    )
    assert decoded.endswith("<|im_start|>assistant\n<think>\n")


def test_disable_thinking_generation_prompt(nemotron_tokenizer, nemotron_renderer_disable_thinking):
    """Nemotron3DisableThinkingRenderer prefills with <think></think>."""
    messages = get_basic_conversation_for_generation()
    decoded = nemotron_tokenizer.decode(
        nemotron_renderer_disable_thinking.build_generation_prompt(messages).to_ints()
    )
    assert decoded.endswith("<|im_start|>assistant\n<think></think>")


def test_custom_prefill_overrides_think(nemotron_tokenizer, nemotron_renderer):
    """Custom prefill overrides the default <think>\\n."""
    messages = get_basic_conversation_for_generation()
    decoded = nemotron_tokenizer.decode(
        nemotron_renderer.build_generation_prompt(messages, prefill="Sure, ").to_ints()
    )
    assert decoded.endswith("Sure, ")
    assert not decoded.endswith("<think>\n")


# =============================================================================
# HF Template Compatibility Tests — Generation
# =============================================================================


def test_basic_conversation_generation_matches_hf(nemotron_tokenizer, nemotron_renderer):
    """Basic conversation generation matches HF template."""
    messages = get_basic_conversation_for_generation()
    cookbook = nemotron_renderer.build_generation_prompt(messages).to_ints()
    hf = _hf_generation_tokens(
        nemotron_tokenizer,
        [nemotron_renderer.to_openai_message(m) for m in messages],
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_tokenizer.decode(cookbook)}\nHF: {nemotron_tokenizer.decode(hf)}"
    )


def test_disable_thinking_generation_matches_hf(
    nemotron_tokenizer, nemotron_renderer_disable_thinking
):
    """Disable-thinking generation matches HF with enable_thinking=False."""
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi!"),
        Message(role="user", content="What is 2+2?"),
    ]
    r = nemotron_renderer_disable_thinking
    cookbook = r.build_generation_prompt(messages).to_ints()
    hf = _hf_generation_tokens(
        nemotron_tokenizer,
        [r.to_openai_message(m) for m in messages],
        enable_thinking=False,
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_tokenizer.decode(cookbook)}\nHF: {nemotron_tokenizer.decode(hf)}"
    )


# =============================================================================
# HF Template Compatibility Tests — Supervised
# =============================================================================


def test_basic_conversation_supervised_matches_hf(nemotron_tokenizer, nemotron_renderer):
    """Basic supervised example matches HF template (no gen prompt)."""
    messages = get_basic_conversation_for_supervised()
    cookbook = nemotron_renderer.build_supervised_example(messages)[0].to_ints()
    hf = _hf_supervised_tokens(
        nemotron_tokenizer,
        [nemotron_renderer.to_openai_message(m) for m in messages],
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_tokenizer.decode(cookbook)}\nHF: {nemotron_tokenizer.decode(hf)}"
    )


def test_thinking_conversation_supervised_matches_hf(nemotron_tokenizer, nemotron_renderer):
    """Supervised example with thinking content matches HF template."""
    messages = get_thinking_conversation_for_supervised()
    cookbook = nemotron_renderer.build_supervised_example(messages)[0].to_ints()
    hf = _hf_supervised_tokens(
        nemotron_tokenizer,
        [nemotron_renderer.to_openai_message(m) for m in messages],
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_tokenizer.decode(cookbook)}\nHF: {nemotron_tokenizer.decode(hf)}"
    )


def test_multiturn_thinking_supervised_matches_hf(nemotron_tokenizer, nemotron_renderer):
    """Multi-turn with thinking in both assistant messages matches HF template.

    Nemotron-3's HF template truncates thinking in historical messages to
    <think></think>. This test verifies our renderer does the same.
    """
    messages = get_multiturn_thinking_conversation()
    cookbook = nemotron_renderer.build_supervised_example(messages)[0].to_ints()
    hf = _hf_supervised_tokens(
        nemotron_tokenizer,
        [nemotron_renderer.to_openai_message(m) for m in messages],
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_tokenizer.decode(cookbook)}\nHF: {nemotron_tokenizer.decode(hf)}"
    )


def test_think_block_added_to_all_assistant_history(nemotron_tokenizer, nemotron_renderer):
    """<think></think> is prepended to historical assistant messages without thinking."""
    messages = get_basic_conversation_for_generation()  # ends with user, has one assistant
    decoded = nemotron_tokenizer.decode(
        nemotron_renderer.build_generation_prompt(messages).to_ints()
    )
    # The historical assistant message should have <think></think> prepended
    assert "<think></think>I'm fine, thank you!" in decoded


# =============================================================================
# HF Template Compatibility Tests — Tool Declarations
# =============================================================================


@pytest.mark.parametrize("build_mode", ["generation", "supervised"])
def test_tool_declaration_matches_hf(build_mode: str, nemotron_tokenizer, nemotron_renderer):
    """Tool declarations match HF template output."""
    tools = [get_tool_spec()]
    openai_tools = [{"type": "function", "function": tool} for tool in tools]
    system_prompt = "You are a helpful assistant."

    prefix_messages = nemotron_renderer.create_conversation_prefix_with_tools(
        tools, system_prompt=system_prompt
    )
    user_msg = Message(role="user", content="What's the weather in NYC?")

    hf_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What's the weather in NYC?"},
    ]

    if build_mode == "generation":
        cookbook = nemotron_renderer.build_generation_prompt(prefix_messages + [user_msg]).to_ints()
        hf = _hf_generation_tokens(nemotron_tokenizer, hf_messages, tools=openai_tools)
    else:
        assistant_msg = Message(role="assistant", content="Let me check that for you.")
        cookbook = nemotron_renderer.build_supervised_example(
            prefix_messages + [user_msg, assistant_msg]
        )[0].to_ints()
        hf_messages.append({"role": "assistant", "content": "Let me check that for you."})
        hf = _hf_supervised_tokens(nemotron_tokenizer, hf_messages, tools=openai_tools)

    assert cookbook == hf, (
        f"Cookbook: {nemotron_tokenizer.decode(cookbook)}\nHF: {nemotron_tokenizer.decode(hf)}"
    )


def test_tool_call_conversation_generation_matches_hf(nemotron_tokenizer, nemotron_renderer):
    """Tool call + tool response conversation (generation) matches HF template."""
    messages, tools = get_tool_call_conversation_for_generation()
    openai_tools = [{"type": "function", "function": tool} for tool in tools]
    system_prompt = "You are a helpful assistant."

    prefix = nemotron_renderer.create_conversation_prefix_with_tools(
        tools, system_prompt=system_prompt
    )
    cookbook = nemotron_renderer.build_generation_prompt(prefix + messages).to_ints()

    hf_messages = [
        {"role": "system", "content": system_prompt},
        *[nemotron_renderer.to_openai_message(m) for m in messages],
    ]
    hf = _hf_generation_tokens(nemotron_tokenizer, hf_messages, tools=openai_tools)

    assert cookbook == hf, (
        f"Cookbook: {nemotron_tokenizer.decode(cookbook)}\nHF: {nemotron_tokenizer.decode(hf)}"
    )


def test_tool_call_conversation_supervised_matches_hf(nemotron_tokenizer, nemotron_renderer):
    """Complete tool call conversation (supervised) matches HF template."""
    messages, tools = get_tool_call_conversation_for_supervised()
    openai_tools = [{"type": "function", "function": tool} for tool in tools]
    system_prompt = "You are a helpful assistant."

    prefix = nemotron_renderer.create_conversation_prefix_with_tools(
        tools, system_prompt=system_prompt
    )
    cookbook = nemotron_renderer.build_supervised_example(prefix + messages)[0].to_ints()

    hf_messages = [
        {"role": "system", "content": system_prompt},
        *[nemotron_renderer.to_openai_message(m) for m in messages],
    ]
    hf = _hf_supervised_tokens(nemotron_tokenizer, hf_messages, tools=openai_tools)

    assert cookbook == hf, (
        f"Cookbook: {nemotron_tokenizer.decode(cookbook)}\nHF: {nemotron_tokenizer.decode(hf)}"
    )


def test_historical_tool_call_with_nonempty_text_generation_matches_hf(
    nemotron_tokenizer, nemotron_renderer
):
    """Historical tool_call message with thinking + non-empty text matches HF.

    In the HF Jinja template's tool_calls branch, ``| trim`` binds tighter than
    ``~``, so the leading ``\\n`` from the content is stripped before concatenation
    with ``<think></think>``, producing ``<think></think>text`` (no newline).
    This differs from the non-tool_calls branch which preserves the newline.
    """
    messages, tools = get_historical_tool_call_with_nonempty_text_conversation()
    openai_tools = [{"type": "function", "function": tool} for tool in tools]
    system_prompt = "You are a helpful assistant."

    prefix = nemotron_renderer.create_conversation_prefix_with_tools(
        tools, system_prompt=system_prompt
    )
    cookbook = nemotron_renderer.build_generation_prompt(prefix + messages).to_ints()

    hf_messages = [
        {"role": "system", "content": system_prompt},
        *[nemotron_renderer.to_openai_message(m) for m in messages],
    ]
    hf = _hf_generation_tokens(nemotron_tokenizer, hf_messages, tools=openai_tools)

    assert cookbook == hf, (
        f"Cookbook: {nemotron_tokenizer.decode(cookbook)}\nHF: {nemotron_tokenizer.decode(hf)}"
    )


def test_historical_tool_call_with_nonempty_text_supervised_matches_hf(
    nemotron_tokenizer, nemotron_renderer
):
    """Supervised version of the historical tool_call + non-empty text edge case."""
    messages, tools = get_historical_tool_call_with_nonempty_text_conversation()
    openai_tools = [{"type": "function", "function": tool} for tool in tools]
    system_prompt = "You are a helpful assistant."

    prefix = nemotron_renderer.create_conversation_prefix_with_tools(
        tools, system_prompt=system_prompt
    )
    cookbook = nemotron_renderer.build_supervised_example(prefix + messages)[0].to_ints()

    hf_messages = [
        {"role": "system", "content": system_prompt},
        *[nemotron_renderer.to_openai_message(m) for m in messages],
    ]
    hf = _hf_supervised_tokens(nemotron_tokenizer, hf_messages, tools=openai_tools)

    assert cookbook == hf, (
        f"Cookbook: {nemotron_tokenizer.decode(cookbook)}\nHF: {nemotron_tokenizer.decode(hf)}"
    )


# =============================================================================
# Parse Response Tests
# =============================================================================


def test_parse_response_plain_text(nemotron_tokenizer, nemotron_renderer):
    """parse_response decodes a plain text response (no thinking)."""
    tokens = nemotron_tokenizer.encode("The answer is 42.<|im_end|>", add_special_tokens=False)
    message, success = nemotron_renderer.parse_response(tokens)
    assert success.is_clean
    from tinker_cookbook.renderers import get_text_content

    assert "42" in get_text_content(message)


def test_parse_response_with_thinking(nemotron_tokenizer, nemotron_renderer):
    """parse_response extracts thinking content from the response."""
    # Simulates what the model generates after the <think>\n prefill
    response_text = "I should reason carefully.\n</think>\nThe answer is 42.<|im_end|>"
    tokens = nemotron_tokenizer.encode(response_text, add_special_tokens=False)
    message, success = nemotron_renderer.parse_response(tokens)

    assert success.is_clean
    content = message.get("content")
    assert isinstance(content, list)
    thinking_parts = [p for p in content if p["type"] == "thinking"]
    text_parts = [p for p in content if p["type"] == "text"]
    assert len(thinking_parts) == 1
    assert "reason" in thinking_parts[0]["thinking"]
    assert len(text_parts) == 1
    assert "42" in text_parts[0]["text"]


def test_parse_response_for_streaming_with_thinking(nemotron_tokenizer, nemotron_renderer):
    """_parse_response_for_streaming preserves the single \\n separator after </think>.

    The inherited _parse_response_for_streaming calls _postprocess_parsed_message
    which strips separator newlines. For Nemotron-3, the separator is one \\n (not
    two like Qwen3.5), so the text after thinking should NOT start with \\n (the
    single separator should be stripped) and must not lose content by over-stripping.
    """
    # Include <think>\n prefix — in real streaming, _normalize_response_tokens
    # prepends this before _parse_response_for_streaming is called.
    response_text = "<think>\nI should reason carefully.\n</think>\nThe answer is 42.<|im_end|>"
    tokens = nemotron_tokenizer.encode(response_text, add_special_tokens=False)
    message, success = nemotron_renderer._parse_response_for_streaming(tokens)

    assert success.is_clean
    content = message.get("content")
    assert isinstance(content, list)
    thinking_parts = [p for p in content if p["type"] == "thinking"]
    text_parts = [p for p in content if p["type"] == "text"]
    assert len(thinking_parts) == 1
    assert "reason" in thinking_parts[0]["thinking"]
    assert len(text_parts) == 1
    # The text should start with "The answer" — the \n separator should be stripped,
    # not left as-is (0 newlines stripped) or double-stripped.
    assert text_parts[0]["text"].startswith("The answer"), (
        f"Expected text to start with 'The answer' but got: {text_parts[0]['text']!r}"
    )


def test_parse_response_tool_call(nemotron_tokenizer, nemotron_renderer):
    """parse_response parses XML-format tool calls."""
    tool_call_text = (
        "\n</think>\n"
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=location>\n"
        "New York, NY\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call><|im_end|>"
    )
    tokens = nemotron_tokenizer.encode(tool_call_text, add_special_tokens=False)
    message, success = nemotron_renderer.parse_response(tokens)

    assert success.is_clean
    tool_calls = message.get("tool_calls", [])
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    args = json.loads(tool_calls[0].function.arguments)
    assert args["location"] == "New York, NY"


# =============================================================================
# Renderer Identity Tests
# =============================================================================


def test_renderer_types(nemotron_renderer, nemotron_renderer_disable_thinking):
    assert isinstance(nemotron_renderer, Nemotron3Renderer)
    assert isinstance(nemotron_renderer_disable_thinking, Nemotron3DisableThinkingRenderer)


def test_renderer_is_not_qwen35(nemotron_renderer):
    from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer

    assert type(nemotron_renderer) is not Qwen3_5Renderer


# =============================================================================
# Low Thinking Renderer Tests (Nemotron-3 Super only)
# =============================================================================

# The low_effort feature is only in the Super model's HF template, so we need
# the Super tokenizer for HF comparison tests.


@pytest.fixture(scope="module")
def nemotron_super_tokenizer():
    return get_tokenizer(NEMOTRON3_SUPER_MODEL)


@pytest.fixture(scope="module")
def nemotron_low_thinking_renderer(nemotron_super_tokenizer):
    return get_renderer("nemotron3_low_thinking", nemotron_super_tokenizer)


def test_low_thinking_renderer_type(nemotron_low_thinking_renderer):
    assert isinstance(nemotron_low_thinking_renderer, Nemotron3LowThinkingRenderer)
    assert isinstance(nemotron_low_thinking_renderer, Nemotron3Renderer)


def test_low_thinking_generation_matches_hf(
    nemotron_super_tokenizer, nemotron_low_thinking_renderer
):
    """Low-thinking generation matches HF template with low_effort=True."""
    messages = get_basic_conversation_for_generation()
    r = nemotron_low_thinking_renderer
    cookbook = r.build_generation_prompt(messages).to_ints()
    hf = _hf_generation_tokens(
        nemotron_super_tokenizer,
        [r.to_openai_message(m) for m in messages],
        low_effort=True,
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_super_tokenizer.decode(cookbook)}\n"
        f"HF: {nemotron_super_tokenizer.decode(hf)}"
    )


def test_low_thinking_supervised_matches_hf(
    nemotron_super_tokenizer, nemotron_low_thinking_renderer
):
    """Low-thinking supervised example matches HF template with low_effort=True."""
    messages = get_basic_conversation_for_supervised()
    r = nemotron_low_thinking_renderer
    cookbook = r.build_supervised_example(messages)[0].to_ints()
    hf = _hf_supervised_tokens(
        nemotron_super_tokenizer,
        [r.to_openai_message(m) for m in messages],
        low_effort=True,
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_super_tokenizer.decode(cookbook)}\n"
        f"HF: {nemotron_super_tokenizer.decode(hf)}"
    )


def test_low_thinking_multiturn_generation_matches_hf(
    nemotron_super_tokenizer, nemotron_low_thinking_renderer
):
    """Multi-turn low-thinking generation matches HF template."""
    messages = get_multiturn_thinking_conversation()[:4]
    r = nemotron_low_thinking_renderer
    cookbook = r.build_generation_prompt(messages).to_ints()
    hf = _hf_generation_tokens(
        nemotron_super_tokenizer,
        [r.to_openai_message(m) for m in messages],
        low_effort=True,
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_super_tokenizer.decode(cookbook)}\n"
        f"HF: {nemotron_super_tokenizer.decode(hf)}"
    )


def test_low_thinking_multiturn_supervised_matches_hf(
    nemotron_super_tokenizer, nemotron_low_thinking_renderer
):
    """Multi-turn low-thinking supervised example matches HF template."""
    messages = get_multiturn_thinking_conversation()
    r = nemotron_low_thinking_renderer
    cookbook = r.build_supervised_example(messages)[0].to_ints()
    hf = _hf_supervised_tokens(
        nemotron_super_tokenizer,
        [r.to_openai_message(m) for m in messages],
        low_effort=True,
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_super_tokenizer.decode(cookbook)}\n"
        f"HF: {nemotron_super_tokenizer.decode(hf)}"
    )


# =============================================================================
# Ultra Renderer Tests
# =============================================================================


@pytest.fixture(scope="module")
def nemotron_ultra_tokenizer():
    return get_tokenizer(NEMOTRON3_ULTRA_MODEL)


@pytest.fixture(scope="module")
def nemotron_ultra_renderer(nemotron_ultra_tokenizer):
    return get_renderer("nemotron3_ultra", nemotron_ultra_tokenizer)


@pytest.fixture(scope="module")
def nemotron_ultra_disable_thinking_renderer(nemotron_ultra_tokenizer):
    return get_renderer("nemotron3_ultra_disable_thinking", nemotron_ultra_tokenizer)


@pytest.fixture(scope="module")
def nemotron_ultra_medium_thinking_renderer(nemotron_ultra_tokenizer):
    return get_renderer("nemotron3_ultra_medium_thinking", nemotron_ultra_tokenizer)


def test_ultra_renderer_types(
    nemotron_ultra_renderer,
    nemotron_ultra_disable_thinking_renderer,
    nemotron_ultra_medium_thinking_renderer,
):
    assert isinstance(nemotron_ultra_renderer, Nemotron3UltraRenderer)
    assert isinstance(
        nemotron_ultra_disable_thinking_renderer, Nemotron3UltraDisableThinkingRenderer
    )
    assert isinstance(nemotron_ultra_medium_thinking_renderer, Nemotron3UltraMediumThinkingRenderer)


def test_ultra_generation_matches_hf(nemotron_ultra_tokenizer, nemotron_ultra_renderer):
    """Ultra full-thinking generation prompt matches HF template."""
    messages = get_basic_conversation_for_generation()
    cookbook = nemotron_ultra_renderer.build_generation_prompt(messages).to_ints()
    hf = _hf_generation_tokens(
        nemotron_ultra_tokenizer,
        [nemotron_ultra_renderer.to_openai_message(m) for m in messages],
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_ultra_tokenizer.decode(cookbook)}\n"
        f"HF: {nemotron_ultra_tokenizer.decode(hf)}"
    )


def test_ultra_thinking_supervised_matches_hf(nemotron_ultra_tokenizer, nemotron_ultra_renderer):
    """Ultra uses no separator newlines around explicit thinking blocks."""
    messages = get_thinking_conversation_for_supervised()
    cookbook = nemotron_ultra_renderer.build_supervised_example(messages)[0].to_ints()
    hf = _hf_supervised_tokens(
        nemotron_ultra_tokenizer,
        [nemotron_ultra_renderer.to_openai_message(m) for m in messages],
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_ultra_tokenizer.decode(cookbook)}\n"
        f"HF: {nemotron_ultra_tokenizer.decode(hf)}"
    )


def test_ultra_multiturn_thinking_supervised_matches_hf(
    nemotron_ultra_tokenizer, nemotron_ultra_renderer
):
    """Ultra historical thinking truncation has no post-</think> separator."""
    messages = get_multiturn_thinking_conversation()
    cookbook = nemotron_ultra_renderer.build_supervised_example(messages)[0].to_ints()
    hf = _hf_supervised_tokens(
        nemotron_ultra_tokenizer,
        [nemotron_ultra_renderer.to_openai_message(m) for m in messages],
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_ultra_tokenizer.decode(cookbook)}\n"
        f"HF: {nemotron_ultra_tokenizer.decode(hf)}"
    )


def test_ultra_tool_call_conversation_supervised_matches_hf(
    nemotron_ultra_tokenizer, nemotron_ultra_renderer
):
    """Ultra tool calls match HF after thinking-block formatting differences."""
    messages, tools = get_tool_call_conversation_for_supervised()
    openai_tools = [{"type": "function", "function": tool} for tool in tools]
    system_prompt = "You are a helpful assistant."

    prefix = nemotron_ultra_renderer.create_conversation_prefix_with_tools(
        tools, system_prompt=system_prompt
    )
    cookbook = nemotron_ultra_renderer.build_supervised_example(prefix + messages)[0].to_ints()

    hf_messages = [
        {"role": "system", "content": system_prompt},
        *[nemotron_ultra_renderer.to_openai_message(m) for m in messages],
    ]
    hf = _hf_supervised_tokens(nemotron_ultra_tokenizer, hf_messages, tools=openai_tools)

    assert cookbook == hf, (
        f"Cookbook: {nemotron_ultra_tokenizer.decode(cookbook)}\n"
        f"HF: {nemotron_ultra_tokenizer.decode(hf)}"
    )


def test_ultra_disable_thinking_generation_matches_hf(
    nemotron_ultra_tokenizer, nemotron_ultra_disable_thinking_renderer
):
    """Ultra disable-thinking generation matches HF with enable_thinking=False."""
    messages = get_basic_conversation_for_generation()
    r = nemotron_ultra_disable_thinking_renderer
    cookbook = r.build_generation_prompt(messages).to_ints()
    hf = _hf_generation_tokens(
        nemotron_ultra_tokenizer,
        [r.to_openai_message(m) for m in messages],
        enable_thinking=False,
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_ultra_tokenizer.decode(cookbook)}\n"
        f"HF: {nemotron_ultra_tokenizer.decode(hf)}"
    )


def test_ultra_medium_thinking_generation_matches_hf(
    nemotron_ultra_tokenizer, nemotron_ultra_medium_thinking_renderer
):
    """Ultra medium-thinking generation matches HF template with medium_effort=True."""
    messages = get_basic_conversation_for_generation()
    r = nemotron_ultra_medium_thinking_renderer
    cookbook = r.build_generation_prompt(messages).to_ints()
    hf = _hf_generation_tokens(
        nemotron_ultra_tokenizer,
        [r.to_openai_message(m) for m in messages],
        medium_effort=True,
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_ultra_tokenizer.decode(cookbook)}\n"
        f"HF: {nemotron_ultra_tokenizer.decode(hf)}"
    )


def test_ultra_medium_thinking_supervised_matches_hf(
    nemotron_ultra_tokenizer, nemotron_ultra_medium_thinking_renderer
):
    """Ultra medium-thinking supervised example matches HF template."""
    messages = get_basic_conversation_for_supervised()
    r = nemotron_ultra_medium_thinking_renderer
    cookbook = r.build_supervised_example(messages)[0].to_ints()
    hf = _hf_supervised_tokens(
        nemotron_ultra_tokenizer,
        [r.to_openai_message(m) for m in messages],
        medium_effort=True,
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_ultra_tokenizer.decode(cookbook)}\n"
        f"HF: {nemotron_ultra_tokenizer.decode(hf)}"
    )


def test_ultra_medium_thinking_multiturn_generation_matches_hf(
    nemotron_ultra_tokenizer, nemotron_ultra_medium_thinking_renderer
):
    """Multi-turn Ultra medium-thinking generation matches HF template."""
    messages = get_multiturn_thinking_conversation()[:4]
    r = nemotron_ultra_medium_thinking_renderer
    cookbook = r.build_generation_prompt(messages).to_ints()
    hf = _hf_generation_tokens(
        nemotron_ultra_tokenizer,
        [r.to_openai_message(m) for m in messages],
        medium_effort=True,
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_ultra_tokenizer.decode(cookbook)}\n"
        f"HF: {nemotron_ultra_tokenizer.decode(hf)}"
    )


def test_ultra_medium_thinking_multiturn_supervised_matches_hf(
    nemotron_ultra_tokenizer, nemotron_ultra_medium_thinking_renderer
):
    """Multi-turn Ultra medium-thinking supervised example matches HF template."""
    messages = get_multiturn_thinking_conversation()
    r = nemotron_ultra_medium_thinking_renderer
    cookbook = r.build_supervised_example(messages)[0].to_ints()
    hf = _hf_supervised_tokens(
        nemotron_ultra_tokenizer,
        [r.to_openai_message(m) for m in messages],
        medium_effort=True,
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_ultra_tokenizer.decode(cookbook)}\n"
        f"HF: {nemotron_ultra_tokenizer.decode(hf)}"
    )


# =============================================================================
# Preserve-Thinking Renderer Tests (HF truncate_history_thinking=False)
# =============================================================================

# The default Nemotron-3 renderers strip <think> blocks from historical assistant
# turns, matching the HF template default truncate_history_thinking=True. The
# preserve-thinking variants keep them, matching truncate_history_thinking=False.


@pytest.fixture(scope="module")
def nemotron_preserve_thinking_renderer(nemotron_tokenizer):
    return get_renderer("nemotron3_preserve_thinking", nemotron_tokenizer)


@pytest.fixture(scope="module")
def nemotron_ultra_preserve_thinking_renderer(nemotron_ultra_tokenizer):
    return get_renderer("nemotron3_ultra_preserve_thinking", nemotron_ultra_tokenizer)


def test_preserve_thinking_renderer_types(
    nemotron_preserve_thinking_renderer, nemotron_ultra_preserve_thinking_renderer
):
    assert isinstance(nemotron_preserve_thinking_renderer, Nemotron3PreserveThinkingRenderer)
    assert isinstance(nemotron_preserve_thinking_renderer, Nemotron3Renderer)
    assert isinstance(
        nemotron_ultra_preserve_thinking_renderer, Nemotron3UltraPreserveThinkingRenderer
    )
    assert isinstance(nemotron_ultra_preserve_thinking_renderer, Nemotron3UltraRenderer)


def test_preserve_thinking_has_extension_property(
    nemotron_preserve_thinking_renderer,
    nemotron_ultra_preserve_thinking_renderer,
    nemotron_renderer,
):
    """Preserving history thinking gives the extension property; the default does not."""
    assert nemotron_preserve_thinking_renderer.has_extension_property is True
    assert nemotron_ultra_preserve_thinking_renderer.has_extension_property is True
    assert nemotron_renderer.has_extension_property is False


def test_default_strips_history_thinking(nemotron_tokenizer, nemotron_renderer):
    """Sanity contrast: the default renderer collapses historical thinking."""
    messages = get_multiturn_thinking_conversation()
    decoded = nemotron_tokenizer.decode(
        nemotron_renderer.build_supervised_example(messages)[0].to_ints()
    )
    assert "First turn reasoning." not in decoded  # historical thinking dropped
    assert "<think></think>" in decoded
    assert "Second turn reasoning." in decoded  # last turn's thinking always kept


def test_preserve_thinking_keeps_history_thinking(
    nemotron_tokenizer, nemotron_preserve_thinking_renderer
):
    """Preserve variant keeps the historical <think>...</think> block verbatim."""
    messages = get_multiturn_thinking_conversation()
    decoded = nemotron_tokenizer.decode(
        nemotron_preserve_thinking_renderer.build_supervised_example(messages)[0].to_ints()
    )
    assert "<think>\nFirst turn reasoning.\n</think>" in decoded
    assert "Second turn reasoning." in decoded


def test_preserve_thinking_multiturn_supervised_matches_hf(
    nemotron_tokenizer, nemotron_preserve_thinking_renderer
):
    """Preserve variant supervised example matches HF truncate_history_thinking=False."""
    messages = get_multiturn_thinking_conversation()
    r = nemotron_preserve_thinking_renderer
    cookbook = r.build_supervised_example(messages)[0].to_ints()
    hf = _hf_supervised_tokens(
        nemotron_tokenizer,
        [r.to_openai_message(m) for m in messages],
        truncate_history_thinking=False,
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_tokenizer.decode(cookbook)}\nHF: {nemotron_tokenizer.decode(hf)}"
    )


def test_preserve_thinking_multiturn_generation_matches_hf(
    nemotron_tokenizer, nemotron_preserve_thinking_renderer
):
    """Preserve variant generation prompt matches HF truncate_history_thinking=False."""
    messages = get_multiturn_thinking_conversation()[:4]
    r = nemotron_preserve_thinking_renderer
    cookbook = r.build_generation_prompt(messages).to_ints()
    hf = _hf_generation_tokens(
        nemotron_tokenizer,
        [r.to_openai_message(m) for m in messages],
        truncate_history_thinking=False,
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_tokenizer.decode(cookbook)}\nHF: {nemotron_tokenizer.decode(hf)}"
    )


def test_preserve_thinking_historical_tool_call_supervised_matches_hf(
    nemotron_tokenizer, nemotron_preserve_thinking_renderer
):
    """Historical assistant turn with thinking + text + tool_calls, preserved, matches HF."""
    messages, tools = get_historical_tool_call_with_nonempty_text_conversation()
    openai_tools = [{"type": "function", "function": tool} for tool in tools]
    system_prompt = "You are a helpful assistant."
    r = nemotron_preserve_thinking_renderer

    prefix = r.create_conversation_prefix_with_tools(tools, system_prompt=system_prompt)
    cookbook = r.build_supervised_example(prefix + messages)[0].to_ints()

    hf_messages = [
        {"role": "system", "content": system_prompt},
        *[r.to_openai_message(m) for m in messages],
    ]
    hf = _hf_supervised_tokens(
        nemotron_tokenizer, hf_messages, tools=openai_tools, truncate_history_thinking=False
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_tokenizer.decode(cookbook)}\nHF: {nemotron_tokenizer.decode(hf)}"
    )


def test_ultra_preserve_thinking_multiturn_supervised_matches_hf(
    nemotron_ultra_tokenizer, nemotron_ultra_preserve_thinking_renderer
):
    """Ultra preserve variant supervised example matches HF truncate_history_thinking=False."""
    messages = get_multiturn_thinking_conversation()
    r = nemotron_ultra_preserve_thinking_renderer
    cookbook = r.build_supervised_example(messages)[0].to_ints()
    hf = _hf_supervised_tokens(
        nemotron_ultra_tokenizer,
        [r.to_openai_message(m) for m in messages],
        truncate_history_thinking=False,
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_ultra_tokenizer.decode(cookbook)}\n"
        f"HF: {nemotron_ultra_tokenizer.decode(hf)}"
    )


def test_ultra_preserve_thinking_multiturn_generation_matches_hf(
    nemotron_ultra_tokenizer, nemotron_ultra_preserve_thinking_renderer
):
    """Ultra preserve variant generation prompt matches HF truncate_history_thinking=False."""
    messages = get_multiturn_thinking_conversation()[:4]
    r = nemotron_ultra_preserve_thinking_renderer
    cookbook = r.build_generation_prompt(messages).to_ints()
    hf = _hf_generation_tokens(
        nemotron_ultra_tokenizer,
        [r.to_openai_message(m) for m in messages],
        truncate_history_thinking=False,
    )
    assert cookbook == hf, (
        f"Cookbook: {nemotron_ultra_tokenizer.decode(cookbook)}\n"
        f"HF: {nemotron_ultra_tokenizer.decode(hf)}"
    )


def test_preserve_thinking_registered_in_model_info():
    from tinker_cookbook import model_info

    for model in [
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    ]:
        assert "nemotron3_preserve_thinking" in model_info.get_recommended_renderer_names(model)
    assert "nemotron3_ultra_preserve_thinking" in model_info.get_recommended_renderer_names(
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
    )


# The preserve-thinking behavior isn't limited to the two named variants above:
# any Nemotron-3 mode renderer keeps history thinking when constructed with
# strip_thinking_from_history=False. These check that opt-in matches HF for the
# disable / low-effort / medium-effort modes, which have no named preserve variant.


def _assert_mode_preserve_matches_hf(tokenizer, renderer, **hf_kwargs):
    """Renderer (built with strip_thinking_from_history=False) matches HF with
    truncate_history_thinking=False plus the mode flags in ``hf_kwargs``."""
    sup_msgs = get_multiturn_thinking_conversation()
    cookbook_sup = renderer.build_supervised_example(sup_msgs)[0].to_ints()
    hf_sup = _hf_supervised_tokens(
        tokenizer,
        [renderer.to_openai_message(m) for m in sup_msgs],
        truncate_history_thinking=False,
        **hf_kwargs,
    )
    assert cookbook_sup == hf_sup, (
        f"supervised: {tokenizer.decode(cookbook_sup)}\nHF: {tokenizer.decode(hf_sup)}"
    )

    gen_msgs = get_multiturn_thinking_conversation()[:4]
    cookbook_gen = renderer.build_generation_prompt(gen_msgs).to_ints()
    hf_gen = _hf_generation_tokens(
        tokenizer,
        [renderer.to_openai_message(m) for m in gen_msgs],
        truncate_history_thinking=False,
        **hf_kwargs,
    )
    assert cookbook_gen == hf_gen, (
        f"generation: {tokenizer.decode(cookbook_gen)}\nHF: {tokenizer.decode(hf_gen)}"
    )


def test_disable_thinking_preserve_matches_hf(nemotron_tokenizer):
    """Reasoning-off + strip_thinking_from_history=False keeps history, matches HF."""
    renderer = Nemotron3DisableThinkingRenderer(
        nemotron_tokenizer, strip_thinking_from_history=False
    )
    _assert_mode_preserve_matches_hf(nemotron_tokenizer, renderer, enable_thinking=False)


def test_low_thinking_preserve_matches_hf(nemotron_super_tokenizer):
    """Low-effort + strip_thinking_from_history=False keeps history, matches HF (Super)."""
    renderer = Nemotron3LowThinkingRenderer(
        nemotron_super_tokenizer, strip_thinking_from_history=False
    )
    _assert_mode_preserve_matches_hf(nemotron_super_tokenizer, renderer, low_effort=True)


def test_ultra_medium_thinking_preserve_matches_hf(nemotron_ultra_tokenizer):
    """Ultra medium-effort + strip_thinking_from_history=False keeps history, matches HF."""
    renderer = Nemotron3UltraMediumThinkingRenderer(
        nemotron_ultra_tokenizer, strip_thinking_from_history=False
    )
    _assert_mode_preserve_matches_hf(nemotron_ultra_tokenizer, renderer, medium_effort=True)


def test_ultra_disable_thinking_preserve_matches_hf(nemotron_ultra_tokenizer):
    """Ultra reasoning-off + strip_thinking_from_history=False keeps history, matches HF."""
    renderer = Nemotron3UltraDisableThinkingRenderer(
        nemotron_ultra_tokenizer, strip_thinking_from_history=False
    )
    _assert_mode_preserve_matches_hf(nemotron_ultra_tokenizer, renderer, enable_thinking=False)
