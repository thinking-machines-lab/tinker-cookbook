"""
Tests for Kimi K2.5 renderer.

Tests verify that the KimiK25Renderer produces correct output:
1. Generation prompt includes `<think>` prefill (thinking enabled)
2. Disable-thinking variant uses `<think></think>` prefill
3. TypeScript-style tool declarations
4. HF template compatibility for both build_generation_prompt and build_supervised_example
"""

from typing import cast

import pytest

from tinker_cookbook.renderers import Message, ToolCall, ToolSpec, get_renderer
from tinker_cookbook.renderers.kimi_k2_5_tool_declaration_ts import encode_tools_to_typescript_style
from tinker_cookbook.tokenizer_utils import get_tokenizer

KIMI_K25_MODEL = "moonshotai/Kimi-K2.5"


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def kimi_tokenizer():
    """Get the Kimi K2.5 tokenizer (cached per module)."""
    try:
        return get_tokenizer(KIMI_K25_MODEL)
    except ModuleNotFoundError as e:
        if "Kimi-K2" in str(e):
            pytest.skip(f"K2.5 tokenizer has HF module import bug: {e}")
        raise


@pytest.fixture(scope="module")
def kimi_renderer(kimi_tokenizer):
    """Get the Kimi K2.5 renderer (cached per module)."""
    return get_renderer("kimi_k25", kimi_tokenizer)


@pytest.fixture(scope="module")
def kimi_renderer_disable_thinking(kimi_tokenizer):
    """Get the Kimi K2.5 disable-thinking renderer (cached per module)."""
    return get_renderer("kimi_k25_disable_thinking", kimi_tokenizer)


@pytest.fixture(scope="module")
def hf_generation_prompt_length(kimi_tokenizer):
    """Calculate the number of tokens in the HF generation prompt (cached per module).

    Uses a dummy conversation to find the difference between with/without generation prompt.
    This is constant regardless of conversation content.
    """
    dummy_msgs = [{"role": "user", "content": "hi"}]
    tokens_with = kimi_tokenizer.apply_chat_template(
        dummy_msgs, add_generation_prompt=True, tokenize=True, thinking=True
    )
    tokens_without = kimi_tokenizer.apply_chat_template(
        dummy_msgs, add_generation_prompt=False, tokenize=True, thinking=True
    )
    if hasattr(tokens_with, "input_ids"):
        tokens_with = tokens_with["input_ids"]
    if hasattr(tokens_without, "input_ids"):
        tokens_without = tokens_without["input_ids"]
    return len(tokens_with) - len(tokens_without)


def get_hf_tokens(tokenizer, hf_messages, gen_prompt_length: int, tools=None, for_generation: bool = True) -> list[int]:
    """Get HF tokens for generation or supervised mode.

    For supervised mode, slices off the generation prompt tokens.
    """
    hf_output = tokenizer.apply_chat_template(
        hf_messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=True,
        thinking=True,
    )
    if hasattr(hf_output, "input_ids"):
        tokens = cast(list[int], hf_output["input_ids"])
    else:
        tokens = cast(list[int], hf_output)

    if for_generation:
        return tokens
    return tokens[:-gen_prompt_length] if gen_prompt_length else tokens


# =============================================================================
# Helpers
# =============================================================================


def get_tool_spec() -> ToolSpec:
    """Sample tool specification for testing."""
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


# =============================================================================
# Test Conversations
# =============================================================================


def get_basic_conversation_for_generation() -> list[Message]:
    """3-turn conversation ending with user message (for generation)."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
    ]


def get_basic_conversation_for_supervised() -> list[Message]:
    """2-turn conversation ending with assistant (for supervised)."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
    ]


def get_tool_call_conversation_for_generation() -> tuple[list[Message], list[ToolSpec]]:
    """Conversation with tool call, ending ready for generation."""
    tools = [get_tool_spec()]
    tool_call = ToolCall(
        id="functions.get_weather:0",
        function=ToolCall.FunctionBody(
            name="get_weather",
            arguments='{"location": "New York, NY"}',
        ),
    )
    messages: list[Message] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in NYC?"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I need to check the weather in New York City."},
                {"type": "text", "text": ""},
            ],
            "tool_calls": [tool_call],
        },
        {
            "role": "tool",
            "name": "get_weather",
            "tool_call_id": "functions.get_weather:0",
            "content": '{"temperature": 72, "condition": "sunny"}',
        },
    ]
    return messages, tools


def get_tool_call_conversation_for_supervised() -> tuple[list[Message], list[ToolSpec]]:
    """Complete tool call conversation with final assistant response (for supervised)."""
    tools = [get_tool_spec()]
    tool_call = ToolCall(
        id="functions.get_weather:0",
        function=ToolCall.FunctionBody(
            name="get_weather",
            arguments='{"location": "New York, NY"}',
        ),
    )
    messages: list[Message] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in NYC?"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I need to check the weather in New York City."},
                {"type": "text", "text": ""},
            ],
            "tool_calls": [tool_call],
        },
        {
            "role": "tool",
            "name": "get_weather",
            "tool_call_id": "functions.get_weather:0",
            "content": '{"temperature": 72, "condition": "sunny"}',
        },
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "The weather data shows 72F and sunny."},
                {"type": "text", "text": "The weather in NYC is 72°F and sunny."},
            ],
        },
    ]
    return messages, tools


def get_multi_tool_call_conversation_for_generation() -> tuple[list[Message], list[ToolSpec]]:
    """Conversation with multiple tool calls in one message."""
    tools = [get_tool_spec()]
    tool_calls = [
        ToolCall(
            id="functions.get_weather:0",
            function=ToolCall.FunctionBody(
                name="get_weather",
                arguments='{"location": "New York, NY"}',
            ),
        ),
        ToolCall(
            id="functions.get_weather:1",
            function=ToolCall.FunctionBody(
                name="get_weather",
                arguments='{"location": "Los Angeles, CA"}',
            ),
        ),
    ]
    messages: list[Message] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in NYC and LA?"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I'll check the weather in both cities."},
                {"type": "text", "text": ""},
            ],
            "tool_calls": tool_calls,
        },
        {
            "role": "tool",
            "name": "get_weather",
            "tool_call_id": "functions.get_weather:0",
            "content": '{"temperature": 72, "condition": "sunny"}',
        },
        {
            "role": "tool",
            "name": "get_weather",
            "tool_call_id": "functions.get_weather:1",
            "content": '{"temperature": 85, "condition": "clear"}',
        },
    ]
    return messages, tools


def get_multi_step_tool_conversation_for_generation() -> tuple[list[Message], list[ToolSpec]]:
    """Multi-step tool calling: multiple rounds of tool calls."""
    tools = [get_tool_spec()]
    messages: list[Message] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Compare the weather in NYC and LA."},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me check NYC weather first."},
                {"type": "text", "text": ""},
            ],
            "tool_calls": [
                ToolCall(
                    id="functions.get_weather:0",
                    function=ToolCall.FunctionBody(
                        name="get_weather",
                        arguments='{"location": "New York, NY"}',
                    ),
                ),
            ],
        },
        {
            "role": "tool",
            "name": "get_weather",
            "tool_call_id": "functions.get_weather:0",
            "content": '{"temperature": 72, "condition": "sunny"}',
        },
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Now let me check LA weather."},
                {"type": "text", "text": ""},
            ],
            "tool_calls": [
                ToolCall(
                    id="functions.get_weather:1",
                    function=ToolCall.FunctionBody(
                        name="get_weather",
                        arguments='{"location": "Los Angeles, CA"}',
                    ),
                ),
            ],
        },
        {
            "role": "tool",
            "name": "get_weather",
            "tool_call_id": "functions.get_weather:1",
            "content": '{"temperature": 85, "condition": "clear"}',
        },
    ]
    return messages, tools


# =============================================================================
# TypeScript Tool Declaration Tests
# =============================================================================


def test_typescript_tool_declaration_basic():
    """Test basic TypeScript tool declaration generation."""
    tools = [{"type": "function", "function": get_tool_spec()}]
    ts_str = encode_tools_to_typescript_style(tools)

    assert "# Tools" in ts_str
    assert "## functions" in ts_str
    assert "namespace functions {" in ts_str
    assert "get_weather" in ts_str
    assert "type get_weather = (_:" in ts_str
    assert "location" in ts_str
    assert "string" in ts_str


def test_typescript_tool_declaration_with_enum():
    """Test TypeScript declaration includes enum values."""
    tools = [{"type": "function", "function": get_tool_spec()}]
    ts_str = encode_tools_to_typescript_style(tools)

    assert '"celsius"' in ts_str or "'celsius'" in ts_str
    assert '"fahrenheit"' in ts_str or "'fahrenheit'" in ts_str


def test_typescript_tool_declaration_description():
    """Test TypeScript declaration includes descriptions as comments."""
    tools = [{"type": "function", "function": get_tool_spec()}]
    ts_str = encode_tools_to_typescript_style(tools)

    assert "// Get the current weather" in ts_str


def test_typescript_tool_declaration_empty():
    """Test TypeScript declaration with empty tools list."""
    ts_str = encode_tools_to_typescript_style([])
    assert ts_str == ""


def test_typescript_tool_declaration_multiple_tools():
    """Test TypeScript declaration with multiple tools."""
    tools = [
        {
            "type": "function",
            "function": ToolSpec(
                name="get_weather",
                description="Get the current weather for a location",
                parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
            ),
        },
        {
            "type": "function",
            "function": ToolSpec(
                name="search_web",
                description="Search the web for information",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            ),
        },
    ]
    ts_str = encode_tools_to_typescript_style(tools)

    assert "type get_weather = (_:" in ts_str
    assert "type search_web = (_:" in ts_str
    assert "// Get the current weather" in ts_str
    assert "// Search the web" in ts_str


# =============================================================================
# Generation Prompt Prefill Tests (specific to generation)
# =============================================================================


def test_kimi_k25_generation_prompt_has_think_prefill(kimi_tokenizer, kimi_renderer):
    """Test that KimiK25Renderer adds <think> prefill for generation."""
    messages = get_basic_conversation_for_generation()
    gen_prompt = kimi_renderer.build_generation_prompt(messages)
    decoded = kimi_tokenizer.decode(gen_prompt.to_ints())

    assert decoded.endswith("<|im_assistant|>assistant<|im_middle|><think>")


def test_kimi_k25_disable_thinking_generation_prompt(kimi_tokenizer, kimi_renderer_disable_thinking):
    """Test that KimiK25DisableThinkingRenderer adds <think></think> prefill."""
    messages = get_basic_conversation_for_generation()
    gen_prompt = kimi_renderer_disable_thinking.build_generation_prompt(messages)
    decoded = kimi_tokenizer.decode(gen_prompt.to_ints())

    assert decoded.endswith("<|im_assistant|>assistant<|im_middle|><think></think>")


def test_kimi_k25_custom_prefill_overrides_default(kimi_tokenizer, kimi_renderer):
    """Test that custom prefill overrides the default <think> prefill."""
    messages = get_basic_conversation_for_generation()
    custom_prefill = "Custom response: "
    gen_prompt = kimi_renderer.build_generation_prompt(messages, prefill=custom_prefill)
    decoded = kimi_tokenizer.decode(gen_prompt.to_ints())

    assert decoded.endswith(custom_prefill)
    assert not decoded.endswith("<think>")


# =============================================================================
# HF Template Compatibility Tests - Parametrized for generation and supervised
# =============================================================================


def test_kimi_k25_basic_conversation_matches_hf(
    kimi_tokenizer, kimi_renderer, hf_generation_prompt_length
):
    """Test basic conversation generation matches HF template."""
    messages = get_basic_conversation_for_generation()
    cookbook_tokens = kimi_renderer.build_generation_prompt(messages).to_ints()

    hf_messages = [kimi_renderer.to_openai_message(m) for m in messages]
    hf_tokens = get_hf_tokens(
        kimi_tokenizer, hf_messages, hf_generation_prompt_length, for_generation=True
    )

    assert cookbook_tokens == hf_tokens, (
        f"Cookbook string: {kimi_tokenizer.decode(cookbook_tokens)}\n"
        f"HF string: {kimi_tokenizer.decode(hf_tokens)}"
    )


def test_kimi_k25_tool_call_conversation_matches_hf(
    kimi_tokenizer, kimi_renderer, hf_generation_prompt_length
):
    """Test tool call conversation generation matches HF template."""
    messages, tools = get_tool_call_conversation_for_generation()
    openai_tools = [{"type": "function", "function": tool} for tool in tools]

    prefix_messages = kimi_renderer.create_conversation_prefix_with_tools(
        tools, system_prompt="You are a helpful assistant."
    )
    prefix_messages = [m for m in prefix_messages if m["role"] == "tool_declare"]
    full_messages = prefix_messages + messages

    cookbook_tokens = kimi_renderer.build_generation_prompt(full_messages).to_ints()

    hf_messages = [kimi_renderer.to_openai_message(m) for m in messages]
    hf_tokens = get_hf_tokens(
        kimi_tokenizer, hf_messages, hf_generation_prompt_length,
        tools=openai_tools, for_generation=True
    )

    assert cookbook_tokens == hf_tokens, (
        f"Cookbook string: {kimi_tokenizer.decode(cookbook_tokens)}\n"
        f"HF string: {kimi_tokenizer.decode(hf_tokens)}"
    )


def test_kimi_k25_multi_tool_calls_matches_hf(
    kimi_tokenizer, kimi_renderer, hf_generation_prompt_length
):
    """Test multiple tool calls in one message matches HF template."""
    messages, tools = get_multi_tool_call_conversation_for_generation()
    openai_tools = [{"type": "function", "function": tool} for tool in tools]

    prefix_messages = kimi_renderer.create_conversation_prefix_with_tools(
        tools, system_prompt="You are a helpful assistant."
    )
    prefix_messages = [m for m in prefix_messages if m["role"] == "tool_declare"]
    full_messages = prefix_messages + messages

    cookbook_tokens = kimi_renderer.build_generation_prompt(full_messages).to_ints()

    hf_messages = [kimi_renderer.to_openai_message(m) for m in messages]
    hf_tokens = get_hf_tokens(
        kimi_tokenizer, hf_messages, hf_generation_prompt_length,
        tools=openai_tools, for_generation=True
    )

    assert cookbook_tokens == hf_tokens, (
        f"Cookbook string: {kimi_tokenizer.decode(cookbook_tokens)}\n"
        f"HF string: {kimi_tokenizer.decode(hf_tokens)}"
    )


def test_kimi_k25_multi_step_tool_calls_matches_hf(
    kimi_tokenizer, kimi_renderer, hf_generation_prompt_length
):
    """Test multi-step tool calling matches HF template."""
    messages, tools = get_multi_step_tool_conversation_for_generation()
    openai_tools = [{"type": "function", "function": tool} for tool in tools]

    prefix_messages = kimi_renderer.create_conversation_prefix_with_tools(
        tools, system_prompt="You are a helpful assistant."
    )
    prefix_messages = [m for m in prefix_messages if m["role"] == "tool_declare"]
    full_messages = prefix_messages + messages

    cookbook_tokens = kimi_renderer.build_generation_prompt(full_messages).to_ints()

    hf_messages = [kimi_renderer.to_openai_message(m) for m in messages]
    hf_tokens = get_hf_tokens(
        kimi_tokenizer, hf_messages, hf_generation_prompt_length,
        tools=openai_tools, for_generation=True
    )

    assert cookbook_tokens == hf_tokens, (
        f"Cookbook string: {kimi_tokenizer.decode(cookbook_tokens)}\n"
        f"HF string: {kimi_tokenizer.decode(hf_tokens)}"
    )


# =============================================================================
# Tool Declaration Format Tests
# =============================================================================


def test_kimi_k25_tool_declaration_is_typescript(kimi_renderer):
    """Test that K2.5 uses TypeScript-style tool declarations."""
    tools = [get_tool_spec()]
    prefix_messages = kimi_renderer.create_conversation_prefix_with_tools(tools)

    assert len(prefix_messages) >= 1
    assert prefix_messages[0]["role"] == "tool_declare"

    tool_content = prefix_messages[0]["content"]
    assert isinstance(tool_content, str)

    # Should be TypeScript style, not JSON
    assert "# Tools" in tool_content
    assert "namespace functions" in tool_content
    assert "type get_weather" in tool_content
    assert '"type":"function"' not in tool_content


@pytest.mark.parametrize("build_mode", ["generation", "supervised"])
def test_kimi_k25_tool_declaration_matches_hf(
    build_mode: str, kimi_tokenizer, kimi_renderer, hf_generation_prompt_length
):
    """Test that tool declarations match HF template output."""
    tools = [get_tool_spec()]
    openai_tools = [{"type": "function", "function": tool} for tool in tools]

    prefix_messages = kimi_renderer.create_conversation_prefix_with_tools(tools)
    user_msg = Message(role="user", content="What's the weather in NYC?")

    if build_mode == "generation":
        full_messages = prefix_messages + [user_msg]
        cookbook_tokens = kimi_renderer.build_generation_prompt(full_messages).to_ints()
    else:
        assistant_msg = Message(role="assistant", content="Let me check that for you.")
        full_messages = prefix_messages + [user_msg, assistant_msg]
        model_input, _ = kimi_renderer.build_supervised_example(full_messages)
        cookbook_tokens = model_input.to_ints()

    hf_messages = [
        {"role": "system", "content": kimi_renderer.DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "What's the weather in NYC?"},
    ]
    if build_mode == "supervised":
        hf_messages.append({"role": "assistant", "content": "Let me check that for you."})

    hf_tokens = get_hf_tokens(
        kimi_tokenizer, hf_messages, hf_generation_prompt_length,
        tools=openai_tools, for_generation=(build_mode == "generation")
    )

    assert cookbook_tokens == hf_tokens, (
        f"Mode: {build_mode}\n"
        f"Cookbook string: {kimi_tokenizer.decode(cookbook_tokens)}\n"
        f"HF string: {kimi_tokenizer.decode(hf_tokens)}"
    )


# =============================================================================
# Thinking Content Tests
# =============================================================================


@pytest.mark.parametrize("build_mode", ["generation", "supervised"])
def test_kimi_k25_thinking_preserved_in_suffix(build_mode: str, kimi_tokenizer, kimi_renderer):
    """Test that thinking is preserved for messages in the suffix (after last non-tool-call assistant)."""
    # For supervised, thinking in last assistant should be preserved
    # For generation with tool calls, thinking in tool-calling assistants should be preserved
    if build_mode == "supervised":
        messages: list[Message] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me calculate. 2+2=4."},
                    {"type": "text", "text": "The answer is 4."},
                ],
            },
        ]
        model_input, _ = kimi_renderer.build_supervised_example(messages)
        decoded = kimi_tokenizer.decode(model_input.to_ints())
    else:
        # Generation with tool calls - thinking should be preserved
        messages, tools = get_tool_call_conversation_for_generation()
        prefix_messages = kimi_renderer.create_conversation_prefix_with_tools(
            tools, system_prompt="You are a helpful assistant."
        )
        prefix_messages = [m for m in prefix_messages if m["role"] == "tool_declare"]
        full_messages = prefix_messages + messages
        gen_prompt = kimi_renderer.build_generation_prompt(full_messages)
        decoded = kimi_tokenizer.decode(gen_prompt.to_ints())

    # Thinking should be preserved
    if build_mode == "supervised":
        assert "<think>Let me calculate. 2+2=4.</think>" in decoded
    else:
        assert "<think>I need to check the weather in New York City.</think>" in decoded


@pytest.mark.parametrize("build_mode", ["generation", "supervised"])
def test_kimi_k25_thinking_stripped_in_history(build_mode: str, kimi_tokenizer, kimi_renderer):
    """Test that thinking is stripped for historical messages (before last non-tool-call assistant)."""
    # Conversation with historical assistant message followed by more turns
    messages: list[Message] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "HISTORICAL_THINKING_SHOULD_BE_STRIPPED"},
                {"type": "text", "text": "The answer is 4."},
            ],
        },
        {"role": "user", "content": "What is 3+3?"},
    ]

    if build_mode == "supervised":
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "SUFFIX_THINKING_PRESERVED"},
                {"type": "text", "text": "The answer is 6."},
            ],
        })
        model_input, _ = kimi_renderer.build_supervised_example(messages)
        decoded = kimi_tokenizer.decode(model_input.to_ints())
    else:
        gen_prompt = kimi_renderer.build_generation_prompt(messages)
        decoded = kimi_tokenizer.decode(gen_prompt.to_ints())

    # Historical thinking should be stripped
    assert "HISTORICAL_THINKING_SHOULD_BE_STRIPPED" not in decoded
    assert "<think></think>The answer is 4." in decoded

    # Suffix thinking should be preserved (only for supervised)
    if build_mode == "supervised":
        assert "SUFFIX_THINKING_PRESERVED" in decoded


# =============================================================================
# EOT Token Tests
# =============================================================================


def test_kimi_k25_eot_parsing(kimi_tokenizer, kimi_renderer):
    """Test EOT token parsing for K2.5 renderer."""
    # Test with EOT token
    test_response = "The answer is 42.<|im_end|>"
    response_tokens = kimi_tokenizer.encode(test_response)

    message, format_correct = kimi_renderer.parse_response(response_tokens)
    assert message["role"] == "assistant"
    assert message["content"] == "The answer is 42."
    assert format_correct is True

    # Test without EOT token
    test_response_no_eot = "The answer is 42."
    response_tokens_no_eot = kimi_tokenizer.encode(test_response_no_eot)

    message, format_correct = kimi_renderer.parse_response(response_tokens_no_eot)
    assert message["role"] == "assistant"
    assert message["content"] == "The answer is 42."
    assert format_correct is False
