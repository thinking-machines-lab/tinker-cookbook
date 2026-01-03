"""
Tests for tinker_cookbook renderers against HuggingFace chat templates.

These tests verify that tinker-cookbook renderers produce identical token sequences
to HuggingFace's chat templates. This is important because:

1. The OpenAI-compatible inference endpoint (/chat/completions) uses HuggingFace
   chat templates to render conversations to tokens.
2. Users who train with tinker-cookbook and want to use the OpenAI endpoint for
   inference need their training to use HF-compatible rendering.

For models with thinking capabilities (Qwen3, DeepSeek), we test both the default
renderer (thinking enabled) and the disable_thinking variant.

See docs/rendering.mdx for more details on the rendering system.
See docs/compatible-apis/openai.mdx for the OpenAI-compatible endpoint documentation.

Testing guidelines:
- Don't test things that are clearly verified by HF equivalence tests (build_generation_prompt,
  build_supervised_example with basic conversations). HF equivalence tests ensure correctness.
- DO test parse_response and parsing logic - HF doesn't do parsing, so we need those tests.
- Keep tests focused on tricky logic, not trivial operations.
"""

from typing import Callable, cast
import copy
import json

import pytest
import tinker
from transformers.models.auto.tokenization_auto import AutoTokenizer

from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.model_info import get_model_attributes, get_recommended_renderer_name
from tinker_cookbook.renderers import (
    DeepSeekV3ThinkingRenderer,
    Message,
    Qwen3Renderer,
    RenderContext,
    TextPart,
    ThinkingPart,
    ToolCall,
    get_renderer,
)
from tinker_cookbook.renderers.base import ensure_list, ensure_text
from tinker_cookbook.tests.conversation_generator import generate_conversation
from tinker_cookbook.tokenizer_utils import get_tokenizer


# =============================================================================
# Test Conversation Definitions
# =============================================================================
# These functions provide reusable conversations for testing different scenarios.
# Each returns a list of Message dicts that can be used across multiple tests.


def get_basic_3turn_conversation() -> list[Message]:
    """Simple 3-turn conversation: user -> assistant -> user.

    This is the standard test case for generation prompt testing.
    """
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
    ]


def get_basic_2turn_conversation() -> list[Message]:
    """Simple 2-turn conversation: user -> assistant.

    This is the standard test case for supervised example testing.
    """
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
    ]


def get_system_message_3turn_conversation() -> list[Message]:
    """3-turn conversation with a nontrivial system message.

    Tests that renderers correctly handle system messages with real instructions.
    """
    return [
        {
            "role": "system",
            "content": "You are a helpful coding assistant. Always explain your reasoning step by step.",
        },
        {"role": "user", "content": "How do I reverse a string in Python?"},
        {"role": "assistant", "content": "You can use slicing: `s[::-1]` to reverse a string."},
        {"role": "user", "content": "Can you show me another way?"},
    ]


def get_system_message_2turn_conversation() -> list[Message]:
    """2-turn conversation with a nontrivial system message.

    Tests that renderers correctly handle system messages with real instructions.
    """
    return [
        {
            "role": "system",
            "content": "You are a helpful coding assistant. Always explain your reasoning step by step.",
        },
        {"role": "user", "content": "How do I reverse a string in Python?"},
        {"role": "assistant", "content": "You can use slicing: `s[::-1]` to reverse a string."},
    ]


def get_basic_4turn_conversation() -> list[Message]:
    """Simple 4-turn conversation: user -> assistant -> user -> assistant.

    This is the standard test case for multi-turn supervised example testing.
    """
    return [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "user", "content": "And what is 3+3?"},
        {"role": "assistant", "content": "The answer is 6."},
    ]


def get_tool_call_conversation() -> list[Message]:
    """Full tool use conversation with tool call and response.

    Includes: user request -> assistant tool call -> tool response -> assistant final answer.
    Ends with assistant (for supervised tests).
    """
    return [
        {"role": "user", "content": "What's the weather in San Francisco?"},
        {
            "role": "assistant",
            "content": "I'll check the weather for you.",
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="get_weather",
                        arguments='{"location": "San Francisco"}',
                    ),
                    id="call_123",
                )
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 72, "condition": "sunny"}',
            "tool_call_id": "call_123",
            "name": "get_weather",  # Required by GptOss, optional for others
        },
        {"role": "assistant", "content": "The weather in San Francisco is sunny with 72°F."},
    ]


def get_tool_call_gen_conversation() -> list[Message]:
    """Tool use conversation for generation testing (ends with tool response).

    Tests generating assistant response after receiving tool output.
    """
    return get_tool_call_conversation()[:-1]


def get_4turn_thinking_conversation() -> list[Message]:
    """4-turn conversation with ThinkingPart in assistant messages.

    Tests thinking content handling in multi-turn conversations.
    """
    return [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="First turn reasoning here."),
                TextPart(type="text", text="The answer is 4."),
            ],
        },
        {"role": "user", "content": "And what is 3+3?"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="Second turn reasoning here."),
                TextPart(type="text", text="The answer is 6."),
            ],
        },
    ]


def get_thinking_with_whitespace_conversation() -> list[Message]:
    """Conversation with leading whitespace in ThinkingPart and TextPart."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="\nLet me respond politely.\n"),
                TextPart(type="text", text="\n\nI'm fine, thank you!"),
            ],
        },
    ]


def get_multiturn_thinking_conversation() -> list[Message]:
    """Multi-turn conversation with thinking in assistant messages.

    Used for testing extension property with thinking content.
    """
    return [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="Let me add 2+2."),
                TextPart(type="text", text="The answer is 4."),
            ],
        },
        {"role": "user", "content": "What is 3+3?"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="Let me add 3+3."),
                TextPart(type="text", text="The answer is 6."),
            ],
        },
    ]


def get_multiturn_tool_conversation() -> list[Message]:
    """Multi-turn conversation with tool calls.

    Used for testing extension property with tool calling.
    """
    return [
        {"role": "user", "content": "What's the weather in NYC?"},
        {
            "role": "assistant",
            "content": "Let me check the weather for you.",
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="get_weather",
                        arguments='{"location": "NYC"}',
                    ),
                    id="call_1",
                )
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 72, "condition": "sunny"}',
            "tool_call_id": "call_1",
            "name": "get_weather",
        },
        {"role": "assistant", "content": "The weather in NYC is sunny with 72°F."},
        {"role": "user", "content": "What about San Francisco?"},
        {
            "role": "assistant",
            "content": "Let me check SF weather.",
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="get_weather",
                        arguments='{"location": "San Francisco"}',
                    ),
                    id="call_2",
                )
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 65, "condition": "foggy"}',
            "tool_call_id": "call_2",
            "name": "get_weather",
        },
        {"role": "assistant", "content": "San Francisco is foggy at 65°F."},
    ]


def get_multiturn_thinking_and_tool_conversation() -> list[Message]:
    """Multi-turn conversation with both thinking AND tool calls.

    Tests complex interactions with both thinking blocks and tool calling.
    """
    return [
        {"role": "user", "content": "What's the weather in NYC?"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="I need to check the weather API."),
                TextPart(type="text", text="Let me look that up."),
            ],
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="get_weather",
                        arguments='{"location": "NYC"}',
                    ),
                    id="call_1",
                )
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 72}',
            "tool_call_id": "call_1",
            "name": "get_weather",
        },
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="The API returned 72 degrees."),
                TextPart(type="text", text="NYC is 72°F."),
            ],
        },
        {"role": "user", "content": "Is that warm?"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="72°F is about 22°C, which is pleasant."),
                TextPart(type="text", text="Yes, 72°F is comfortable room temperature."),
            ],
        },
    ]


# Conversation registry for parametrized tests
# Maps conversation ID to (factory_function, description, requires_tools)
CONVERSATION_REGISTRY: dict[str, tuple[Callable[[], list[Message]], str, bool]] = {
    "basic_3turn": (get_basic_3turn_conversation, "basic 3-turn conversation", False),
    "basic_2turn": (get_basic_2turn_conversation, "basic 2-turn conversation", False),
    "system_3turn": (get_system_message_3turn_conversation, "3-turn with system message", False),
    "system_2turn": (get_system_message_2turn_conversation, "2-turn with system message", False),
    "tool_call": (get_tool_call_conversation, "tool call conversation (supervised)", True),
    "tool_call_gen": (get_tool_call_gen_conversation, "tool call conversation (generation)", True),
    "thinking_4turn": (get_4turn_thinking_conversation, "4-turn with thinking", False),
    "thinking_and_tool": (
        get_multiturn_thinking_and_tool_conversation,
        "multi-turn with thinking and tools",
        True,
    ),
    # Random conversations ending with user (for generation tests) - with tools/thinking
    **{
        f"random_gen_{seed}": (
            lambda s=seed: generate_conversation(s, end_with_assistant=False),
            f"random gen conversation (seed={seed})",
            True,  # May contain tools
        )
        for seed in [1, 42, 123, 456, 999]
    },
    # Random conversations ending with assistant (for supervised tests) - with tools/thinking
    **{
        f"random_sup_{seed}": (
            lambda s=seed: generate_conversation(s, end_with_assistant=True),
            f"random sup conversation (seed={seed})",
            True,  # May contain tools
        )
        for seed in [1, 42, 123, 456, 999]
    },
}


# Models that support tool calling in their renderers
TOOL_CAPABLE_MODELS = {
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "deepseek-ai/DeepSeek-V3.1",
    "moonshotai/Kimi-K2-Thinking",
    "openai/gpt-oss-20b",
}


# =============================================================================
# HF Compatibility Tests (parametrized by model and conversation)
# =============================================================================


# Models for HF generation/supervised tests
# Format: (model_name, renderer_override, hf_kwargs)
# - model_name: HuggingFace model ID
# - renderer_override: None to use get_recommended_renderer_name, or a specific renderer name
# - hf_kwargs: Extra kwargs to pass to apply_chat_template (e.g., {"thinking": True})
_HF_TEST_MODELS = [
    ("meta-llama/Llama-3.2-1B-Instruct", None, {}),
    ("Qwen/Qwen3-30B-A3B", None, {}),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", None, {}),
    ("deepseek-ai/DeepSeek-V3.1", None, {}),  # non-thinking (default)
    ("deepseek-ai/DeepSeek-V3.1", "deepseekv3_thinking", {"thinking": True}),  # thinking mode
    ("openai/gpt-oss-20b", None, {}),
    ("moonshotai/Kimi-K2-Thinking", None, {}),
    ("Qwen/Qwen3-VL-30B-A3B-Instruct", None, {}),
]

# Models whose tool call format matches HF's apply_chat_template exactly.
# Excluded models with intentional differences:
# - Llama3: see llama3.py docstring (double-encoding, assistant content handling)
# - gpt-oss: no HF template
_HF_TOOL_COMPATIBLE_MODELS = {
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "deepseek-ai/DeepSeek-V3.1",
    "moonshotai/Kimi-K2-Thinking",
}

# Conversations for generation tests (end with user message or tool response)
# Note: Random conversations are tested in consistency tests, not HF comparison,
# because they can have complex thinking+tool combinations with HF format quirks.
_GENERATION_CONVERSATIONS = [
    "basic_3turn",
    "system_3turn",
    "tool_call_gen",
]


def _conversation_has_tools(messages: list[Message]) -> bool:
    """Check if a conversation contains tool calls or tool responses."""
    for m in messages:
        if "tool_calls" in m or m["role"] == "tool":
            return True
    return False


def _add_llama3_date_prefix(messages: list[Message]) -> list[Message]:
    """Add date prefix to messages for Llama models."""
    # Use the hardcoded date from the mirrored tokenizer's chat template
    date_prefix = "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
    messages = copy.deepcopy(messages)
    if messages and messages[0]["role"] == "system":
        assert isinstance(messages[0]["content"], str)
        messages[0]["content"] = date_prefix + messages[0]["content"]
    else:
        messages = [Message(role="system", content=date_prefix)] + messages
    return messages


@pytest.mark.parametrize("conv_id", _GENERATION_CONVERSATIONS)
@pytest.mark.parametrize("model_name,renderer_override,hf_kwargs", _HF_TEST_MODELS)
def test_generation_against_hf_chat_templates(
    model_name: str, renderer_override: str | None, hf_kwargs: dict, conv_id: str
):
    """Test generation prompt against HF chat templates.

    Parametrized by model and conversation type. Tests that our renderer produces
    identical tokens to HuggingFace's chat template for the same conversation.
    """
    conv_factory, conv_desc, requires_tools = CONVERSATION_REGISTRY[conv_id]
    convo = conv_factory()

    # Skip tool-containing conversations for models with intentional HF differences
    if _conversation_has_tools(convo) and model_name not in _HF_TOOL_COMPATIBLE_MODELS:
        pytest.skip(
            f"{model_name} has intentional tool format differences from HF (see renderer docstring)"
        )

    tokenizer = get_tokenizer(model_name)
    attributes = get_model_attributes(model_name)
    image_processor = get_image_processor(model_name) if attributes.is_vl else None

    # Use renderer_override if provided, otherwise use default logic
    if renderer_override is not None:
        render_name = renderer_override
    elif model_name.startswith("openai"):
        render_name = "gpt_oss_medium_reasoning"
    else:
        render_name = get_recommended_renderer_name(model_name)
    cookbook_renderer = get_renderer(render_name, tokenizer, image_processor)

    modified_cookbook_convo = (
        _add_llama3_date_prefix(convo) if model_name.startswith("meta") else convo
    )
    # ^^^ modify the cookbook convo just for llama3, where we chose not to match the HF template

    # Extract tools from tool_declare messages and filter them out when converting to OpenAI format
    tools_for_hf = None
    hf_convo = []
    for m in convo:
        if m["role"] == "tool_declare":
            # Parse the JSON content to extract tools for HF
            tools_for_hf = json.loads(ensure_text(m["content"]))
        else:
            openai_msg = cookbook_renderer.to_openai_message(m)
            hf_convo.append(openai_msg)

    cookbook_tokens = cookbook_renderer.build_generation_prompt(modified_cookbook_convo).to_ints()
    hf_tokens = tokenizer.apply_chat_template(
        hf_convo, tools=tools_for_hf, add_generation_prompt=True, tokenize=True, **hf_kwargs
    )

    # Cast hf_tokens to list[int] for type checker - apply_chat_template with tokenize=True returns list[int]
    hf_tokens_list = cast(list[int], hf_tokens)

    assert cookbook_tokens == hf_tokens_list, (
        f"[{conv_desc}] Cookbook tokens: {cookbook_tokens}\n"
        f"Cookbook string: {tokenizer.decode(cookbook_tokens)}\n"
        f"HF tokens: {hf_tokens_list}\n"
        f"HF string: {tokenizer.decode(hf_tokens_list)}"
    )


# Models for supervised tests
# Excluded:
# - gpt-oss: analysis channel diverges from HF template
# - Qwen/Qwen3-30B-A3B: HF template adds empty <think> blocks to non-thinking messages
# Format: (model_name, renderer_override, hf_kwargs) - same as _HF_TEST_MODELS
_SUPERVISED_TEST_MODELS = [
    ("meta-llama/Llama-3.2-1B-Instruct", None, {}),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", None, {}),
    ("deepseek-ai/DeepSeek-V3.1", None, {}),  # non-thinking (default)
    ("deepseek-ai/DeepSeek-V3.1", "deepseekv3_thinking", {"thinking": True}),  # thinking mode
    ("moonshotai/Kimi-K2-Thinking", None, {}),
    ("Qwen/Qwen3-VL-30B-A3B-Instruct", None, {}),
]

# Conversations for supervised tests (end with assistant message)
# Note: Random conversations are tested in consistency tests, not HF comparison.
_SUPERVISED_CONVERSATIONS = [
    "basic_2turn",
    "system_2turn",
    "tool_call",
]


@pytest.mark.parametrize("conv_id", _SUPERVISED_CONVERSATIONS)
@pytest.mark.parametrize("model_name,renderer_override,hf_kwargs", _SUPERVISED_TEST_MODELS)
def test_supervised_example_against_hf_chat_templates(
    model_name: str, renderer_override: str | None, hf_kwargs: dict, conv_id: str
):
    """Test supervised example against HF chat templates.

    Parametrized by model and conversation type. Tests that our renderer produces
    identical tokens to HuggingFace's chat template for the same conversation.
    """
    conv_factory, conv_desc, requires_tools = CONVERSATION_REGISTRY[conv_id]
    convo = conv_factory()

    # Skip tool-containing conversations for models with intentional HF differences
    if _conversation_has_tools(convo) and model_name not in _HF_TOOL_COMPATIBLE_MODELS:
        pytest.skip(
            f"{model_name} has intentional tool format differences from HF (see renderer docstring)"
        )

    # Skip supervised tests for thinking renderer - we intentionally don't add </think> to the
    # last message (supervised target) so it can preserve ThinkingPart, unlike HF which always adds it
    if renderer_override in _RENDERERS_WITH_DIFFERENT_SUPERVISED_GEN_HEADERS:
        pytest.skip(
            f"{renderer_override} intentionally differs from HF for supervised target (no </think>)"
        )

    tokenizer = get_tokenizer(model_name)
    attributes = get_model_attributes(model_name)
    image_processor = get_image_processor(model_name) if attributes.is_vl else None

    # Use renderer_override if provided, otherwise use default logic
    if renderer_override is not None:
        render_name = renderer_override
    elif model_name.startswith("openai"):
        render_name = "gpt_oss_medium_reasoning"
    else:
        render_name = get_recommended_renderer_name(model_name)
    cookbook_renderer = get_renderer(render_name, tokenizer, image_processor)

    modified_cookbook_convo = (
        _add_llama3_date_prefix(convo) if model_name.startswith("meta") else convo
    )
    # ^^^ modify the cookbook convo just for llama3, where we chose not to match the HF template

    # Extract tools from tool_declare messages and filter them out when converting to OpenAI format
    tools_for_hf = None
    hf_convo = []
    for m in convo:
        if m["role"] == "tool_declare":
            # Parse the JSON content to extract tools for HF
            tools_for_hf = json.loads(ensure_text(m["content"]))
        else:
            openai_msg = cookbook_renderer.to_openai_message(m)
            hf_convo.append(openai_msg)

    cookbook_model_input, _ = cookbook_renderer.build_supervised_example(modified_cookbook_convo)
    cookbook_tokens = cookbook_model_input.to_ints()
    hf_output = tokenizer.apply_chat_template(
        hf_convo, tools=tools_for_hf, tokenize=False, add_generation_prompt=False, **hf_kwargs
    )
    assert isinstance(hf_output, str)
    hf_tokens = tokenizer.encode(hf_output.rstrip("\n"), add_special_tokens=False)

    assert cookbook_tokens == hf_tokens, (
        f"[{conv_desc}] Cookbook tokens: {cookbook_tokens}\n"
        f"Cookbook string: {tokenizer.decode(cookbook_tokens)}\n"
        f"HF tokens: {hf_tokens}\n"
        f"HF string: {tokenizer.decode(hf_tokens)}"
    )


@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen3-30B-A3B",
    ],
)
def test_tokenization_boundary_with_whitespace(model_name: str):
    """Test that whitespace in ThinkingPart/TextPart tokenizes correctly vs HF.

    Qwen3 is excluded from supervised HF tests (empty <think> blocks), so we
    test the whitespace case separately here.
    """
    convo = get_thinking_with_whitespace_conversation()

    tokenizer = get_tokenizer(model_name)
    attributes = get_model_attributes(model_name)
    image_processor = get_image_processor(model_name) if attributes.is_vl else None
    render_name = get_recommended_renderer_name(model_name)
    cookbook_renderer = get_renderer(render_name, tokenizer, image_processor)

    hf_convo = [cookbook_renderer.to_openai_message(m) for m in convo]

    cookbook_model_input, _ = cookbook_renderer.build_supervised_example(convo)
    cookbook_tokens = cookbook_model_input.to_ints()
    hf_output = tokenizer.apply_chat_template(hf_convo, tokenize=False, add_generation_prompt=False)
    assert isinstance(hf_output, str)
    hf_tokens = tokenizer.encode(hf_output.rstrip("\n"), add_special_tokens=False)

    # Verify decoded strings match (this should pass)
    cookbook_decoded = tokenizer.decode(cookbook_tokens)
    hf_decoded = tokenizer.decode(hf_tokens)
    assert cookbook_decoded == hf_decoded, "Decoded strings should match even if tokens differ"

    # Verify tokens match
    assert cookbook_tokens == hf_tokens, (
        f"Token mismatch with whitespace content.\n"
        f"Cookbook tokens: {cookbook_tokens}\n"
        f"HF tokens: {hf_tokens}\n"
        f"Both decode to: {cookbook_decoded!r}"
    )


# =============================================================================
# Tool Use Rendering Tests
# =============================================================================


@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen3-30B-A3B",
        # Llama3 does not support tool calling - see llama3.py docstring
        "deepseek-ai/DeepSeek-V3.1",
        "moonshotai/Kimi-K2-Thinking",
        "openai/gpt-oss-20b",
    ],
)
def test_tool_call_supervised_rendering(model_name: str):
    """Test that tool call conversations render without errors.

    Verifies that our renderers handle tool call conversations correctly
    for supervised learning.
    """
    convo = get_tool_call_conversation()

    tokenizer = get_tokenizer(model_name)
    attributes = get_model_attributes(model_name)
    image_processor = get_image_processor(model_name) if attributes.is_vl else None
    render_name = (
        get_recommended_renderer_name(model_name)
        if not model_name.startswith("openai")
        else "gpt_oss_medium_reasoning"
    )
    cookbook_renderer = get_renderer(render_name, tokenizer, image_processor)

    # Build supervised example - should not raise
    model_input, weights = cookbook_renderer.build_supervised_example(convo)
    tokens = model_input.to_ints()
    decoded = tokenizer.decode(tokens)

    # Verify basic structure
    assert len(tokens) > 0, "Should produce non-empty token sequence"
    assert len(weights) == len(tokens), "Weights should match token count"

    # Verify tool-related content appears in output
    # Different renderers format tool calls differently:
    # - Qwen3: <tool_call>{"name": "get_weather", ...}</tool_call>
    # - Llama3: <function=get_weather>...</function>
    # - DeepSeek: <｜tool▁sep｜>get_weather
    # - Kimi K2: Uses tool_id (functions.name:idx or just the id) + arguments
    # - GptOss: to=functions.get_weather<|channel|>commentary <|constrain|>json<|message|>{args}
    # Check for tool arguments which all formats include
    assert "San Francisco" in decoded, f"Tool argument should appear in rendered output: {decoded}"

    # Check for either the function name or the tool_call_id
    has_tool_indicator = "get_weather" in decoded or "call_123" in decoded
    assert has_tool_indicator, f"Tool name or ID should appear in rendered output: {decoded}"


# =============================================================================
# Thinking Stripping Tests (multi-turn with thinking content)
# =============================================================================


@pytest.mark.parametrize(
    "model_name,renderer_class",
    [
        ("Qwen/Qwen3-8B", Qwen3Renderer),
        ("deepseek-ai/DeepSeek-V3.1", DeepSeekV3ThinkingRenderer),
    ],
)
def test_strip_thinking_from_history_default(model_name: str, renderer_class):
    """
    Test that renderers with strip_thinking_from_history=True (default) only preserve
    the last assistant message's thinking. Earlier assistant thinking blocks are stripped.
    """
    tokenizer = get_tokenizer(model_name)
    renderer = renderer_class(tokenizer)  # Default strip_thinking_from_history=True

    messages = get_4turn_thinking_conversation()
    model_input, _ = renderer.build_supervised_example(messages)
    decoded = tokenizer.decode(model_input.to_ints())

    # First assistant message should have thinking stripped
    assert "First turn reasoning" not in decoded, (
        f"First turn thinking should be stripped:\n{decoded}"
    )
    # Second (last) assistant message should preserve thinking
    assert "Second turn reasoning" in decoded, f"Last turn thinking should be preserved:\n{decoded}"


@pytest.mark.parametrize(
    "model_name,renderer_class",
    [
        ("Qwen/Qwen3-8B", Qwen3Renderer),
        ("deepseek-ai/DeepSeek-V3.1", DeepSeekV3ThinkingRenderer),
    ],
)
def test_strip_thinking_from_history_false(model_name: str, renderer_class):
    """
    Test that strip_thinking_from_history=False preserves thinking in ALL messages.
    This mode is used for multi-turn RL where the extension property is needed.
    """
    tokenizer = get_tokenizer(model_name)
    renderer = renderer_class(tokenizer, strip_thinking_from_history=False)

    messages = get_4turn_thinking_conversation()
    model_input, _ = renderer.build_supervised_example(messages)
    decoded = tokenizer.decode(model_input.to_ints())

    # Both thinking blocks should be present
    assert "First turn reasoning" in decoded, (
        f"First thinking should be preserved with strip_thinking_from_history=False: {decoded}"
    )
    assert "Second turn reasoning" in decoded, f"Second thinking should be preserved: {decoded}"


# =============================================================================
# Qwen3 Disable Thinking Tests
# =============================================================================


def test_qwen3_disable_thinking_supervised():
    """
    Test that Qwen3DisableThinkingRenderer adds the correct empty thinking block
    to assistant messages for SFT, matching HF tokenizer with thinking=False.
    """
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    renderer = get_renderer("qwen3_disable_thinking", tokenizer)

    messages = get_basic_2turn_conversation()

    model_input, _ = renderer.build_supervised_example(messages)
    tinker_tokens = model_input.to_ints()
    tinker_decoded = tokenizer.decode(tinker_tokens)

    # Get expected format from official Qwen3 tokenizer with thinking=False
    hf_decoded = tokenizer.apply_chat_template(
        cast(list[dict[str, str]], messages), tokenize=False, thinking=False
    )

    # Verify the complete empty thinking block is present
    assert "<think>\n\n</think>\n\n" in tinker_decoded, (
        f"Renderer must add '<think>\\n\\n</think>\\n\\n' but got: {tinker_decoded}"
    )

    # Verify matches HF
    assert tinker_decoded == hf_decoded.rstrip("\n"), (
        f"Tinker and HuggingFace outputs differ:\n"
        f"TINKER:\n{tinker_decoded!r}\n\n"
        f"HUGGINGFACE:\n{hf_decoded!r}"
    )


def test_qwen3_disable_thinking_generation():
    """Test Qwen3DisableThinkingRenderer generation matches HF with enable_thinking=False."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    cookbook_renderer = get_renderer("qwen3_disable_thinking", tokenizer)

    convo = get_basic_3turn_conversation()

    cookbook_tokens = cookbook_renderer.build_generation_prompt(convo).to_ints()
    hf_tokens = tokenizer.apply_chat_template(
        cast(list[dict[str, str]], convo),
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )

    # Cast hf_tokens to list[int] for type checker - apply_chat_template with tokenize=True returns list[int]
    hf_tokens_list = cast(list[int], hf_tokens)

    assert cookbook_tokens == hf_tokens_list, (
        f"Cookbook tokens: {cookbook_tokens}\n"
        f"Cookbook string: {tokenizer.decode(cookbook_tokens)}\n"
        f"HF tokens: {hf_tokens_list}\n"
        f"HF string: {tokenizer.decode(hf_tokens_list)}"
    )


def test_qwen3_disable_thinking_4turn():
    """
    Test Qwen3DisableThinkingRenderer with 4-turn conversation.
    Only the last assistant message should have the empty thinking block
    (historical thinking is stripped, matching HF behavior).
    """
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    renderer = get_renderer("qwen3_disable_thinking", tokenizer)

    messages = get_basic_4turn_conversation()

    model_input, _ = renderer.build_supervised_example(messages)
    tinker_tokens = model_input.to_ints()
    tinker_decoded = tokenizer.decode(tinker_tokens)

    # Get expected format from HF
    hf_decoded = tokenizer.apply_chat_template(
        cast(list[dict[str, str]], messages), tokenize=False, thinking=False
    )

    assert tinker_decoded == hf_decoded.rstrip("\n"), (
        f"Tinker and HuggingFace outputs differ:\n"
        f"TINKER:\n{tinker_decoded!r}\n\n"
        f"HUGGINGFACE:\n{hf_decoded!r}"
    )


# =============================================================================
# Supervised/Generation/Parse Consistency Tests
# =============================================================================


def _split_by_weights(tokens: list[int], weights: list[float]) -> tuple[list[int], list[int]]:
    """Split token sequence into observation (weight=0) and action (weight=1) parts.

    Assumes weights are like 000...0111...1 (zeros then ones).
    Returns (ob, ac) where ob has all weight=0 tokens and ac has all weight=1 tokens.
    """
    assert len(tokens) == len(weights), (
        f"Token/weight length mismatch: {len(tokens)} vs {len(weights)}"
    )

    # Find the first non-zero weight
    first_nonzero = None
    for i, w in enumerate(weights):
        if w > 0:
            first_nonzero = i
            break

    if first_nonzero is None:
        # All zeros - no action tokens
        return tokens, []

    # Verify the pattern: all zeros before first_nonzero, all ones after
    for i, w in enumerate(weights):
        if i < first_nonzero:
            assert w == 0, f"Expected weight=0 at index {i}, got {w}"
        else:
            assert w == 1, f"Expected weight=1 at index {i}, got {w}"

    ob = tokens[:first_nonzero]
    ac = tokens[first_nonzero:]
    return ob, ac


def get_2turn_with_thinking() -> list[Message]:
    """2-turn conversation with thinking content in assistant message."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="Let me respond politely."),
                TextPart(type="text", text="I'm fine, thank you!"),
            ],
        },
    ]


# Renderers for the consistency test - (model_name, renderer_name)
_CONSISTENCY_RENDERERS = [
    ("meta-llama/Llama-3.2-1B-Instruct", "llama3"),
    ("meta-llama/Llama-3.2-1B-Instruct", "role_colon"),
    ("Qwen/Qwen3-8B", "qwen3"),
    ("Qwen/Qwen3-8B", "qwen3_disable_thinking"),
    ("Qwen/Qwen3-8B", "qwen3_instruct"),
    ("deepseek-ai/DeepSeek-V3.1", "deepseekv3"),
    ("deepseek-ai/DeepSeek-V3.1", "deepseekv3_thinking"),
    ("openai/gpt-oss-20b", "gpt_oss_medium_reasoning"),
    ("moonshotai/Kimi-K2-Thinking", "kimi_k2"),
]

# Conversations for the consistency test
_CONSISTENCY_CONVERSATIONS = [
    get_basic_2turn_conversation,
    get_2turn_with_thinking,
    get_tool_call_conversation,
    # Random conversations with tools/thinking
    lambda: generate_conversation(1, end_with_assistant=True),
    lambda: generate_conversation(42, end_with_assistant=True),
    lambda: generate_conversation(999, end_with_assistant=True),
]


# Renderers that don't support ThinkingPart content (use ensure_text)
_RENDERERS_WITHOUT_THINKING_SUPPORT = {"llama3", "role_colon"}

# Renderers that don't support tool calling
_RENDERERS_WITHOUT_TOOL_SUPPORT = {"role_colon"}

# Renderers that strip thinking in non-thinking mode (conversation must not have ThinkingPart)
_RENDERERS_WITH_THINKING_STRIPPING = {"qwen3_disable_thinking", "deepseekv3", "kimi_k2"}

# Renderers where supervised and generation have different headers (HF thinking=True behavior).
# These add </think> to supervised assistant headers but <think> to generation prompt,
# so observation != generation_prompt by design.
_RENDERERS_WITH_DIFFERENT_SUPERVISED_GEN_HEADERS = {"deepseekv3_thinking"}


@pytest.mark.parametrize("conversation_fn", _CONSISTENCY_CONVERSATIONS)
@pytest.mark.parametrize("model_name,renderer_name", _CONSISTENCY_RENDERERS)
def test_supervised_generation_parse_consistency(
    model_name: str, renderer_name: str, conversation_fn
):
    """Test consistency between build_supervised_example, build_generation_prompt, and parse_response.

    For train_on_what=LAST_ASSISTANT_MESSAGE, this test verifies:
    1. The supervised example produces weights like 000...0111...1
    2. Split tokens into (ob, ac) based on weights
    3. ob == build_generation_prompt(messages[:-1]).to_ints()
    4. parse_response(ac) returns the final message

    This ensures that:
    - The observation tokens match what the model would see at generation time
    - The action tokens can be parsed back to the original message
    """
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    messages = conversation_fn()

    # Check if this combination is supported based on actual message content
    def has_thinking(msgs):
        for m in msgs:
            content = m["content"]
            if isinstance(content, list):
                for p in content:
                    if p.get("type") == "thinking":
                        return True
        return False

    def has_tools(msgs):
        for m in msgs:
            if "tool_calls" in m or m["role"] == "tool":
                return True
        return False

    has_thinking_content = has_thinking(messages)
    has_tool_content = has_tools(messages)

    if has_thinking_content and renderer_name in _RENDERERS_WITHOUT_THINKING_SUPPORT:
        pytest.skip(f"{renderer_name} doesn't support ThinkingPart content")
    if has_thinking_content and renderer_name in _RENDERERS_WITH_THINKING_STRIPPING:
        pytest.skip(f"{renderer_name} strips thinking content, breaking roundtrip consistency")
    if has_tool_content and renderer_name in _RENDERERS_WITHOUT_TOOL_SUPPORT:
        pytest.skip(f"{renderer_name} doesn't support tool calling")
    if renderer_name in _RENDERERS_WITH_DIFFERENT_SUPERVISED_GEN_HEADERS:
        pytest.skip(
            f"{renderer_name} has different headers for supervised (</think>) vs generation (<think>)"
        )
    assert len(messages) >= 2, "Need at least 2 messages for this test"
    assert messages[-1]["role"] == "assistant", "Last message must be assistant"

    prefix_messages = messages[:-1]
    final_message = messages[-1]

    # Build supervised example
    from tinker_cookbook.renderers import TrainOnWhat

    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
    )
    sup_tokens = model_input.to_ints()
    weights_list = weights.tolist()

    # Split into observation and action
    ob, ac = _split_by_weights(sup_tokens, weights_list)

    # Build generation prompt for prefix
    gen_prompt = renderer.build_generation_prompt(prefix_messages)
    gen_tokens = gen_prompt.to_ints()

    # Check 1: Observation should match generation prompt
    ob_matches_gen = ob == gen_tokens
    if not ob_matches_gen:
        # Find where they diverge
        min_len = min(len(ob), len(gen_tokens))
        diverge_idx = min_len
        for i in range(min_len):
            if ob[i] != gen_tokens[i]:
                diverge_idx = i
                break

        ob_decoded = tokenizer.decode(ob)
        gen_decoded = tokenizer.decode(gen_tokens)

        # Show the discrepancy
        assert False, (
            f"Observation tokens do not match generation prompt for {renderer_name}.\n"
            f"Divergence at token {diverge_idx}:\n"
            f"  ob[{diverge_idx}:]:  {ob[diverge_idx : diverge_idx + 10]} = {tokenizer.decode(ob[diverge_idx : diverge_idx + 10])!r}\n"
            f"  gen[{diverge_idx}:]: {gen_tokens[diverge_idx : diverge_idx + 10]} = {tokenizer.decode(gen_tokens[diverge_idx : diverge_idx + 10])!r}\n"
            f"\nFull observation ({len(ob)} tokens):\n{ob_decoded!r}\n"
            f"\nFull generation prompt ({len(gen_tokens)} tokens):\n{gen_decoded!r}"
        )

    # Check 2: Parse the action tokens
    parsed_message, parse_success = renderer.parse_response(ac)

    # Check parse success
    assert parse_success, (
        f"Failed to parse action tokens for {renderer_name}.\n"
        f"Action tokens: {ac}\n"
        f"Decoded: {tokenizer.decode(ac)!r}\n"
        f"Parsed message: {parsed_message}"
    )

    # Check 3: Parsed content should match final message content
    # Normalize both to list form for comparison (handles string vs list[TextPart])
    parsed_normalized = ensure_list(parsed_message["content"])
    expected_normalized = ensure_list(final_message["content"])
    assert parsed_normalized == expected_normalized, (
        f"Parsed content does not match final message for {renderer_name}.\n"
        f"Expected: {expected_normalized!r}\n"
        f"Got: {parsed_normalized!r}"
    )


# =============================================================================
# EOT Parsing Tests
# =============================================================================


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [
        ("Qwen/Qwen3-30B-A3B", "qwen3"),
        ("Qwen/Qwen3-8B", "qwen3_disable_thinking"),
        ("meta-llama/Llama-3.2-1B-Instruct", "llama3"),
        # deepseekv3 defaults to non-thinking, deepseekv3_thinking is thinking mode
        ("deepseek-ai/DeepSeek-V3.1", "deepseekv3"),
        ("deepseek-ai/DeepSeek-V3.1", "deepseekv3_thinking"),
        ("openai/gpt-oss-20b", "gpt_oss_medium_reasoning"),
        ("moonshotai/Kimi-K2-Thinking", "kimi_k2"),
    ],
)
def test_eot_parsing(model_name: str, renderer_name: str):
    """Test EOT token parsing behavior for different renderers using real tokenizers."""
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # Get the appropriate EOT token for each renderer
    # Note: DeepSeek uses full-width pipes (｜) not ASCII pipes (|)
    eot_tokens = {
        "llama3": "<|eot_id|>",
        "qwen3": "<|im_end|>",
        "qwen3_disable_thinking": "<|im_end|>",
        "deepseekv3": "<｜end▁of▁sentence｜>",  # Full-width pipes
        "deepseekv3_thinking": "<｜end▁of▁sentence｜>",  # Full-width pipes
        "deepseekv3_disable_thinking": "<｜end▁of▁sentence｜>",  # Full-width pipes (alias)
        "gpt_oss_medium_reasoning": "<|return|>",
        "kimi_k2": "<|im_end|>",
    }
    eot_token = eot_tokens.get(renderer_name)
    if eot_token is None:
        raise ValueError(f"Unknown renderer: {renderer_name}")

    # Test case 1: Normal case with single EOT - should parse correctly
    test_response_with_eot = f"53 + 18 = 71{eot_token}"
    response_tokens = tokenizer.encode(test_response_with_eot, add_special_tokens=False)

    message, format_correct = renderer.parse_response(response_tokens)
    assert message["role"] == "assistant"
    assert message["content"] == "53 + 18 = 71"
    assert format_correct is True

    # Test case 2: No EOT token - should have format=False
    test_response_no_eot = "53 + 18 = 71"
    response_tokens_no_eot = tokenizer.encode(test_response_no_eot, add_special_tokens=False)

    message, format_correct = renderer.parse_response(response_tokens_no_eot)
    assert message["role"] == "assistant"
    assert message["content"] == "53 + 18 = 71"
    assert format_correct is False

    # Test case 3: Double EOT token - should raise ValueError
    test_response_double_eot = f"53 + 18 = 71{eot_token}{eot_token}"
    response_tokens_double_eot = tokenizer.encode(
        test_response_double_eot, add_special_tokens=False
    )

    with pytest.raises(ValueError, match="expected .* 1"):
        _ = renderer.parse_response(response_tokens_double_eot)


# =============================================================================
# DeepSeek Tool Call Tests
# =============================================================================


def test_deepseek_thinking_preserved_with_tool_calls():
    """
    Test that thinking is preserved in messages that have tool_calls.
    The thinking represents the model's reasoning about WHY it's making the tool call.
    """
    model_name = "deepseek-ai/DeepSeek-V3.1"
    tokenizer = get_tokenizer(model_name)
    renderer = DeepSeekV3ThinkingRenderer(tokenizer)  # Default strip_thinking_from_history=True

    messages: list[Message] = [
        {"role": "user", "content": "What's the weather in NYC?"},
        {
            "role": "assistant",
            "content": "<think>I need to check the weather.</think>Let me look that up.",
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="get_weather",
                        arguments='{"location": "NYC"}',
                    ),
                    id="call_1",
                )
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 72}',
            "tool_call_id": "call_1",
        },
        {"role": "assistant", "content": "The temperature in NYC is 72°F."},
    ]

    model_input, _ = renderer.build_supervised_example(messages)
    decoded = tokenizer.decode(model_input.to_ints())

    # Thinking in message with tool_calls should be preserved
    assert "I need to check the weather" in decoded, (
        f"Thinking in tool_call message should be preserved: {decoded}"
    )


def test_deepseek_post_tool_formatting():
    """
    Test that assistant messages following tool responses have correct formatting.
    Post-tool assistant messages should not have the role token or </think> prefix.
    """
    model_name = "deepseek-ai/DeepSeek-V3.1"
    tokenizer = get_tokenizer(model_name)
    renderer = DeepSeekV3ThinkingRenderer(tokenizer)

    messages: list[Message] = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="get_weather",
                        arguments='{"location": "NYC"}',
                    ),
                    id="call_1",
                )
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 72}',
            "tool_call_id": "call_1",
        },
        {"role": "assistant", "content": "The temperature is 72°F."},
    ]

    # Render each message individually to check formatting
    for idx, message in enumerate(messages):
        ctx = RenderContext(
            idx=idx,
            is_last=idx == len(messages) - 1,
            prev_message=messages[idx - 1] if idx > 0 else None,
        )
        follows_tool = ctx.prev_message is not None and ctx.prev_message["role"] == "tool"
        rendered = renderer.render_message(message, ctx)

        if message["role"] == "assistant" and follows_tool:
            # Post-tool assistant should have no header (no role token)
            header = rendered.header
            assert header is None or len(header.tokens) == 0, (
                f"Post-tool assistant should have no header, got: {header}"
            )

            # Output should not start with </think>
            output_chunk = rendered.output[0]
            assert isinstance(output_chunk, tinker.EncodedTextChunk), "Expected EncodedTextChunk"
            output_str = tokenizer.decode(list(output_chunk.tokens))
            assert not output_str.startswith("</think>"), (
                f"Post-tool assistant should not have </think> prefix: {output_str}"
            )


# =============================================================================
# Sequence Extension Property Tests
# =============================================================================


def _verify_extension_property(renderer, messages: list[Message], tokenizer):
    """
    Verify the sequence extension property for multi-turn conversations.

    The extension property holds when the full sequence at timestep t (observation + action)
    is a prefix of the observation at timestep t+1. This enables KV-cache reuse and O(T)
    compute scaling for T-turn trajectories.

    For a conversation [user1, asst1, user2, asst2, ...], we check:
    - (prompt_before_asst1 + asst1_completion) is prefix of prompt_before_asst2
    - (prompt_before_asst2 + asst2_completion) is prefix of prompt_before_asst3
    - etc.

    The "completion" for an assistant message is how it would be rendered as the model's
    output (with thinking, tool calls, etc.), not how it appears in history (where thinking
    might be stripped).
    """
    # Find all assistant message indices
    assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]

    if len(assistant_indices) < 2:
        return  # Need at least 2 assistant messages to test extension

    # Build sequences for comparison
    # seq[i] = observation before assistant i + completion for assistant i
    # We check if seq[i] is a prefix of observation before assistant i+1
    for i in range(len(assistant_indices) - 1):
        asst_idx = assistant_indices[i]
        next_asst_idx = assistant_indices[i + 1]

        # Build the assistant's completion - we need to render the assistant message
        # as it would appear when generated (with thinking preserved), not as it
        # would appear in history. We do this by building a supervised example and
        # extracting the tokens after the prompt.
        messages_through_asst = messages[: asst_idx + 1]
        model_input_through_asst, _ = renderer.build_supervised_example(messages_through_asst)
        seq_through_asst = model_input_through_asst.to_ints()

        # Build prompt before the next assistant message (observation_{t+1})
        context_before_next = messages[:next_asst_idx]
        prompt_before_next = renderer.build_generation_prompt(context_before_next).to_ints()

        # Check if seq_through_asst is a prefix of prompt_before_next
        is_prefix = prompt_before_next[: len(seq_through_asst)] == seq_through_asst
        if not is_prefix:
            # Decode for debugging
            seq_str = tokenizer.decode(seq_through_asst)
            next_prompt_str = tokenizer.decode(prompt_before_next)
            # Find where they diverge
            diverge_idx = 0
            for j in range(min(len(seq_through_asst), len(prompt_before_next))):
                if seq_through_asst[j] != prompt_before_next[j]:
                    diverge_idx = j
                    break
            else:
                diverge_idx = min(len(seq_through_asst), len(prompt_before_next))

            raise AssertionError(
                f"Extension property violated between assistant {i} and {i + 1}.\n"
                f"Full sequence through asst {i} (len={len(seq_through_asst)}) is NOT a prefix "
                f"of prompt before asst {i + 1} (len={len(prompt_before_next)}).\n"
                f"Divergence at token {diverge_idx}:\n"
                f"  Seq through asst[{diverge_idx}:]: {seq_through_asst[diverge_idx : diverge_idx + 10]}\n"
                f"  Next prompt[{diverge_idx}:]:      {prompt_before_next[diverge_idx : diverge_idx + 10]}\n"
                f"Sequence through assistant: {seq_str}\n"
                f"Next prompt: {next_prompt_str}"
            )


# Test extension property actually holds for renderers that claim it
# Format: (model_name, renderer_name_or_class, renderer_kwargs, conversation_fn)
# If renderer_name_or_class is a string, use get_renderer; if a class, instantiate directly
_EXTENSION_PROPERTY_TEST_PARAMS = [
    # Llama3 with basic multi-turn (tool calling not supported - see llama3.py docstring)
    ("meta-llama/Llama-3.2-1B-Instruct", "llama3", {}, get_basic_4turn_conversation),
    # RoleColon with basic multi-turn (doesn't support tools)
    ("meta-llama/Llama-3.2-1B-Instruct", "role_colon", {}, get_basic_4turn_conversation),
    # Qwen3 Instruct with basic multi-turn
    ("Qwen/Qwen3-8B", "qwen3_instruct", {}, get_basic_4turn_conversation),
    # Qwen3 Instruct with tool calls
    ("Qwen/Qwen3-8B", "qwen3_instruct", {}, get_multiturn_tool_conversation),
    # Qwen3 with strip_thinking_from_history=False (preserves thinking)
    (
        "Qwen/Qwen3-8B",
        Qwen3Renderer,
        {"strip_thinking_from_history": False},
        get_multiturn_thinking_conversation,
    ),
    # DeepSeek non-thinking with basic multi-turn
    ("deepseek-ai/DeepSeek-V3.1", "deepseekv3", {}, get_basic_4turn_conversation),
    # DeepSeek non-thinking with tool calls
    ("deepseek-ai/DeepSeek-V3.1", "deepseekv3", {}, get_multiturn_tool_conversation),
    # DeepSeek with strip_thinking_from_history=False (preserves thinking)
    (
        "deepseek-ai/DeepSeek-V3.1",
        DeepSeekV3ThinkingRenderer,
        {"strip_thinking_from_history": False},
        get_multiturn_thinking_conversation,
    ),
    # DeepSeek with strip_thinking_from_history=False + tool calls
    (
        "deepseek-ai/DeepSeek-V3.1",
        DeepSeekV3ThinkingRenderer,
        {"strip_thinking_from_history": False},
        get_multiturn_thinking_and_tool_conversation,
    ),
]


@pytest.mark.parametrize(
    "model_name,renderer_name_or_class,renderer_kwargs,conversation_fn",
    _EXTENSION_PROPERTY_TEST_PARAMS,
)
def test_extension_property_holds(
    model_name, renderer_name_or_class, renderer_kwargs, conversation_fn
):
    """
    Test that renderers with has_extension_property=True actually satisfy the property.
    For each conversation, verify that build_generation_prompt at successive assistant
    turns produces token sequences where each is a prefix of the next.
    """
    tokenizer = get_tokenizer(model_name)

    if isinstance(renderer_name_or_class, str):
        renderer = get_renderer(renderer_name_or_class, tokenizer)
    else:
        renderer = renderer_name_or_class(tokenizer, **renderer_kwargs)

    assert renderer.has_extension_property, (
        f"Expected {renderer_name_or_class} to have has_extension_property=True"
    )

    messages = conversation_fn()
    _verify_extension_property(renderer, messages, tokenizer)


def test_extension_property_breaks_when_expected():
    """
    Verify that extension property actually breaks for renderers that strip thinking.
    This confirms our test helper can detect violations.
    """
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    renderer = Qwen3Renderer(tokenizer, strip_thinking_from_history=True)

    assert not renderer.has_extension_property, "Default Qwen3Renderer should NOT have extension"

    messages = get_multiturn_thinking_conversation()

    # Extension should break - expect an assertion error
    with pytest.raises(AssertionError, match="Extension property violated"):
        _verify_extension_property(renderer, messages, tokenizer)
