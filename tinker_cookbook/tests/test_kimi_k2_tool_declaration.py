"""Tests for Kimi K2 tool declaration rendering."""

import json

import pytest
from transformers import AutoTokenizer

from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import ToolSpec, Message, ensure_text
from tinker_cookbook.tokenizer_utils import get_tokenizer


@pytest.mark.parametrize(
    "tools,expected_order",
    [
        # Single tool
        (
            [
                {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string", "description": "City name"}},
                        "required": ["location"],
                    },
                }
            ],
            ["tool_declare", "system"],
        ),
        # Multiple tools
        (
            [
                {
                    "name": "tool_a",
                    "description": "Tool A",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "name": "tool_b",
                    "description": "Tool B",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
            ["tool_declare", "system"],
        ),
    ],
)
def test_tool_declaration_message_order(tools, expected_order):
    """Test that tool_declare message comes before system message."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    renderer = get_renderer("kimi_k2", tokenizer)

    messages = renderer.create_conversation_prefix_with_tools(tools, "")

    actual_order = [msg["role"] for msg in messages]
    assert actual_order == expected_order, (
        f"Expected message order {expected_order}, got {actual_order}. "
        f"Tool declaration should come BEFORE system message per HF chat template."
    )


def test_tool_declaration_no_duplicate_system():
    """Test that tool declaration doesn't result in duplicate system messages."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    renderer = get_renderer("kimi_k2", tokenizer)

    tools: list[ToolSpec] = [
        {"name": "test", "description": "Test", "parameters": {"type": "object", "properties": {}}}
    ]
    prefix = renderer.create_conversation_prefix_with_tools(tools, "")
    messages = prefix + [{"role": "user", "content": "Test"}]

    # Build generation prompt (this triggers _ensure_system_message)
    prompt = renderer.build_generation_prompt(messages)
    prompt_str = tokenizer.decode(prompt.to_ints())

    # Count occurrences of system messages
    system_count = prompt_str.count("<|im_system|>system<|im_middle|>")

    assert system_count == 1, (
        f"Expected exactly 1 system message, found {system_count}. Prompt:\n{prompt_str[:500]}"
    )


def test_tool_json_keys_are_sorted():
    """Test that tool declaration JSON has sorted keys at all nesting levels."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    renderer = get_renderer("kimi_k2", tokenizer)

    tools: list[ToolSpec] = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit": {"type": "string", "description": "Temperature unit"},
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location", "unit"],
            },
        }
    ]

    messages = renderer.create_conversation_prefix_with_tools(tools, "")
    tool_declare_content = ensure_text(messages[0]["content"])

    # Parse the JSON to check key ordering
    tools_parsed = json.loads(tool_declare_content)

    # Check top-level keys (should be alphabetically sorted)
    top_level_keys = list(tools_parsed[0].keys())
    sorted_top_level = sorted(top_level_keys)
    assert top_level_keys == sorted_top_level, (
        f"Top-level keys not sorted: {top_level_keys} != {sorted_top_level}"
    )

    # Check function object keys
    function_keys = list(tools_parsed[0]["function"].keys())
    sorted_function_keys = sorted(function_keys)
    assert function_keys == sorted_function_keys, (
        f"Function keys not sorted: {function_keys} != {sorted_function_keys}"
    )

    # Check nested parameters keys
    params_keys = list(tools_parsed[0]["function"]["parameters"].keys())
    sorted_params_keys = sorted(params_keys)
    assert params_keys == sorted_params_keys, (
        f"Parameters keys not sorted: {params_keys} != {sorted_params_keys}"
    )


def test_tool_declaration_matches_hf_tokens():
    """Test that tool declaration produces identical tokens to HuggingFace."""
    # Define tools in ToolSpec format (what tinker-cookbook accepts)
    tools_toolspec: list[ToolSpec] = [
        {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state"},
                    "unit": {
                        "type": "string",
                        "enum": ["C", "F"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location", "unit"],
            },
        }
    ]

    # Convert to OpenAI format for HF (tinker-cookbook does this wrapping internally)
    tools_openai = [{"type": "function", "function": tool} for tool in tools_toolspec]

    messages: list[Message] = [{"role": "user", "content": "What's the weather in SF?"}]

    # Tinker-cookbook approach
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    renderer = get_renderer("kimi_k2", tokenizer)
    convo = renderer.create_conversation_prefix_with_tools(tools_toolspec, "") + messages
    cookbook_tokens = renderer.build_generation_prompt(convo).to_ints()

    # HuggingFace approach (pass OpenAI format to match tinker-cookbook's output)
    hf_tokenizer = AutoTokenizer.from_pretrained(
        "moonshotai/Kimi-K2-Thinking", trust_remote_code=True
    )
    hf_tokens = hf_tokenizer.apply_chat_template(
        messages, tools=tools_openai, tokenize=True, add_generation_prompt=True
    )

    # Compare tokens
    cookbook_str = tokenizer.decode(cookbook_tokens)
    hf_str = hf_tokenizer.decode(hf_tokens)

    assert cookbook_tokens == hf_tokens, (
        f"Token mismatch!\n"
        f"Cookbook tokens: {len(cookbook_tokens)}\n"
        f"HF tokens: {len(hf_tokens)}\n"
        f"\nCookbook string:\n{cookbook_str[:500]}\n"
        f"\nHF string:\n{hf_str[:500]}\n"
        f"\nFirst difference at token {_find_first_diff_index(cookbook_tokens, hf_tokens)}"
    )


def test_tool_declaration_string_matches_hf():
    """Test that tool declaration string matches HuggingFace exactly."""
    # ToolSpec format for tinker-cookbook
    tools_toolspec: list[ToolSpec] = [
        {
            "name": "test",
            "description": "Test tool",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    # OpenAI format for HF
    tools_openai = [{"type": "function", "function": tool} for tool in tools_toolspec]
    messages_list: list[Message] = [{"role": "user", "content": "Test"}]

    # Tinker-cookbook
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    renderer = get_renderer("kimi_k2", tokenizer)
    convo = renderer.create_conversation_prefix_with_tools(tools_toolspec, "") + messages_list
    cookbook_prompt = renderer.build_generation_prompt(convo)
    cookbook_str = tokenizer.decode(cookbook_prompt.to_ints())

    # HuggingFace (pass OpenAI format)
    hf_tokenizer = AutoTokenizer.from_pretrained(
        "moonshotai/Kimi-K2-Thinking", trust_remote_code=True
    )
    hf_str = hf_tokenizer.apply_chat_template(
        messages_list, tools=tools_openai, tokenize=False, add_generation_prompt=True
    )

    assert cookbook_str == hf_str, (
        f"String mismatch!\n"
        f"\n=== COOKBOOK ===\n{cookbook_str[:800]}\n"
        f"\n=== HF ===\n{hf_str[:800]}\n"
    )


def test_empty_tools_list():
    """Test that empty tools list doesn't cause issues."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    renderer = get_renderer("kimi_k2", tokenizer)

    messages = renderer.create_conversation_prefix_with_tools([], "")

    # Should have exactly one system message
    assert len(messages) == 1
    assert messages[0]["role"] == "system"


def test_custom_system_prompt_with_tools():
    """Test that custom system prompt is preserved when using tools."""
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    renderer = get_renderer("kimi_k2", tokenizer)

    tools: list[ToolSpec] = [
        {"name": "test", "description": "Test", "parameters": {"type": "object", "properties": {}}}
    ]
    custom_prompt = "You are a helpful assistant specialized in weather."

    messages = renderer.create_conversation_prefix_with_tools(tools, custom_prompt)

    # Should have tool_declare first, then system with custom prompt
    assert len(messages) == 2
    assert messages[0]["role"] == "tool_declare"
    assert messages[1]["role"] == "system"
    assert messages[1]["content"] == custom_prompt


def _find_first_diff_index(list1, list2):
    """Helper to find first index where two lists differ."""
    for i, (a, b) in enumerate(zip(list1, list2)):
        if a != b:
            return i
    return min(len(list1), len(list2))
