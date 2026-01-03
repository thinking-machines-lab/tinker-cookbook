"""Tests for Qwen3 tool declaration format compatibility with HuggingFace.

These tests verify that Qwen3 renderers produce identical tool declarations
to HuggingFace's chat templates when using the tools parameter.
"""

import json

import pytest
from transformers import AutoTokenizer

from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import ToolSpec, Message, ensure_text
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Test against multiple Qwen3 model variants
QWEN3_MODELS = [
    ("Qwen/Qwen3-30B-A3B", "qwen3"),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", "qwen3_instruct"),
]


@pytest.mark.parametrize("model_name,renderer_name", QWEN3_MODELS)
def test_qwen3_tool_json_formatting(model_name: str, renderer_name: str):
    """Test that Qwen3 tool JSON uses correct separators to match HF.

    HF's tojson filter uses:
    - separators=(', ', ': ') with spaces after colons/commas
    - No key sorting (preserves insertion order)
    """
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # Tools with nested structure
    tools: list[ToolSpec] = [
        {
            "name": "search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results"},
                },
                "required": ["query"],
            },
        }
    ]

    messages = renderer.create_conversation_prefix_with_tools(tools, "")
    system_msg = messages[0]
    content_str = ensure_text(system_msg["content"])

    # Extract the JSON from the <tools>...</tools> section
    start_marker = "<tools>\n"
    end_marker = "\n</tools>"
    start_idx = content_str.index(start_marker) + len(start_marker)
    end_idx = content_str.index(end_marker)
    tool_json_str = content_str[start_idx:end_idx]

    # Parse to verify it's valid JSON
    _ = json.loads(tool_json_str)

    # Re-serialize with HF-compatible settings (no sort_keys)
    expected_json = json.dumps(
        {"type": "function", "function": tools[0]},
        separators=(", ", ": "),
    )

    # Check formatting
    assert tool_json_str == expected_json, (
        f"JSON formatting doesn't match HF expectations.\n"
        f"Expected (HF format):\n{expected_json}\n\n"
        f"Got (cookbook):\n{tool_json_str}\n\n"
        f"Differences:\n"
        f"  - Separators: HF uses (', ', ': ') with spaces\n"
        f"  - Key order: HF preserves insertion order (no sorting)"
    )

    # Verify JSON uses spaces after colons
    assert '": ' in tool_json_str, "JSON should have space after colons"
    assert '", ' in tool_json_str, "JSON should have space after commas"


@pytest.mark.parametrize("model_name,renderer_name", QWEN3_MODELS)
def test_qwen3_tool_declaration_matches_hf_tokens(model_name: str, renderer_name: str):
    """Test that tool declaration produces identical tokens to HuggingFace."""
    tokenizer = get_tokenizer(model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # Define tools in ToolSpec format (what tinker-cookbook accepts)
    tools_toolspec: list[ToolSpec] = [
        {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]

    # Convert to OpenAI format for HF
    tools_openai = [{"type": "function", "function": tool} for tool in tools_toolspec]

    messages_list: list[Message] = [{"role": "user", "content": "What's the weather in SF?"}]

    # Tinker-cookbook approach
    convo = renderer.create_conversation_prefix_with_tools(tools_toolspec, "") + messages_list
    cookbook_tokens = renderer.build_generation_prompt(convo).to_ints()

    # HuggingFace approach
    hf_tokens = hf_tokenizer.apply_chat_template(
        messages_list, tools=tools_openai, tokenize=True, add_generation_prompt=True
    )

    assert cookbook_tokens == hf_tokens, (
        f"Token mismatch between cookbook and HF!\n"
        f"Cookbook tokens ({len(cookbook_tokens)}): {cookbook_tokens}\n"
        f"Cookbook string:\n{tokenizer.decode(cookbook_tokens)}\n\n"
        f"HF tokens ({len(hf_tokens)}): {hf_tokens}\n"
        f"HF string:\n{hf_tokenizer.decode(hf_tokens)}"
    )


@pytest.mark.parametrize("model_name,renderer_name", QWEN3_MODELS)
def test_qwen3_tool_declaration_string_matches_hf(model_name: str, renderer_name: str):
    """Test that tool declaration produces identical string to HuggingFace."""
    tokenizer = get_tokenizer(model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    tools_toolspec: list[ToolSpec] = [
        {
            "name": "calculate",
            "description": "Perform calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        }
    ]

    tools_openai = [{"type": "function", "function": tool} for tool in tools_toolspec]
    messages_list: list[Message] = [{"role": "user", "content": "What is 2+2?"}]

    # Tinker-cookbook approach
    convo = renderer.create_conversation_prefix_with_tools(tools_toolspec, "") + messages_list
    cookbook_tokens = renderer.build_generation_prompt(convo).to_ints()
    cookbook_string = tokenizer.decode(cookbook_tokens)

    # HuggingFace approach
    hf_string = hf_tokenizer.apply_chat_template(
        messages_list, tools=tools_openai, tokenize=False, add_generation_prompt=True
    )

    assert cookbook_string == hf_string, (
        f"String mismatch between cookbook and HF!\n"
        f"=== COOKBOOK ===\n{cookbook_string}\n\n"
        f"=== HF ===\n{hf_string}"
    )


@pytest.mark.parametrize("model_name,renderer_name", QWEN3_MODELS)
def test_qwen3_multiple_tools(model_name: str, renderer_name: str):
    """Test that multiple tools are formatted correctly."""
    tokenizer = get_tokenizer(model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    tools_toolspec: list[ToolSpec] = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "get_time",
            "description": "Get time",
            "parameters": {"type": "object", "properties": {}},
        },
    ]

    tools_openai = [{"type": "function", "function": tool} for tool in tools_toolspec]
    messages_list: list[Message] = [{"role": "user", "content": "Hello"}]

    # Tinker-cookbook approach
    convo = renderer.create_conversation_prefix_with_tools(tools_toolspec, "") + messages_list
    cookbook_tokens = renderer.build_generation_prompt(convo).to_ints()

    # HuggingFace approach
    hf_tokens = hf_tokenizer.apply_chat_template(
        messages_list, tools=tools_openai, tokenize=True, add_generation_prompt=True
    )

    assert cookbook_tokens == hf_tokens, (
        f"Token mismatch with multiple tools!\n"
        f"Cookbook: {tokenizer.decode(cookbook_tokens)}\n\n"
        f"HF: {hf_tokenizer.decode(hf_tokens)}"
    )


@pytest.mark.parametrize("model_name,renderer_name", QWEN3_MODELS)
def test_qwen3_empty_tools_list(model_name: str, renderer_name: str):
    """Test that empty tools list doesn't include tool section."""
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    messages = renderer.create_conversation_prefix_with_tools([], "")

    # Should return a system message with just the default system prompt (or empty)
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    # Should not contain tool-related text
    assert "<tools>" not in messages[0]["content"]


@pytest.mark.parametrize("model_name,renderer_name", QWEN3_MODELS)
def test_qwen3_custom_system_prompt_with_tools(model_name: str, renderer_name: str):
    """Test that custom system prompt is combined with tools."""
    tokenizer = get_tokenizer(model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    custom_prompt = "You are a helpful assistant."
    tools_toolspec: list[ToolSpec] = [
        {
            "name": "search",
            "description": "Search",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    tools_openai = [{"type": "function", "function": tool} for tool in tools_toolspec]
    messages_list: list[Message] = [{"role": "user", "content": "Help me"}]

    # Tinker-cookbook approach
    convo = (
        renderer.create_conversation_prefix_with_tools(tools_toolspec, custom_prompt)
        + messages_list
    )
    cookbook_tokens = renderer.build_generation_prompt(convo).to_ints()

    # HuggingFace approach - need to manually add system message
    hf_messages = [{"role": "system", "content": custom_prompt}] + messages_list
    hf_tokens = hf_tokenizer.apply_chat_template(
        hf_messages, tools=tools_openai, tokenize=True, add_generation_prompt=True
    )

    assert cookbook_tokens == hf_tokens, (
        f"Token mismatch with custom system prompt!\n"
        f"Cookbook: {tokenizer.decode(cookbook_tokens)}\n\n"
        f"HF: {hf_tokenizer.decode(hf_tokens)}"
    )


@pytest.mark.parametrize("model_name,renderer_name", QWEN3_MODELS)
def test_qwen3_preserves_insertion_order(model_name: str, renderer_name: str):
    """Test that JSON keys preserve insertion order (not sorted)."""
    tokenizer = get_tokenizer(model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # Tool with properties in specific order
    tools_toolspec: list[ToolSpec] = [
        {
            "name": "complex_tool",
            "description": "A complex tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "zebra": {"type": "string"},
                    "apple": {"type": "string"},
                },
            },
        }
    ]

    tools_openai = [{"type": "function", "function": tool} for tool in tools_toolspec]
    messages_list: list[Message] = [{"role": "user", "content": "Test"}]

    # Tinker-cookbook approach
    convo = renderer.create_conversation_prefix_with_tools(tools_toolspec, "") + messages_list
    cookbook_tokens = renderer.build_generation_prompt(convo).to_ints()

    # HuggingFace approach
    hf_tokens = hf_tokenizer.apply_chat_template(
        messages_list, tools=tools_openai, tokenize=True, add_generation_prompt=True
    )

    # Should match exactly (HF doesn't sort, preserves insertion order)
    assert cookbook_tokens == hf_tokens, (
        f"Token mismatch - key ordering issue!\n"
        f"Cookbook: {tokenizer.decode(cookbook_tokens)}\n\n"
        f"HF: {hf_tokenizer.decode(hf_tokens)}"
    )
