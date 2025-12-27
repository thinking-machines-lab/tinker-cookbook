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

from functools import cache
from typing import Any, Callable, cast
import copy
from datetime import date


import pytest
import tinker
from transformers.models.auto.tokenization_auto import AutoTokenizer

from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.model_info import get_model_attributes, get_recommended_renderer_name
from tinker_cookbook.renderers import (
    DeepSeekV3ThinkingRenderer,
    GptOssRenderer,
    Message,
    Qwen3Renderer,
    RenderContext,
    TextPart,
    ThinkingPart,
    ToolCall,
    get_renderer,
    parse_content_blocks,
)
from tinker_cookbook.tokenizer_utils import Tokenizer


# TEMPORARY: get_tokenizer uses a mirror for Llama 3 to avoid needing the HF_TOKEN,
# however, the mirrored tokenizer does not include the chat template.
# Remove this once the mirrored tokenizer (thinkingmachineslabinc/meta-llama-3-tokenizer)
# includes the chat template.
@cache
def get_tokenizer(model_name: str) -> Tokenizer:
    """Get tokenizer with chat template."""
    kwargs: dict[str, Any] = {}
    if model_name == "moonshotai/Kimi-K2-Thinking":
        kwargs["trust_remote_code"] = True
        kwargs["revision"] = "612681931a8c906ddb349f8ad0f582cb552189cd"

    return AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)


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


def get_tool_call_conversation() -> list[Message]:
    """Full tool use conversation with tool call and response.

    Includes: user request -> assistant tool call -> tool response -> assistant final answer.
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
        },
        {"role": "assistant", "content": "The weather in San Francisco is sunny with 72°F."},
    ]


# Conversation registry for parametrized tests
# Maps conversation ID to (factory_function, description, requires_system)
CONVERSATION_REGISTRY: dict[str, tuple[Callable[[], list[Message]], str, bool]] = {
    "basic_3turn": (get_basic_3turn_conversation, "basic 3-turn conversation", False),
    "basic_2turn": (get_basic_2turn_conversation, "basic 2-turn conversation", False),
    "system_3turn": (get_system_message_3turn_conversation, "3-turn with system message", True),
    "system_2turn": (get_system_message_2turn_conversation, "2-turn with system message", True),
}


# Models that support tool calling in their renderers
TOOL_CAPABLE_MODELS = {
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "deepseek-ai/DeepSeek-V3.1",
    "moonshotai/Kimi-K2-Thinking",
}


# =============================================================================
# HF Compatibility Tests (parametrized by model and conversation)
# =============================================================================


def _prepare_conversation_for_model(
    model_name: str, convo: list[Message], is_generation: bool
) -> tuple[list[Message], list[Message]]:
    """Prepare conversation for both cookbook renderer and HF template comparison.

    Unfortunately, aug_convo and hf_convo sometimes need to differ because HF templates
    auto-add certain content that our renderers expect to be present in the input:

    - Qwen (supervised): HF auto-adds "<think>\\n\\n</think>\\n\\n" to assistant content,
      but Qwen3Renderer expects thinking tags to already be in the input. So we must
      manually add them to aug_convo while leaving hf_convo without them.

    - Llama: HF auto-adds "Cutting Knowledge Date" content to the system message, so we add
      it to aug_convo but skip the system message in hf_convo to avoid duplication.

    - OpenAI/DeepSeek/Kimi: No modifications needed.

    Args:
        model_name: The model name (e.g., "Qwen/Qwen3-30B-A3B").
        convo: The base conversation to prepare.
        is_generation: True for generation prompts, False for supervised examples.

    Note: HF templates only use certain Message fields (role, content). Fields like
    tool_calls are not handled by HF templates in the same way, so tool calling tests
    use separate comparison methods.

    Returns:
        Tuple of (aug_convo, hf_convo):
        - aug_convo: Augmented conversation for the cookbook renderer. May include
          model-specific modifications like system messages, thinking tags, etc.
          that the renderer expects in its input.
        - hf_convo: Conversation for HF's apply_chat_template. Generally simpler
          since HF templates auto-add things like thinking blocks.
    """
    # Deep copy to avoid mutating the original

    # Check if conversation already has a system message
    has_system = convo and convo[0]["role"] == "system"

    # Apply model-specific modifications first, then create hf_convo
    if model_name.startswith("meta"):
        # HACK: HF template auto-prepends "Cutting Knowledge Date" to system messages.
        # Our Llama3Renderer doesn't do this, so we manually add it to aug_convo
        # to match what HF produces.
        today = date.today().strftime("%d %b %Y")
        date_prefix = f"Cutting Knowledge Date: December 2023\nToday Date: {today}\n\n"
        if has_system:
            aug_convo = copy.deepcopy(convo)
            assert isinstance(convo[0]["content"], str)
            aug_convo[0]["content"] = date_prefix + convo[0]["content"]
        else:
            aug_convo = [Message(role="system", content=date_prefix)] + convo
    elif model_name.startswith("Qwen"):
        if not is_generation:
            # This is a hack needed for test_supervised_example_against_hf_chat_templates.
            # The Qwen3Renderer does NOT auto-add empty thinking blocks like HF's apply_chat_template does.
            # Instead, Qwen3Renderer only adds "<think>\n" to the prefix for generation prompting
            # (to prompt the model to start reasoning). We probably want to make the SFT behavior
            # match the generation behavior, but that's for a future PR. (TODO)
            for i in range(len(convo) - 1, -1, -1):
                if convo[i]["role"] == "assistant":
                    content = convo[i]["content"]
                    if "<think>" not in content:
                        convo[i]["content"] = f"<think>\n\n</think>\n\n{content}"
                    break
        aug_convo = convo
    elif model_name.startswith("deepseek-ai"):
        aug_convo = convo
    elif model_name.startswith("openai"):
        aug_convo = convo
    elif model_name.startswith("moonshotai"):
        aug_convo = convo
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return aug_convo, convo


# Test matrix: models x conversations for generation tests
_GENERATION_TEST_PARAMS = [
    (model, conv_id)
    for model in [
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen3-30B-A3B",
        "deepseek-ai/DeepSeek-V3.1",
        "openai/gpt-oss-20b",
        "moonshotai/Kimi-K2-Thinking",
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
    ]
    for conv_id in ["basic_3turn", "system_3turn"]
]


@pytest.mark.parametrize("model_name,conv_id", _GENERATION_TEST_PARAMS)
def test_generation_against_hf_chat_templates(model_name: str, conv_id: str):
    """Test generation prompt against HF chat templates.

    Parametrized by model and conversation type. Tests that our renderer produces
    identical tokens to HuggingFace's chat template for the same conversation.
    """
    conv_factory, conv_desc, requires_tools = CONVERSATION_REGISTRY[conv_id]
    convo = conv_factory()

    tokenizer = get_tokenizer(model_name)
    attributes = get_model_attributes(model_name)
    image_processor = get_image_processor(model_name) if attributes.is_vl else None
    render_name = (
        get_recommended_renderer_name(model_name)
        if not model_name.startswith("openai")
        else "gpt_oss_medium_reasoning"
    )
    cookbook_renderer = get_renderer(render_name, tokenizer, image_processor)

    aug_convo, hf_convo = _prepare_conversation_for_model(model_name, convo, is_generation=True)

    cookbook_tokens = cookbook_renderer.build_generation_prompt(aug_convo).to_ints()
    hf_tokens = tokenizer.apply_chat_template(
        cast(list[dict[str, str]], hf_convo), add_generation_prompt=True, tokenize=True
    )

    assert cookbook_tokens == hf_tokens, (
        f"[{conv_desc}] Cookbook tokens: {cookbook_tokens}\n"
        f"Cookbook string: {tokenizer.decode(cookbook_tokens)}\n"
        f"HF tokens: {hf_tokens}\n"
        f"HF string: {tokenizer.decode(hf_tokens)}"
    )


# Test matrix: models x conversations for supervised tests
# Note: OpenAI excluded because we intentionally include empty analysis channel for train-test
# consistency, which diverges from HF template (HF only adds analysis channel during generation)
_SUPERVISED_TEST_PARAMS = [
    (model, conv_id)
    for model in [
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen3-30B-A3B",
        "deepseek-ai/DeepSeek-V3.1",
        "moonshotai/Kimi-K2-Thinking",
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
    ]
    for conv_id in ["basic_2turn", "system_2turn"]
]


@pytest.mark.parametrize("model_name,conv_id", _SUPERVISED_TEST_PARAMS)
def test_supervised_example_against_hf_chat_templates(model_name: str, conv_id: str):
    """Test supervised example against HF chat templates.

    Parametrized by model and conversation type. Tests that our renderer produces
    identical tokens to HuggingFace's chat template for the same conversation.
    """
    conv_factory, conv_desc, requires_tools = CONVERSATION_REGISTRY[conv_id]
    convo = conv_factory()

    tokenizer = get_tokenizer(model_name)
    attributes = get_model_attributes(model_name)
    image_processor = get_image_processor(model_name) if attributes.is_vl else None
    render_name = (
        get_recommended_renderer_name(model_name)
        if not model_name.startswith("openai")
        else "gpt_oss_medium_reasoning"
    )
    cookbook_renderer = get_renderer(render_name, tokenizer, image_processor)

    aug_convo, hf_convo = _prepare_conversation_for_model(model_name, convo, is_generation=False)

    cookbook_model_input, _ = cookbook_renderer.build_supervised_example(aug_convo)
    cookbook_tokens = cookbook_model_input.to_ints()
    hf_output = tokenizer.apply_chat_template(
        cast(list[dict[str, str]], hf_convo), tokenize=False, add_generation_prompt=False
    )
    assert isinstance(hf_output, str)
    hf_tokens = tokenizer.encode(hf_output.rstrip("\n"), add_special_tokens=False)

    assert cookbook_tokens == hf_tokens, (
        f"[{conv_desc}] Cookbook tokens: {cookbook_tokens}\n"
        f"Cookbook string: {tokenizer.decode(cookbook_tokens)}\n"
        f"HF tokens: {hf_tokens}\n"
        f"HF string: {tokenizer.decode(hf_tokens)}"
    )


# =============================================================================
# Tool Use Rendering Tests
# =============================================================================


@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen3-30B-A3B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "deepseek-ai/DeepSeek-V3.1",
        "moonshotai/Kimi-K2-Thinking",
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
    render_name = get_recommended_renderer_name(model_name)
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
    # Check for tool arguments which all formats include
    assert "San Francisco" in decoded, f"Tool argument should appear in rendered output: {decoded}"

    # Check for either the function name or the tool_call_id
    has_tool_indicator = "get_weather" in decoded or "call_123" in decoded
    assert has_tool_indicator, f"Tool name or ID should appear in rendered output: {decoded}"


# Models where our tool call rendering matches HF templates exactly
_TOOL_CALL_HF_COMPATIBLE_MODELS = [
    ("Qwen/Qwen3-8B", "qwen3"),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", "qwen3_instruct"),
    # deepseekv3 defaults to non-thinking mode (matches HF template)
    ("deepseek-ai/DeepSeek-V3.1", "deepseekv3"),
    ("moonshotai/Kimi-K2-Thinking", "kimi_k2"),
]


def _convert_tool_calls_to_hf_format(convo: list[Message]) -> list[dict]:
    """Convert ToolCall objects to HF dict format for template comparison."""
    result = []
    for msg in convo:
        msg_dict: dict = {"role": msg["role"], "content": msg["content"]}
        if "tool_calls" in msg:
            msg_dict["tool_calls"] = [
                {
                    "type": "function",
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg["tool_calls"]
            ]
        if "tool_call_id" in msg:
            msg_dict["tool_call_id"] = msg["tool_call_id"]
        result.append(msg_dict)
    return result


@pytest.mark.parametrize("model_name,renderer_name", _TOOL_CALL_HF_COMPATIBLE_MODELS)
def test_tool_call_generation_against_hf_templates(model_name: str, renderer_name: str):
    """Test tool call generation rendering matches HF templates.

    For models with HF-compatible tool call support, verify our renderer produces
    identical tokens to HuggingFace's chat template.
    """
    # Use first 3 messages (without final assistant response) for generation prompt
    convo_ours = get_tool_call_conversation()[:-1]
    convo_hf = _convert_tool_calls_to_hf_format(convo_ours)

    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    ours = renderer.build_generation_prompt(convo_ours).to_ints()
    hf = tokenizer.apply_chat_template(convo_hf, add_generation_prompt=True, tokenize=True)

    assert ours == hf, (
        f"Tool call rendering mismatch for {model_name}\n"
        f"Ours: {tokenizer.decode(ours)}\n"
        f"HF:   {tokenizer.decode(hf)}"
    )


# =============================================================================
# Qwen3 Thinking Tests (multi-turn with thinking content)
# =============================================================================


def test_qwen3_2turn_preserves_thinking():
    """
    For 2-turn conversations (user + assistant), thinking should be fully preserved
    since the assistant message is the last message.
    """
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    renderer = Qwen3Renderer(tokenizer)

    messages: list[Message] = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "<think>\nLet me calculate this.\n</think>\n\nThe answer is 4.",
        },
    ]

    model_input, _ = renderer.build_supervised_example(messages)
    tinker_tokens = model_input.to_ints()
    tinker_decoded = tokenizer.decode(tinker_tokens)

    # Get expected format from HuggingFace tokenizer
    hf_decoded = tokenizer.apply_chat_template(cast(list[dict[str, str]], messages), tokenize=False)

    # Tinker and HuggingFace should produce the same output (strip trailing newline from HF)
    assert tinker_decoded == hf_decoded.rstrip("\n"), (
        f"Tinker and HuggingFace outputs differ:\n"
        f"TINKER:\n{tinker_decoded!r}\n\n"
        f"HUGGINGFACE:\n{hf_decoded!r}"
    )


def test_qwen3_4turn_only_last_thinking_preserved():
    """
    For 4-turn conversations with ThinkingPart, only the last assistant message's thinking
    should be preserved. Earlier assistant thinking blocks are stripped.
    """
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    renderer = Qwen3Renderer(tokenizer)

    # Use structured content with ThinkingPart
    messages: list[Message] = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="\nFirst turn reasoning here.\n"),
                TextPart(type="text", text="\n\nThe answer is 4."),
            ],
        },
        {"role": "user", "content": "And what is 3+3?"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="\nSecond turn reasoning here.\n"),
                TextPart(type="text", text="\n\nThe answer is 6."),
            ],
        },
    ]

    model_input, _ = renderer.build_supervised_example(messages)
    tinker_decoded = tokenizer.decode(model_input.to_ints())

    # First assistant message should have thinking stripped
    assert "First turn reasoning" not in tinker_decoded, (
        f"First turn thinking should be stripped:\n{tinker_decoded}"
    )
    # Second (last) assistant message should preserve thinking
    assert "Second turn reasoning" in tinker_decoded, (
        f"Last turn thinking should be preserved:\n{tinker_decoded}"
    )


def test_qwen3_generation_matches_hf():
    """Test Qwen3Renderer generation prompt matches HF with enable_thinking=True (default)."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    cookbook_renderer = get_renderer("qwen3", tokenizer)

    convo: list[Message] = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    cookbook_tokens = cookbook_renderer.build_generation_prompt(convo).to_ints()
    hf_tokens = tokenizer.apply_chat_template(
        cast(list[dict[str, str]], convo),
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=True,  # Explicit, though this is the default
    )

    assert cookbook_tokens == hf_tokens, (
        f"Cookbook tokens: {cookbook_tokens}\n"
        f"Cookbook string: {tokenizer.decode(cookbook_tokens)}\n"
        f"HF tokens: {hf_tokens}\n"
        f"HF string: {tokenizer.decode(hf_tokens)}"
    )


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

    messages: list[Message] = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
    ]

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

    convo: list[Message] = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    cookbook_tokens = cookbook_renderer.build_generation_prompt(convo).to_ints()
    hf_tokens = tokenizer.apply_chat_template(
        cast(list[dict[str, str]], convo),
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )

    assert cookbook_tokens == hf_tokens, (
        f"Cookbook tokens: {cookbook_tokens}\n"
        f"Cookbook string: {tokenizer.decode(cookbook_tokens)}\n"
        f"HF tokens: {hf_tokens}\n"
        f"HF string: {tokenizer.decode(hf_tokens)}"
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

    messages: list[Message] = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "user", "content": "And what is 3+3?"},
        {"role": "assistant", "content": "The answer is 6."},
    ]

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

    with pytest.raises(ValueError, match="expected to split into 1 or 2 pieces"):
        _ = renderer.parse_response(response_tokens_double_eot)


# =============================================================================
# strip_thinking_from_history=False Tests (Extension Property)
# =============================================================================


def test_qwen3_strip_thinking_false_preserves_all():
    """
    Test that strip_thinking_from_history=False preserves thinking in ALL messages.
    This mode is used for multi-turn RL where the extension property is needed.
    Note: This mode does NOT match HF behavior - it's a special mode for efficiency.
    """
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    renderer = Qwen3Renderer(tokenizer, strip_thinking_from_history=False)

    messages: list[Message] = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "<think>\nFirst calculation.\n</think>\n\nThe answer is 4.",
        },
        {"role": "user", "content": "And what is 3+3?"},
        {
            "role": "assistant",
            "content": "<think>\nSecond calculation.\n</think>\n\nThe answer is 6.",
        },
    ]

    model_input, _ = renderer.build_supervised_example(messages)
    decoded = tokenizer.decode(model_input.to_ints())

    # Both thinking blocks should be present
    assert decoded.count("<think>") == 2, (
        f"Expected 2 thinking blocks with strip_thinking_from_history=False, got: {decoded}"
    )
    assert "First calculation" in decoded
    assert "Second calculation" in decoded


# =============================================================================
# DeepSeek Thinking Trace Tests
# =============================================================================


def test_deepseek_strip_thinking_from_history_default():
    """
    Test that DeepSeek strips thinking from historical assistant messages by default.
    Only the last assistant message should preserve thinking.
    """
    model_name = "deepseek-ai/DeepSeek-V3.1"
    tokenizer = get_tokenizer(model_name)
    renderer = DeepSeekV3ThinkingRenderer(tokenizer)

    # Use structured content with ThinkingPart
    messages: list[Message] = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="First calculation."),
                TextPart(type="text", text="The answer is 4."),
            ],
        },
        {"role": "user", "content": "And what is 3+3?"},
        {
            "role": "assistant",
            "content": [
                ThinkingPart(type="thinking", thinking="Second calculation."),
                TextPart(type="text", text="The answer is 6."),
            ],
        },
    ]

    model_input, _ = renderer.build_supervised_example(messages)
    decoded = tokenizer.decode(model_input.to_ints())

    # First thinking block should be stripped, second preserved
    assert "First calculation" not in decoded, (
        f"Historical thinking should be stripped by default: {decoded}"
    )
    assert "Second calculation" in decoded, (
        f"Last assistant thinking should be preserved: {decoded}"
    )


def test_deepseek_strip_thinking_false_preserves_all():
    """
    Test that strip_thinking_from_history=False preserves thinking in ALL messages.
    This mode is used for multi-turn RL where the extension property is needed.
    """
    model_name = "deepseek-ai/DeepSeek-V3.1"
    tokenizer = get_tokenizer(model_name)
    renderer = DeepSeekV3ThinkingRenderer(tokenizer, strip_thinking_from_history=False)

    messages: list[Message] = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "<think>First calculation.</think>The answer is 4.",
        },
        {"role": "user", "content": "And what is 3+3?"},
        {
            "role": "assistant",
            "content": "<think>Second calculation.</think>The answer is 6.",
        },
    ]

    model_input, _ = renderer.build_supervised_example(messages)
    decoded = tokenizer.decode(model_input.to_ints())

    # Both thinking blocks should be present
    assert "First calculation" in decoded, (
        f"First thinking should be preserved with strip_thinking_from_history=False: {decoded}"
    )
    assert "Second calculation" in decoded, f"Second thinking should be preserved: {decoded}"


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
# parse_content_blocks Tests
# =============================================================================


def test_parse_content_blocks_no_special_tags():
    """Test parse_content_blocks returns None when no special tags."""
    parts = parse_content_blocks("Just plain text")
    assert parts is None


def test_parse_content_blocks_single_think_block():
    """Test parse_content_blocks with a single think block."""
    parts = parse_content_blocks("<think>reasoning</think>visible answer")
    assert parts is not None

    assert len(parts) == 2
    assert parts[0]["type"] == "thinking"
    assert parts[0]["thinking"] == "reasoning"  # type: ignore[typeddict-item]
    assert parts[1]["type"] == "text"
    assert parts[1]["text"] == "visible answer"  # type: ignore[typeddict-item]


def test_parse_content_blocks_multiple_think_blocks():
    """Test parse_content_blocks with multiple interleaved think blocks."""
    content = "<think>step 1</think>partial<think>step 2</think>final"
    parts = parse_content_blocks(content)
    assert parts is not None

    assert len(parts) == 4
    assert parts[0] == ThinkingPart(type="thinking", thinking="step 1")
    assert parts[1] == TextPart(type="text", text="partial")
    assert parts[2] == ThinkingPart(type="thinking", thinking="step 2")
    assert parts[3] == TextPart(type="text", text="final")


def test_parse_content_blocks_empty_blocks_omitted():
    """Test parse_content_blocks omits empty think blocks."""
    parts = parse_content_blocks("<think></think>visible")
    assert parts is not None

    assert len(parts) == 1
    assert parts[0]["type"] == "text"
    assert parts[0]["text"] == "visible"  # type: ignore[typeddict-item]


def test_parse_content_blocks_whitespace_handling():
    """Test parse_content_blocks preserves whitespace for identity roundtrip."""
    parts = parse_content_blocks("<think>  thinking  </think>  answer  ")
    assert parts is not None

    assert len(parts) == 2
    # Whitespace is preserved exactly for identity roundtrip
    assert parts[0]["type"] == "thinking" and parts[0]["thinking"] == "  thinking  "  # type: ignore[typeddict-item]
    assert parts[1]["type"] == "text" and parts[1]["text"] == "  answer  "  # type: ignore[typeddict-item]


def test_parse_content_blocks_tool_call_only():
    """Test parse_content_blocks parses tool calls."""
    content = '<tool_call>{"name": "search", "arguments": {"query": "test"}}</tool_call>'
    parts = parse_content_blocks(content)
    assert parts is not None

    assert len(parts) == 1
    assert parts[0]["type"] == "tool_call"
    tool_call = parts[0]["tool_call"]  # type: ignore[typeddict-item]
    assert tool_call.function.name == "search"
    assert tool_call.function.arguments == '{"query": "test"}'


def test_parse_content_blocks_interleaved():
    """Test parse_content_blocks handles interleaved think and tool_call blocks."""
    content = '<think>Let me search</think>Searching...<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>Done'
    parts = parse_content_blocks(content)
    assert parts is not None

    assert len(parts) == 4
    assert parts[0] == ThinkingPart(type="thinking", thinking="Let me search")
    assert parts[1] == TextPart(type="text", text="Searching...")
    assert parts[2]["type"] == "tool_call"
    assert parts[2]["tool_call"].function.name == "search"  # type: ignore[typeddict-item]
    assert parts[3] == TextPart(type="text", text="Done")


def test_parse_content_blocks_invalid_tool_call():
    """Test parse_content_blocks handles invalid tool call JSON as UnparsedToolCallPart."""
    content = "<tool_call>not valid json</tool_call>text after"
    parts = parse_content_blocks(content)
    assert parts is not None

    # Invalid tool call is included as UnparsedToolCallPart, text is still captured
    assert len(parts) == 2
    assert parts[0]["type"] == "unparsed_tool_call"
    assert "Invalid JSON" in parts[0]["error"]  # type: ignore[typeddict-item]
    assert parts[1] == TextPart(type="text", text="text after")


# =============================================================================
# Qwen3 parse_response Tests
# =============================================================================


def test_qwen3_parse_response_extracts_thinking():
    """Test Qwen3Renderer.parse_response extracts thinking to ThinkingPart."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    # Simulate a response with thinking
    response_str = "<think>Let me reason about this.</think>The answer is 42.<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    assert message["role"] == "assistant"

    # Content should be a list with ThinkingPart and TextPart
    content = message["content"]
    assert isinstance(content, list)

    thinking_parts = [p for p in content if p["type"] == "thinking"]
    text_parts = [p for p in content if p["type"] == "text"]

    assert len(thinking_parts) == 1
    assert thinking_parts[0]["thinking"] == "Let me reason about this."

    assert len(text_parts) == 1
    assert text_parts[0]["text"] == "The answer is 42."


def test_qwen3_parse_response_multiple_think_blocks():
    """Test Qwen3Renderer.parse_response handles multiple interleaved think blocks."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = "<think>step 1</think>partial answer<think>step 2</think>final answer<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    content = message["content"]
    assert isinstance(content, list)
    assert len(content) == 4

    assert content[0] == ThinkingPart(type="thinking", thinking="step 1")
    assert content[1] == TextPart(type="text", text="partial answer")
    assert content[2] == ThinkingPart(type="thinking", thinking="step 2")
    assert content[3] == TextPart(type="text", text="final answer")


def test_qwen3_parse_response_no_thinking_returns_string():
    """Test Qwen3Renderer.parse_response returns string when no thinking."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = "Just a plain response without thinking.<|im_end|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    # Content should remain a string for backward compatibility
    assert isinstance(message["content"], str)
    assert message["content"] == "Just a plain response without thinking."


def test_qwen3_parse_response_with_tool_calls():
    """Test Qwen3Renderer.parse_response parses tool calls into ToolCallPart."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = '<think>Let me search</think>I will search for that.<tool_call>{"name": "web_search", "arguments": {"query": "weather"}}</tool_call><|im_end|>'
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    content = message["content"]
    assert isinstance(content, list)

    # Should have ThinkingPart, TextPart, ToolCallPart in order
    assert len(content) == 3
    assert content[0]["type"] == "thinking"
    assert content[0]["thinking"] == "Let me search"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "I will search for that."
    assert content[2]["type"] == "tool_call"
    assert content[2]["tool_call"].function.name == "web_search"

    # Also check backward-compatible tool_calls field
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "web_search"


def test_qwen3_parse_response_tool_call_only():
    """Test Qwen3Renderer.parse_response with only a tool call."""
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = Qwen3Renderer(tokenizer)

    response_str = (
        '<tool_call>{"name": "calculator", "arguments": {"expr": "2+2"}}</tool_call><|im_end|>'
    )
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    content = message["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]["type"] == "tool_call"

    # Backward-compatible field
    assert "tool_calls" in message and len(message["tool_calls"]) == 1


# =============================================================================
# DeepSeek parse_response Tests
# =============================================================================


def test_deepseek_parse_response_extracts_thinking():
    """Test DeepSeekV3ThinkingRenderer.parse_response extracts thinking."""
    tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3.1")
    renderer = DeepSeekV3ThinkingRenderer(tokenizer)

    # Note: DeepSeek uses full-width pipes in special tokens
    response_str = "<think>Let me think about this.</think>The answer is 42.<｜end▁of▁sentence｜>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    content = message["content"]
    assert isinstance(content, list)

    thinking_parts = [p for p in content if p["type"] == "thinking"]
    text_parts = [p for p in content if p["type"] == "text"]

    assert len(thinking_parts) == 1
    assert thinking_parts[0]["thinking"] == "Let me think about this."
    assert len(text_parts) == 1
    assert text_parts[0]["text"] == "The answer is 42."


def test_deepseek_parse_response_no_thinking_returns_string():
    """Test DeepSeekV3ThinkingRenderer.parse_response returns string when no thinking."""
    tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3.1")
    renderer = DeepSeekV3ThinkingRenderer(tokenizer)

    response_str = "Just a plain response.<｜end▁of▁sentence｜>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    assert isinstance(message["content"], str)
    assert message["content"] == "Just a plain response."


def test_deepseek_parse_response_multiple_think_blocks():
    """Test DeepSeekV3ThinkingRenderer.parse_response handles multiple think blocks."""
    tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3.1")
    renderer = DeepSeekV3ThinkingRenderer(tokenizer)

    response_str = "<think>step 1</think>partial<think>step 2</think>final<｜end▁of▁sentence｜>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    content = message["content"]
    assert isinstance(content, list)
    assert len(content) == 4

    assert content[0] == ThinkingPart(type="thinking", thinking="step 1")
    assert content[1] == TextPart(type="text", text="partial")
    assert content[2] == ThinkingPart(type="thinking", thinking="step 2")
    assert content[3] == TextPart(type="text", text="final")


# =============================================================================
# GptOss parse_response Tests
# =============================================================================


def test_gptoss_parse_response_extracts_thinking():
    """Test GptOssRenderer.parse_response extracts analysis channel as thinking."""
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")

    # GptOss format: analysis channel then final channel
    response_str = "<|channel|>analysis<|message|>Let me think about this.<|end|><|start|>assistant<|channel|>final<|message|>The answer is 42.<|return|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    content = message["content"]
    assert isinstance(content, list)

    thinking_parts = [p for p in content if p["type"] == "thinking"]
    text_parts = [p for p in content if p["type"] == "text"]

    assert len(thinking_parts) == 1
    assert thinking_parts[0]["thinking"] == "Let me think about this."
    assert len(text_parts) == 1
    assert text_parts[0]["text"] == "The answer is 42."


def test_gptoss_parse_response_multiple_analysis():
    """Test GptOssRenderer.parse_response handles multiple analysis messages."""
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")

    # Multiple analysis channels (interleaved thinking)
    response_str = "<|channel|>analysis<|message|>First thought.<|end|><|start|>assistant<|channel|>analysis<|message|>Second thought.<|end|><|start|>assistant<|channel|>final<|message|>Done.<|return|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    content = message["content"]
    assert isinstance(content, list)
    assert len(content) == 3

    assert content[0] == ThinkingPart(type="thinking", thinking="First thought.")
    assert content[1] == ThinkingPart(type="thinking", thinking="Second thought.")
    assert content[2] == TextPart(type="text", text="Done.")


def test_gptoss_parse_response_final_only():
    """Test GptOssRenderer.parse_response with only final channel."""
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")

    response_str = "<|channel|>final<|message|>Simple answer.<|return|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    content = message["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0] == TextPart(type="text", text="Simple answer.")


def test_gptoss_parse_response_no_channels():
    """Test GptOssRenderer.parse_response returns string when no channel markers."""
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")

    response_str = "Plain response without channels.<|return|>"
    response_tokens = tokenizer.encode(response_str, add_special_tokens=False)

    message, success = renderer.parse_response(response_tokens)

    assert success
    # No channel markers, so content stays as string
    assert isinstance(message["content"], str)
    assert message["content"] == "Plain response without channels."
