"""
Tests for tool calling support in renderers.

These tests verify that renderers correctly handle:
1. Tool response message rendering (role mapping and content wrapping)
2. Parsing of single and multiple tool calls from model output
3. Stripping tool call blocks from parsed message content
4. Extracting function names from model-specific tool call formats
"""

import pytest
import tinker

from tinker_cookbook.renderers import (
    Message,
    RenderContext,
    ToolCall,
    get_renderer,
    get_text_content,
)
from tinker_cookbook.renderers.testing_utils import (
    extract_token_ids,
    skip_deepseek_tokenizer_bug,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

# =============================================================================
# Tool Response Rendering Tests
# =============================================================================


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [
        ("Qwen/Qwen3-8B", "qwen3"),
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", "qwen3_instruct"),
        ("Qwen/Qwen3.6-35B-A3B", "qwen3_5"),
        ("Qwen/Qwen3.6-35B-A3B", "qwen3_5_disable_thinking"),
        ("Qwen/Qwen3.8-27B", "qwen3_8_xhigh_reasoning"),
        ("Qwen/Qwen3.8-27B", "qwen3_8_disable_thinking"),
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron3"),
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron3_disable_thinking"),
    ],
)
def test_qwen3_tool_response_rendering(model_name: str, renderer_name: str):
    """Test that Qwen3 renders tool responses with user role and tool_response tags.

    Per the Qwen3 chat template, tool messages should render as
    <|im_start|>user with content wrapped in <tool_response> tags.
    """
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    tool_message: Message = {"role": "tool", "content": '{"weather": "sunny", "high": 72}'}

    ctx = RenderContext(idx=0, is_last=False, prev_message=None)
    rendered = renderer.render_message(tool_message, ctx)
    header = rendered.header
    assert header is not None, "Expected header in rendered message"
    output = rendered.output
    assert len(output) > 0, "Expected output in rendered message"

    header_str = tokenizer.decode(list(header.tokens))
    # output[0] is an EncodedTextChunk for text-only messages
    output_chunk = output[0]
    assert isinstance(output_chunk, tinker.EncodedTextChunk), "Expected EncodedTextChunk"
    output_str = tokenizer.decode(list(output_chunk.tokens))

    # Tool messages should be rendered as "user" role
    assert "<|im_start|>user" in header_str
    # Content should be wrapped in tool_response tags
    assert "<tool_response>" in output_str
    assert "</tool_response>" in output_str
    assert '"weather": "sunny"' in output_str


# =============================================================================
# Tool Call Parsing Tests
# =============================================================================


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [
        ("Qwen/Qwen3-8B", "qwen3"),
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", "qwen3_instruct"),
        ("Qwen/Qwen3.6-35B-A3B", "qwen3_5"),
        ("Qwen/Qwen3.6-35B-A3B", "qwen3_5_disable_thinking"),
        ("Qwen/Qwen3.8-27B", "qwen3_8_xhigh_reasoning"),
        ("Qwen/Qwen3.8-27B", "qwen3_8_disable_thinking"),
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron3"),
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron3_disable_thinking"),
    ],
)
def test_qwen3_parse_single_tool_call(model_name: str, renderer_name: str):
    """Test parsing a single tool call from Qwen3 response."""
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # Simulate model response with tool call
    response_text = """I'll search for that information.
<tool_call>
{"name": "search", "arguments": {"query": "weather in NYC"}}
</tool_call><|im_end|>"""
    if renderer_name.startswith(("qwen3_5", "qwen3_8", "nemotron3")):
        response_text = """I'll search for that information.
<tool_call>
<function=search>
<parameter=query>
weather in NYC
</parameter>
</function>
</tool_call><|im_end|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success.is_clean
    assert message["role"] == "assistant"
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "search"
    # Content should have tool_call block stripped (text content only)
    text_content = get_text_content(message)
    assert "<tool_call>" not in text_content


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [
        ("Qwen/Qwen3-8B", "qwen3"),
        ("Qwen/Qwen3.6-35B-A3B", "qwen3_5"),
        ("Qwen/Qwen3.8-27B", "qwen3_8_xhigh_reasoning"),
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron3"),
    ],
)
def test_qwen3_parse_multiple_tool_calls(model_name: str, renderer_name: str):
    """Test parsing multiple tool calls from Qwen3 response.

    When a model response contains multiple <tool_call> blocks, all should be parsed.
    """
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # Simulate model response with multiple tool calls
    response_text = """I'll get the weather for both cities.
<tool_call>
{"name": "get_weather", "arguments": {"location": "NYC"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"location": "LA"}}
</tool_call><|im_end|>"""
    if renderer_name in ("qwen3_5", "qwen3_8_xhigh_reasoning", "nemotron3"):
        response_text = """I'll get the weather for both cities.
<tool_call>
<function=get_weather>
<parameter=location>
NYC
</parameter>
</function>
</tool_call>
<tool_call>
<function=get_weather>
<parameter=location>
LA
</parameter>
</function>
</tool_call><|im_end|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success.is_clean
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 2
    assert message["tool_calls"][0].function.name == "get_weather"
    assert message["tool_calls"][1].function.name == "get_weather"
    # Verify different arguments
    assert "NYC" in message["tool_calls"][0].function.arguments
    assert "LA" in message["tool_calls"][1].function.arguments


def test_kimi_k2_parse_tool_call():
    """Test parsing tool call from Kimi K2 response.

    Kimi K2 uses tool_id format "functions.{name}:{idx}", and the function
    name should be extracted correctly.
    """
    model_name = "moonshotai/Kimi-K2-Thinking"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("kimi_k2", tokenizer)

    # Simulate model response with tool call (Kimi K2 format)
    response_text = """<think></think>I'll search for that.
<|tool_calls_section_begin|><|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{"query": "weather NYC"}<|tool_call_end|><|tool_calls_section_end|><|im_end|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success.is_clean
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    # Verify function name is extracted from tool_id
    assert message["tool_calls"][0].function.name == "search"
    assert message["tool_calls"][0].id == "functions.search:0"


@skip_deepseek_tokenizer_bug
def test_deepseek_parse_tool_call():
    """Test parsing tool call from DeepSeek V3 response.

    DeepSeek V3 HF template format: <｜tool▁call▁begin｜>name<｜tool▁sep｜>args<｜tool▁call▁end｜>
    """
    model_name = "deepseek-ai/DeepSeek-V3.1"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("deepseekv3", tokenizer)

    response_text = """I'll check the weather.
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "NYC"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success.is_clean
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "get_weather"
    assert "NYC" in message["tool_calls"][0].function.arguments
    # Content should have tool calls section stripped (text content only)
    text_content = get_text_content(message)
    assert "<｜tool▁calls▁begin｜>" not in text_content


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


def test_qwen3_parse_invalid_tool_call_json():
    """Test that invalid JSON in tool call is captured as unparsed_tool_calls."""
    model_name = "Qwen/Qwen3-8B"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("qwen3", tokenizer)

    # Invalid JSON in tool call
    response_text = """<tool_call>
{invalid json here}
</tool_call><|im_end|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    # Parse succeeds, but tool call is captured as unparsed
    assert success.is_clean
    assert "tool_calls" not in message or len(message.get("tool_calls", [])) == 0
    assert "unparsed_tool_calls" in message
    assert len(message["unparsed_tool_calls"]) == 1
    assert "Invalid JSON" in message["unparsed_tool_calls"][0].error
    # Raw text should contain the original tool call
    assert "<tool_call>" in message["unparsed_tool_calls"][0].raw_text


def test_qwen3_mixed_valid_invalid_tool_calls():
    """Test parsing when some tool calls are valid and some are invalid.

    Valid tool calls should be parsed, invalid ones captured in unparsed_tool_calls.
    """
    model_name = "Qwen/Qwen3-8B"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("qwen3", tokenizer)

    # First tool call is valid, second has invalid JSON
    response_text = """I'll try both.
<tool_call>
{"name": "search", "arguments": {"query": "weather"}}
</tool_call>
<tool_call>
{bad json here}
</tool_call><|im_end|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success.is_clean
    # Valid tool call should be parsed
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "search"
    # Invalid tool call should be in unparsed_tool_calls
    assert "unparsed_tool_calls" in message
    assert len(message["unparsed_tool_calls"]) == 1
    assert "Invalid JSON" in message["unparsed_tool_calls"][0].error


@skip_deepseek_tokenizer_bug
def test_deepseek_parse_invalid_tool_call_json():
    """Test that invalid JSON in DeepSeek tool call is captured as unparsed."""
    model_name = "deepseek-ai/DeepSeek-V3.1"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("deepseekv3", tokenizer)

    response_text = """I'll check.
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{invalid json}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success.is_clean
    assert "tool_calls" not in message or len(message.get("tool_calls", [])) == 0
    assert "unparsed_tool_calls" in message
    assert len(message["unparsed_tool_calls"]) == 1
    assert "Invalid JSON" in message["unparsed_tool_calls"][0].error


def test_kimi_k2_parse_invalid_tool_call_json():
    """Test that invalid JSON in Kimi K2 tool call is captured as unparsed."""
    model_name = "moonshotai/Kimi-K2-Thinking"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer("kimi_k2", tokenizer)

    response_text = """<think></think>I'll search.
<|tool_calls_section_begin|><|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{invalid}<|tool_call_end|><|tool_calls_section_end|><|im_end|>"""

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, success = renderer.parse_response(response_tokens)

    assert success.is_clean
    assert "tool_calls" not in message or len(message.get("tool_calls", [])) == 0
    assert "unparsed_tool_calls" in message
    assert len(message["unparsed_tool_calls"]) == 1
    assert "Invalid JSON" in message["unparsed_tool_calls"][0].error


# =============================================================================
# HF-parity tests for tool shapes the shared fixtures don't cover
# =============================================================================

# Renderers whose HF templates gate the tool-call separator on content and wrap a
# run of consecutive tool responses in a single <|im_start|>user block.
_TOOL_BLOCK_PARITY_RENDERERS = [
    ("Qwen/Qwen3.5-4B", "qwen3_5"),
    ("Qwen/Qwen3.6-35B-A3B", "qwen3_5"),
    ("Qwen/Qwen3.8-27B", "qwen3_8_xhigh_reasoning"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron3"),
    ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "nemotron3_ultra"),
]


def _assert_generation_matches_hf(model_name: str, renderer_name: str, msgs: list[Message]):
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)
    ours = renderer.build_generation_prompt(msgs).to_ints()
    hf_tokens = extract_token_ids(
        tokenizer.apply_chat_template(
            [renderer.to_openai_message(m) for m in msgs],
            add_generation_prompt=True,
            tokenize=True,
        )
    )
    assert ours == hf_tokens, (
        f"[{renderer_name}]\nours: {tokenizer.decode(ours)!r}\nhf:   {tokenizer.decode(hf_tokens)!r}"
    )


@pytest.mark.parametrize("model_name,renderer_name", _TOOL_BLOCK_PARITY_RENDERERS)
def test_contentless_tool_call_matches_hf(model_name: str, renderer_name: str):
    """No separator before <tool_call> when the assistant message has no visible text.

    The templates gate the blank line on ``content|trim``; an OpenAI-style tool-call
    message with empty content must start its <tool_call> block immediately.
    """
    tc = ToolCall(function=ToolCall.FunctionBody(name="search", arguments='{"query": "x"}'))
    msgs: list[Message] = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [tc]},
        {"role": "tool", "content": "res"},
        {"role": "user", "content": "next"},
    ]
    _assert_generation_matches_hf(model_name, renderer_name, msgs)


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [
        ("Qwen/Qwen3.5-4B", "qwen3_5"),
        # Qwen3.6 changed scalar serialization from Qwen3.5's |string (True/None) to
        # |tojson (true/null) but reuses the qwen3_5 renderer, so boolean and null
        # arguments diverge from its template. Qwen3.8's renderer follows the new rule.
        pytest.param(
            "Qwen/Qwen3.6-35B-A3B",
            "qwen3_5",
            marks=pytest.mark.xfail(
                strict=True,
                reason="Qwen3.6's template tojson-serializes scalars; the shared "
                "qwen3_5 renderer keeps Qwen3.5's |string form (True/None)",
            ),
        ),
        ("Qwen/Qwen3.8-27B", "qwen3_8_xhigh_reasoning"),
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron3"),
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "nemotron3_ultra"),
    ],
)
def test_tool_argument_value_serialization_matches_hf(model_name: str, renderer_name: str):
    """Non-string argument values serialize the way each model's template does.

    Qwen3.5 and Nemotron templates apply |tojson only to mappings/sequences and
    |string to scalars (True/None); Qwen3.6/3.8 templates tojson everything
    non-string (true/null). Booleans and nulls are where the rules diverge.
    """
    tc = ToolCall(
        function=ToolCall.FunctionBody(
            name="cfg",
            arguments=(
                '{"flag": true, "opt": null, "num": 3, "ratio": 0.5,'
                ' "text": "s", "obj": {"b": 1, "a": 2}, "arr": [1, "x"]}'
            ),
        )
    )
    msgs: list[Message] = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "Setting up.", "tool_calls": [tc]},
        {"role": "tool", "content": "ok"},
        {"role": "user", "content": "next"},
    ]
    _assert_generation_matches_hf(model_name, renderer_name, msgs)


def test_grouped_tool_response_keeps_terminator_in_final_output():
    """The <|im_end|> closing a tool-response run belongs to the final response's
    OUTPUT (so e.g. ALL_MESSAGES weights it), not to the next message's header;
    mid-run responses emit none at all."""
    tokenizer = get_tokenizer("Qwen/Qwen3.6-35B-A3B")
    renderer = get_renderer("qwen3_5", tokenizer)
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    tool_msg: Message = {"role": "tool", "content": "r1"}
    next_tool: Message = {"role": "tool", "content": "r2"}
    next_user: Message = {"role": "user", "content": "go"}

    def output_tokens(ctx: RenderContext) -> list[int]:
        rendered = renderer.render_message(tool_msg, ctx)
        return [
            t
            for chunk in rendered.output
            if isinstance(chunk, tinker.EncodedTextChunk)
            for t in chunk.tokens
        ]

    mid_run = RenderContext(idx=2, is_last=False, prev_message=None, next_message=next_tool)
    run_final = RenderContext(idx=2, is_last=False, prev_message=None, next_message=next_user)
    assert im_end not in output_tokens(mid_run)
    assert im_end in output_tokens(run_final)


@pytest.mark.parametrize("model_name,renderer_name", _TOOL_BLOCK_PARITY_RENDERERS)
@pytest.mark.parametrize("ends_with_tools", [False, True])
def test_consecutive_tool_responses_match_hf(
    model_name: str, renderer_name: str, ends_with_tools: bool
):
    """Consecutive tool results share one <|im_start|>user block (parallel tool calls).

    The templates open the user block before the first result of a run and close it
    after the last one — whether the run is followed by another message or ends the
    conversation.
    """
    msgs: list[Message] = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "calling"},
        {"role": "tool", "content": "r1"},
        {"role": "tool", "content": "r2"},
    ]
    if not ends_with_tools:
        msgs.append({"role": "user", "content": "go on"})
    _assert_generation_matches_hf(model_name, renderer_name, msgs)
