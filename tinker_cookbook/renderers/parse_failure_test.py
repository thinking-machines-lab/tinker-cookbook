"""Tests for the structural-vs-content parse-failure contract.

Covers:

1. ``classify_parse_failure`` over the failure matrix (MALFORMED,
   unparsed-only, mixed valid+unparsed, clean).
2. ``detect_unterminated_tool_block`` marker counting.
3. Per-renderer detection of the blind spot: a tool block opened but never
   closed in a response that still terminates cleanly on the renderer's stop
   token. Without detection this silently degrades to plain text; with it,
   the response carries an ``unparsed_tool_calls`` entry (a recoverable
   CONTENT failure). ``role_colon`` and ``llama3`` have no tool-block syntax
   and are deliberately not covered.
"""

import pytest

from tinker_cookbook.renderers import Message, get_renderer
from tinker_cookbook.renderers.base import (
    PARSE_FAILURE_DETAIL_MAX_CHARS,
    UNTERMINATED_TOOL_BLOCK_ERROR,
    ParseFailureKind,
    ParseTermination,
    ToolCall,
    UnparsedToolCall,
    classify_parse_failure,
    detect_unterminated_tool_block,
)
from tinker_cookbook.renderers.testing_utils import skip_deepseek_tokenizer_bug
from tinker_cookbook.tokenizer_utils import get_tokenizer

# =============================================================================
# classify_parse_failure matrix
# =============================================================================


def _valid_tool_call() -> ToolCall:
    return ToolCall(function=ToolCall.FunctionBody(name="search", arguments="{}"))


def _unparsed(error: str = "Invalid JSON: boom") -> UnparsedToolCall:
    return UnparsedToolCall(raw_text="<tool_call>{bad}</tool_call>", error=error)


class TestClassifyParseFailure:
    def test_clean_response_is_none(self):
        message: Message = {"role": "assistant", "content": "all good"}
        assert classify_parse_failure(message, ParseTermination.STOP_SEQUENCE) is None
        assert classify_parse_failure(message, ParseTermination.EOS) is None

    def test_clean_with_valid_tool_calls_is_none(self):
        message: Message = {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [_valid_tool_call()],
        }
        assert classify_parse_failure(message, ParseTermination.STOP_SEQUENCE) is None

    def test_malformed_is_structural(self):
        message: Message = {"role": "assistant", "content": "truncated..."}
        result = classify_parse_failure(message, ParseTermination.MALFORMED)
        assert result is not None
        kind, detail = result
        assert kind == ParseFailureKind.STRUCTURAL
        assert "did not terminate cleanly" in detail

    def test_malformed_wins_over_unparsed(self):
        """A structural failure makes the content signal unreliable."""
        message: Message = {
            "role": "assistant",
            "content": "truncated...",
            "unparsed_tool_calls": [_unparsed()],
        }
        result = classify_parse_failure(message, ParseTermination.MALFORMED)
        assert result is not None
        assert result[0] == ParseFailureKind.STRUCTURAL

    def test_unparsed_only_is_content(self):
        message: Message = {
            "role": "assistant",
            "content": "calling",
            "unparsed_tool_calls": [_unparsed("Invalid JSON: a"), _unparsed("bad args")],
        }
        result = classify_parse_failure(message, ParseTermination.STOP_SEQUENCE)
        assert result is not None
        kind, detail = result
        assert kind == ParseFailureKind.CONTENT
        assert "Invalid JSON: a" in detail
        assert "bad args" in detail

    def test_mixed_valid_and_unparsed_is_content(self):
        """Mixed calls still classify as CONTENT; acting on it is caller policy."""
        message: Message = {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [_valid_tool_call()],
            "unparsed_tool_calls": [_unparsed()],
        }
        result = classify_parse_failure(message, ParseTermination.STOP_SEQUENCE)
        assert result is not None
        assert result[0] == ParseFailureKind.CONTENT

    def test_detail_is_truncated(self):
        message: Message = {
            "role": "assistant",
            "content": "",
            "unparsed_tool_calls": [_unparsed("x" * (2 * PARSE_FAILURE_DETAIL_MAX_CHARS))],
        }
        result = classify_parse_failure(message, ParseTermination.STOP_SEQUENCE)
        assert result is not None
        assert len(result[1]) == PARSE_FAILURE_DETAIL_MAX_CHARS


class TestDetectUnterminatedToolBlock:
    def test_no_markers(self):
        assert detect_unterminated_tool_block("plain text", "<tool_call>", "</tool_call>") is None

    def test_balanced_markers(self):
        content = "<tool_call>{}</tool_call>"
        assert detect_unterminated_tool_block(content, "<tool_call>", "</tool_call>") is None

    def test_dangling_open(self):
        content = 'text <tool_call>{"name": "search"'
        result = detect_unterminated_tool_block(content, "<tool_call>", "</tool_call>")
        assert result is not None
        assert result.error.startswith(UNTERMINATED_TOOL_BLOCK_ERROR)
        assert result.raw_text.startswith("<tool_call>")

    def test_balanced_plus_dangling(self):
        content = "<tool_call>{}</tool_call><tool_call>{dangling"
        result = detect_unterminated_tool_block(content, "<tool_call>", "</tool_call>")
        assert result is not None
        assert result.raw_text == "<tool_call>{dangling"


# =============================================================================
# Per-renderer blind-spot detection: unterminated tool block + clean stop
# =============================================================================


def _assert_unterminated_content_failure(message: Message, termination: ParseTermination):
    """The dangling block must surface as a recoverable CONTENT failure."""
    assert termination.is_clean  # the stop token fired; framing is fine
    assert "unparsed_tool_calls" in message
    unterminated = [
        tc
        for tc in message["unparsed_tool_calls"]
        if tc.error.startswith(UNTERMINATED_TOOL_BLOCK_ERROR)
    ]
    assert len(unterminated) == 1
    result = classify_parse_failure(message, termination)
    assert result is not None
    assert result[0] == ParseFailureKind.CONTENT


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [
        ("Qwen/Qwen3-8B", "qwen3"),
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", "qwen3_instruct"),
    ],
)
def test_qwen3_unterminated_tool_block(model_name: str, renderer_name: str):
    """<tool_call> opened, never closed, clean <|im_end|> stop."""
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    response_text = """I'll search for that.
<tool_call>
{"name": "search", "arguments": {"query": "weather<|im_end|>"""
    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, termination = renderer.parse_response(response_tokens)

    _assert_unterminated_content_failure(message, termination)
    assert "tool_calls" not in message


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [
        ("Qwen/Qwen3.6-35B-A3B", "qwen3_5"),
        ("Qwen/Qwen3.6-35B-A3B", "qwen3_5_disable_thinking"),
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron3"),
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron3_disable_thinking"),
    ],
)
def test_qwen3_5_family_unterminated_tool_block(model_name: str, renderer_name: str):
    """Dangling XML-style <tool_call> block survives the XML conversion pass."""
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    response_text = """I'll search for that.
<tool_call>
<function=search>
<parameter=query>
weather in NYC<|im_end|>"""
    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, termination = renderer.parse_response(response_tokens)

    _assert_unterminated_content_failure(message, termination)
    assert "tool_calls" not in message


@pytest.mark.parametrize(
    "dangling_text",
    [
        # Section opened, never closed.
        '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":',
        # Section closed but the call inside never terminates.
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{}<｜tool▁calls▁end｜>",
    ],
)
@skip_deepseek_tokenizer_bug
def test_deepseek_unterminated_tool_block(dangling_text: str):
    tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3.1")
    renderer = get_renderer("deepseekv3", tokenizer)

    response_text = f"I'll check the weather.\n{dangling_text}<｜end▁of▁sentence｜>"
    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, termination = renderer.parse_response(response_tokens)

    _assert_unterminated_content_failure(message, termination)
    assert "tool_calls" not in message


@pytest.mark.parametrize(
    "dangling_text",
    [
        # Section opened, never closed.
        '<|tool_calls_section_begin|><|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{"query":',
        # Section closed but the call inside never terminates.
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{}<|tool_calls_section_end|>",
    ],
)
@pytest.mark.parametrize("renderer_name", ["kimi_k2", "kimi_k25"])
def test_kimi_k2_family_unterminated_tool_block(renderer_name: str, dangling_text: str):
    tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    renderer = get_renderer(renderer_name, tokenizer)

    response_text = f"<think></think>I'll search.\n{dangling_text}<|im_end|>"
    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, termination = renderer.parse_response(response_tokens)

    _assert_unterminated_content_failure(message, termination)
    assert "tool_calls" not in message


@pytest.mark.parametrize(
    "response_text",
    [
        # Tool-call header whose <|message|> marker never arrives before the stop.
        "<|channel|>analysis<|message|>Thinking.<|end|><|start|>assistant<|channel|>commentary"
        " to=functions.get_weather <|constrain|>json<|return|>",
        # Tool-call header with an empty arguments payload.
        "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|><|call|>",
    ],
)
def test_gptoss_unterminated_tool_block(response_text: str):
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer)

    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, termination = renderer.parse_response(response_tokens)

    _assert_unterminated_content_failure(message, termination)
    assert "tool_calls" not in message


def test_gptoss_complete_tool_call_not_flagged():
    """A well-formed tool call must not trip the dangling-header heuristic."""
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer)

    response_text = (
        "<|channel|>commentary to=functions.get_weather <|constrain|>json"
        '<|message|>{"location": "NYC"}<|call|>'
    )
    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    message, termination = renderer.parse_response(response_tokens)

    assert termination.is_clean
    assert "unparsed_tool_calls" not in message
    assert "tool_calls" in message and len(message["tool_calls"]) == 1
