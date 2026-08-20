"""Tests for Qwen3.8-specific renderer behavior.

Covers what qwen3_8 changes over qwen3_5 (see qwen3_8.py):
- reasoning-effort instruction injection into the system message
- thinking preserved in history by default (HF preserve_thinking=true)
- empty system messages rendering nothing

HF-parity, tool-calling, parsing, and consistency coverage lives in the shared
parametrized suites (renderers_test.py, parsing_test.py, tool_calling_test.py,
qwen3_tool_declaration_test.py).
"""

import pytest

from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.renderers import (
    Message,
    TextPart,
    ThinkingPart,
    ToolSpec,
    TrainOnWhat,
    get_renderer,
)
from tinker_cookbook.renderers.qwen3_8 import (
    REASONING_EFFORT_INSTRUCTIONS,
    Qwen3_8Renderer,
)
from tinker_cookbook.renderers.testing_utils import extract_token_ids
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL_NAME = "Qwen/Qwen3.8-27B"

XHIGH = REASONING_EFFORT_INSTRUCTIONS["xhigh"]
LOW = REASONING_EFFORT_INSTRUCTIONS["low"]


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL_NAME)


def _prompt_str(renderer, messages: list[Message]) -> str:
    return renderer.tokenizer.decode(renderer.build_generation_prompt(messages).to_ints())


@pytest.mark.parametrize(
    "renderer_name,instruction",
    [
        ("qwen3_8_xhigh_reasoning", XHIGH),
        ("qwen3_8_low_reasoning", LOW),
    ],
)
def test_instruction_injected_without_system_message(tokenizer, renderer_name, instruction):
    """With no system message, the instruction becomes its own system message."""
    renderer = get_renderer(renderer_name, tokenizer)
    prompt = _prompt_str(renderer, [Message(role="user", content="hi")])
    assert prompt.startswith(f"<|im_start|>system\n{instruction}<|im_end|>\n")


@pytest.mark.parametrize("renderer_name", ["qwen3_8_medium_reasoning", "qwen3_8_disable_thinking"])
def test_no_instruction_for_medium_and_disabled(tokenizer, renderer_name):
    """Medium effort and disabled thinking inject nothing."""
    renderer = get_renderer(renderer_name, tokenizer)
    prompt = _prompt_str(renderer, [Message(role="user", content="hi")])
    assert prompt.startswith("<|im_start|>user\n")
    assert "Reasoning effort" not in prompt


def test_instruction_prepended_to_existing_system_message(tokenizer):
    renderer = get_renderer("qwen3_8_xhigh_reasoning", tokenizer)
    prompt = _prompt_str(
        renderer,
        [Message(role="system", content="Be terse."), Message(role="user", content="hi")],
    )
    assert prompt.startswith(f"<|im_start|>system\n{XHIGH}\n\nBe terse.<|im_end|>\n")


def test_empty_system_message_renders_nothing_without_instruction(tokenizer):
    """The Qwen3.8 template emits no system block for an empty system message."""
    renderer = get_renderer("qwen3_8_medium_reasoning", tokenizer)
    prompt = _prompt_str(
        renderer,
        [Message(role="system", content=""), Message(role="user", content="hi")],
    )
    assert prompt.startswith("<|im_start|>user\n")


def test_empty_system_message_carries_just_the_instruction(tokenizer):
    renderer = get_renderer("qwen3_8_xhigh_reasoning", tokenizer)
    prompt = _prompt_str(
        renderer,
        [Message(role="system", content=""), Message(role="user", content="hi")],
    )
    assert prompt.startswith(f"<|im_start|>system\n{XHIGH}<|im_end|>\n")


def test_instruction_comes_before_tool_declarations(tokenizer):
    """With tools, the template puts the instruction before the # Tools block."""
    renderer = get_renderer("qwen3_8_xhigh_reasoning", tokenizer)
    tools: list[ToolSpec] = [
        {"name": "search", "description": "d", "parameters": {"type": "object"}}
    ]
    convo = renderer.create_conversation_prefix_with_tools(tools, "Be terse.") + [
        Message(role="user", content="hi")
    ]
    prompt = _prompt_str(renderer, convo)
    assert prompt.startswith(f"<|im_start|>system\n{XHIGH}\n\n# Tools")
    assert "Be terse." in prompt


def test_invalid_reasoning_effort_raises(tokenizer):
    with pytest.raises(RendererError, match="Unexpected reasoning effort"):
        Qwen3_8Renderer(tokenizer, reasoning_effort="high")


def test_history_thinking_preserved_by_default(tokenizer):
    """Every assistant message keeps its think block; turns without reasoning get
    an empty block, at any position (HF preserve_thinking defaults to true)."""
    renderer = get_renderer("qwen3_8_medium_reasoning", tokenizer)
    messages: list[Message] = [
        Message(role="user", content="q1"),
        Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking="first turn reasoning"),
                TextPart(type="text", text="a1"),
            ],
        ),
        Message(role="user", content="q2"),
        Message(role="assistant", content="a2"),
        Message(role="user", content="q3"),
    ]
    prompt = _prompt_str(renderer, messages)
    assert "<think>\nfirst turn reasoning\n</think>\n\na1" in prompt
    assert "<think>\n\n</think>\n\na2" in prompt


def test_strip_thinking_from_history_true_matches_qwen3_5_rule(tokenizer):
    """strip_thinking_from_history=True falls back to the positional rule."""
    renderer = Qwen3_8Renderer(
        tokenizer, strip_thinking_from_history=True, reasoning_effort="medium"
    )
    messages: list[Message] = [
        Message(role="user", content="q1"),
        Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking="first turn reasoning"),
                TextPart(type="text", text="a1"),
            ],
        ),
        Message(role="user", content="q2"),
    ]
    prompt = _prompt_str(renderer, messages)
    assert "first turn reasoning" not in prompt
    assert "<think>\n\n</think>" not in prompt.rsplit("<|im_start|>assistant", 1)[0]


def _sequence_through(renderer, messages: list[Message]) -> list[int]:
    model_input, _ = renderer.build_supervised_example(messages)
    return model_input.to_ints()


def test_extension_holds_behaviorally_for_reasoning_turns(tokenizer):
    """With thinking preserved (the default), a turn that reasoned renders the same
    as history, so the sampled sequence extends into the next prompt."""
    renderer = get_renderer("qwen3_8_medium_reasoning", tokenizer)
    messages: list[Message] = [
        Message(role="user", content="q1"),
        Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking="some reasoning"),
                TextPart(type="text", text="a1"),
            ],
        ),
        Message(role="user", content="q2"),
    ]
    seq = _sequence_through(renderer, messages[:2])
    next_prompt = renderer.build_generation_prompt(messages).to_ints()
    assert next_prompt[: len(seq)] == seq

    disable = get_renderer("qwen3_8_disable_thinking", tokenizer)
    plain: list[Message] = [
        Message(role="user", content="q1"),
        Message(role="assistant", content="a1"),
        Message(role="user", content="q2"),
    ]
    seq = _sequence_through(disable, plain[:2])
    next_prompt = disable.build_generation_prompt(plain).to_ints()
    assert next_prompt[: len(seq)] == seq


def test_extension_breaks_for_no_reasoning_turn(tokenizer):
    """Documents why has_extension_property is False: the sampled form of a turn
    that did not reason (open <think>\\n prefill, then its own \\n) is not a
    token-level prefix of the next prompt, where the closed empty block's \\n\\n
    merges into one token."""
    renderer = get_renderer("qwen3_8_medium_reasoning", tokenizer)
    messages: list[Message] = [
        Message(role="user", content="q1"),
        Message(role="assistant", content="a1"),
        Message(role="user", content="q2"),
    ]
    seq = _sequence_through(renderer, messages[:2])
    next_prompt = renderer.build_generation_prompt(messages).to_ints()
    assert next_prompt[: len(seq)] != seq


def test_does_not_claim_extension_property(tokenizer):
    """A no-reasoning turn is sampled after the open <think>\\n (its own \\n token),
    but history writes the closed empty block whose \\n\\n merges into one token —
    so the renderer must not advertise the extension property."""
    assert not get_renderer("qwen3_8_xhigh_reasoning", tokenizer).has_extension_property
    assert not get_renderer("qwen3_8_disable_thinking", tokenizer).has_extension_property
    assert not Qwen3_8Renderer(tokenizer, strip_thinking_from_history=True).has_extension_property


@pytest.mark.parametrize("with_system", [True, False])
def test_customized_training_keeps_trainable_metadata(tokenizer, with_system):
    """Normalization must not drop `trainable` from a rebuilt system message, and a
    synthetic instruction-only system message must carry one (CUSTOMIZED asserts
    every message has the field)."""
    renderer = get_renderer("qwen3_8_xhigh_reasoning", tokenizer)
    messages: list[Message] = [
        Message(role="user", content="q", trainable=False),
        Message(role="assistant", content="a", trainable=True),
    ]
    if with_system:
        messages.insert(0, Message(role="system", content="sys", trainable=False))
    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.CUSTOMIZED
    )
    assert len(weights) == len(model_input.to_ints())
    assert any(w > 0 for w in weights.tolist())


def test_inline_think_string_matches_hf(tokenizer):
    """Qwen3.8 does not extract inline <think> tags from string content: the HF
    template still writes the empty framing block and keeps the inline block as
    ordinary content, so the renderer must too."""
    renderer = get_renderer("qwen3_8_medium_reasoning", tokenizer)
    messages: list[Message] = [
        Message(role="user", content="q"),
        Message(role="assistant", content="<think>hidden</think>ans"),
        Message(role="user", content="q2"),
    ]
    ours = renderer.build_generation_prompt(messages).to_ints()
    hf_tokens = extract_token_ids(
        tokenizer.apply_chat_template(
            [renderer.to_openai_message(m) for m in messages],
            add_generation_prompt=True,
            tokenize=True,
            reasoning_effort="medium",
        )
    )
    assert ours == hf_tokens


def test_empty_list_system_message_renders_like_empty_string(tokenizer):
    """List content whose text parts are all whitespace renders no system block
    (medium) or just the instruction (xhigh), same as an empty string."""
    messages: list[Message] = [
        Message(role="system", content=[TextPart(type="text", text="  ")]),
        Message(role="user", content="hi"),
    ]
    medium = get_renderer("qwen3_8_medium_reasoning", tokenizer)
    assert _prompt_str(medium, messages).startswith("<|im_start|>user\n")
    xhigh = get_renderer("qwen3_8_xhigh_reasoning", tokenizer)
    assert _prompt_str(xhigh, messages).startswith(f"<|im_start|>system\n{XHIGH}<|im_end|>\n")


def test_list_system_content_is_trimmed(tokenizer):
    """The template applies render_content|trim to system content, so list-based
    text is trimmed exactly like string content."""
    messages: list[Message] = [
        Message(role="system", content=[TextPart(type="text", text="  sys  ")]),
        Message(role="user", content="hi"),
    ]
    medium = get_renderer("qwen3_8_medium_reasoning", tokenizer)
    assert _prompt_str(medium, messages).startswith("<|im_start|>system\nsys<|im_end|>\n")
    xhigh = get_renderer("qwen3_8_xhigh_reasoning", tokenizer)
    assert _prompt_str(xhigh, messages).startswith(
        f"<|im_start|>system\n{XHIGH}\n\nsys<|im_end|>\n"
    )


def test_strip_true_keeps_post_query_reasoning(tokenizer):
    """With strip_thinking_from_history=True (HF preserve_thinking=false), the
    template still keeps reasoning for every assistant AFTER the last user
    message — e.g. a tool-call turn followed by tool results."""
    renderer = Qwen3_8Renderer(
        tokenizer, strip_thinking_from_history=True, reasoning_effort="medium"
    )
    messages: list[Message] = [
        Message(role="user", content="q1"),
        Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking="old reasoning"),
                TextPart(type="text", text="a1"),
            ],
        ),
        Message(role="user", content="q2"),
        Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking="tool reasoning"),
                TextPart(type="text", text="calling"),
            ],
        ),
        Message(role="tool", content="res"),
    ]
    ours = renderer.build_generation_prompt(messages).to_ints()
    hf_tokens = extract_token_ids(
        tokenizer.apply_chat_template(
            [renderer.to_openai_message(m) for m in messages],
            add_generation_prompt=True,
            tokenize=True,
            preserve_thinking=False,
            reasoning_effort="medium",
        )
    )
    assert ours == hf_tokens
    decoded = tokenizer.decode(ours)
    assert "old reasoning" not in decoded
    assert "tool reasoning" in decoded
