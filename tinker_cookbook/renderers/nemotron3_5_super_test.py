"""
Tests for Nemotron-3.5 Super, which uses the Ultra-format renderers and adds image input.

Tests verify, against the Nemotron-3.5 Super HF template, in thinking, chat, and medium-effort
modes:
1. Thinking history, tool calls, and parallel tool results match HF for generation and
   supervised examples; nemotron3_ultra_preserve_thinking matches truncate_history_thinking=False
2. List content joins its text parts, as this template does
3. Each image takes the place of the template's <image> placeholder, between <img> and </img>;
   an image after text starts on a new line; a message with more than one image labels each
   one <image N>, counted per message; images render in user, system, and tool messages,
   before, between, or after text
4. Medium effort appends its suffix after the images of the last user message
"""

import json
from collections.abc import Callable
from typing import Any, cast

import pytest
import tinker
from PIL import Image

from tinker_cookbook.renderers import (
    ImagePart,
    Message,
    Renderer,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    get_renderer,
)
from tinker_cookbook.renderers.testing_utils import extract_token_ids
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"


# =============================================================================
# Test Fixtures
# =============================================================================


class _StubImageProcessor:
    """Stands in for NVIDIA's processor, which is remote code. The template writes one
    placeholder per image whatever its token count; image_token_count_test checks the counts."""

    merge_size = 1

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs: dict[str, Any] | None = None
    ) -> int:
        return 3


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL)


_HF_MODES = pytest.mark.parametrize(
    "renderer_name,hf_kwargs",
    [
        ("nemotron3_ultra", {}),
        ("nemotron3_ultra_disable_thinking", {"enable_thinking": False}),
        ("nemotron3_ultra_medium_thinking", {"medium_effort": True}),
    ],
    ids=["thinking", "chat", "medium"],
)

# Every renderer takes images; the image tests also cover the preserve-thinking variant.
_IMAGE_MODES = pytest.mark.parametrize(
    "renderer_name,hf_kwargs",
    [
        ("nemotron3_ultra", {}),
        ("nemotron3_ultra_disable_thinking", {"enable_thinking": False}),
        ("nemotron3_ultra_medium_thinking", {"medium_effort": True}),
        ("nemotron3_ultra_preserve_thinking", {"truncate_history_thinking": False}),
    ],
    ids=["thinking", "chat", "medium", "preserve"],
)

# The chat renderer refuses to train on a turn with reasoning.
_THINKING_MODES = pytest.mark.parametrize(
    "renderer_name,hf_kwargs",
    [("nemotron3_ultra", {}), ("nemotron3_ultra_medium_thinking", {"medium_effort": True})],
    ids=["thinking", "medium"],
)


def _renderer(renderer_name: str, tokenizer) -> Renderer:
    return get_renderer(renderer_name, tokenizer, cast(Any, _StubImageProcessor()))


def _hf_generation_tokens(tokenizer, hf_messages, tools=None, **hf_kwargs) -> list[int]:
    """Run HF apply_chat_template with generation prompt and return token list."""
    return extract_token_ids(
        tokenizer.apply_chat_template(
            hf_messages, tools=tools, add_generation_prompt=True, tokenize=True, **hf_kwargs
        )
    )


def _hf_supervised_tokens(tokenizer, hf_messages, tools=None, **hf_kwargs) -> list[int]:
    """Run HF apply_chat_template without generation prompt and drop its trailing newline,
    which the renderer writes as part of the next message's header."""
    hf_str = tokenizer.apply_chat_template(hf_messages, tools=tools, tokenize=False, **hf_kwargs)
    assert isinstance(hf_str, str)
    return tokenizer.encode(hf_str.removesuffix("\n"), add_special_tokens=False)


def _with_template_placeholders(tokenizer, model_input: tinker.ModelInput) -> list[int]:
    """Undo the processor's part of the layout.

    The template writes one ``<image>`` token per image, and the HF processor expands it to
    ``<img>``, the image tokens, and ``</img>``. The renderer writes the expanded form, so this
    turns each ``<img>``, image chunk, ``</img>`` run back into ``<image>``.
    """
    image, img, end_img = (
        tokenizer.convert_tokens_to_ids(token) for token in ("<image>", "<img>", "</img>")
    )
    tokens: list[int] = []
    after_image = False
    for chunk in model_input.chunks:
        if isinstance(chunk, tinker.types.ImageChunk):
            assert tokens[-1] == img
            tokens[-1] = image
            after_image = True
            continue
        assert isinstance(chunk, tinker.types.EncodedTextChunk)
        chunk_tokens = list(chunk.tokens)
        if after_image:
            assert chunk_tokens[0] == end_img
            chunk_tokens = chunk_tokens[1:]
            after_image = False
        tokens += chunk_tokens
    return tokens


# =============================================================================
# Test Conversations
# =============================================================================

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Get the weather",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["location"],
    },
)
HF_WEATHER_TOOLS = [{"type": "function", "function": WEATHER_TOOL}]


def _weather_call(arguments: dict, call_id: str) -> ToolCall:
    return ToolCall(
        function=ToolCall.FunctionBody(name="get_weather", arguments=json.dumps(arguments)),
        id=call_id,
    )


def _image(width: int = 64, height: int = 48) -> ImagePart:
    return ImagePart(type="image", image=Image.new("RGB", (width, height)))


def _text(text: str) -> TextPart:
    return TextPart(type="text", text=text)


def get_thinking_conversation_for_supervised() -> list[Message]:
    """Conversation with thinking content, ending with assistant."""
    return [
        Message(role="user", content="Solve 2+2."),
        Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking="2 plus 2 equals 4."),
                TextPart(type="text", text="The answer is 4."),
            ],
        ),
    ]


def get_multiturn_thinking_conversation() -> list[Message]:
    """Multi-turn with thinking in both assistant messages."""
    return [
        Message(role="system", content="You are concise."),
        Message(role="user", content="First question."),
        Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking="First turn reasoning."),
                TextPart(type="text", text="First answer."),
            ],
        ),
        Message(role="user", content="Second question."),
        Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking="Second turn reasoning."),
                TextPart(type="text", text="Second answer."),
            ],
        ),
    ]


def get_tool_call_conversation() -> list[Message]:
    """A tool call with thinking, its result, and the answer, followed by a new user turn."""
    return [
        Message(role="user", content="Weather in Paris?"),
        Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking="Look it up."),
                TextPart(type="text", text="Checking."),
            ],
            tool_calls=[_weather_call({"location": "Paris", "days": 2}, "c1")],
        ),
        Message(role="tool", content="sunny", tool_call_id="c1"),
        Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking="The tool says sunny."),
                TextPart(type="text", text="It is sunny."),
            ],
        ),
        Message(role="user", content="And tomorrow?"),
    ]


def get_parallel_tool_call_conversation() -> list[Message]:
    """Two tool calls and their consecutive results, ending with the results."""
    return [
        Message(role="user", content="Weather in Paris and Rome?"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                _weather_call({"location": "Paris"}, "c1"),
                _weather_call({"location": "Rome", "days": 3}, "c2"),
            ],
        ),
        Message(role="tool", content="sun", tool_call_id="c1"),
        Message(role="tool", content="rain", tool_call_id="c2"),
    ]


# =============================================================================
# HF Template Compatibility Tests — Text
# =============================================================================


@_THINKING_MODES
@pytest.mark.parametrize(
    "train_on_what",
    [TrainOnWhat.LAST_ASSISTANT_MESSAGE, TrainOnWhat.ALL_ASSISTANT_MESSAGES],
    ids=["last", "all"],
)
@pytest.mark.parametrize(
    "conversation_fn",
    [get_thinking_conversation_for_supervised, get_multiturn_thinking_conversation],
    ids=["thinking-then-text", "thinking-in-history-and-current"],
)
def test_thinking_supervised_matches_hf(
    tokenizer,
    renderer_name: str,
    hf_kwargs: dict,
    train_on_what: TrainOnWhat,
    conversation_fn: Callable[[], list[Message]],
):
    renderer = _renderer(renderer_name, tokenizer)
    messages = conversation_fn()

    model_input, _ = renderer.build_supervised_example(messages, train_on_what=train_on_what)
    expected = _hf_supervised_tokens(
        tokenizer, [renderer.to_openai_message(m) for m in messages], **hf_kwargs
    )

    assert model_input.to_ints() == expected


@_HF_MODES
def test_tool_call_history_generation_matches_hf(tokenizer, renderer_name: str, hf_kwargs: dict):
    """A tool call before the last user turn loses its thinking but keeps its text."""
    renderer = _renderer(renderer_name, tokenizer)
    conversation = get_tool_call_conversation()
    prefix = renderer.create_conversation_prefix_with_tools([WEATHER_TOOL])

    expected = _hf_generation_tokens(
        tokenizer,
        [renderer.to_openai_message(m) for m in conversation],
        tools=HF_WEATHER_TOOLS,
        **hf_kwargs,
    )

    assert renderer.build_generation_prompt(prefix + conversation).to_ints() == expected


@_THINKING_MODES
def test_tool_call_in_current_turn_supervised_matches_hf(
    tokenizer, renderer_name: str, hf_kwargs: dict
):
    """A tool call after the last user message keeps its thinking, though it is not last."""
    renderer = _renderer(renderer_name, tokenizer)
    conversation = get_tool_call_conversation()[:-1]
    prefix = renderer.create_conversation_prefix_with_tools([WEATHER_TOOL])

    model_input, _ = renderer.build_supervised_example(prefix + conversation)
    expected = _hf_supervised_tokens(
        tokenizer,
        [renderer.to_openai_message(m) for m in conversation],
        tools=HF_WEATHER_TOOLS,
        **hf_kwargs,
    )

    assert model_input.to_ints() == expected


@_HF_MODES
def test_parallel_tool_results_generation_matches_hf(
    tokenizer, renderer_name: str, hf_kwargs: dict
):
    """Consecutive tool results share one user turn."""
    renderer = _renderer(renderer_name, tokenizer)
    messages = get_parallel_tool_call_conversation()

    expected = _hf_generation_tokens(
        tokenizer, [renderer.to_openai_message(m) for m in messages], **hf_kwargs
    )

    assert renderer.build_generation_prompt(messages).to_ints() == expected


def test_preserve_thinking_supervised_matches_hf(tokenizer):
    renderer = _renderer("nemotron3_ultra_preserve_thinking", tokenizer)
    messages = get_multiturn_thinking_conversation()

    model_input, _ = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    expected = _hf_supervised_tokens(
        tokenizer,
        [renderer.to_openai_message(m) for m in messages],
        truncate_history_thinking=False,
    )

    assert model_input.to_ints() == expected


@_HF_MODES
def test_list_text_content_matches_hf(tokenizer, renderer_name: str, hf_kwargs: dict):
    """This template joins the text parts of list content; Lightning's prints the list."""
    renderer = _renderer(renderer_name, tokenizer)
    messages = [Message(role="user", content=[_text("Hello, "), _text("world.")])]

    expected = _hf_generation_tokens(tokenizer, messages, **hf_kwargs)

    assert renderer.build_generation_prompt(messages).to_ints() == expected


# =============================================================================
# HF Template Compatibility Tests — Images
# =============================================================================
#
# The HF messages are the cookbook messages: to_openai_message drops image parts, and the
# template reads only each part's type and text.


@_IMAGE_MODES
@pytest.mark.parametrize(
    "content",
    [
        [_image(), _text("What is this?")],
        [_text("Look: "), _image()],
        [_text("Look:\n"), _image()],
        [_image(), _image(32, 32), _text("Compare.")],
        [_image(), _text(" and "), _image(32, 32), _text("?")],
    ],
    ids=[
        "image-then-text",
        "text-then-image",
        "text-with-newline-then-image",
        "two-images",
        "image-text-image",
    ],
)
def test_user_images_generation_matches_hf(
    tokenizer, renderer_name: str, hf_kwargs: dict, content: list
):
    renderer = _renderer(renderer_name, tokenizer)
    messages = [Message(role="user", content=content)]

    expected = _hf_generation_tokens(tokenizer, messages, **hf_kwargs)

    assert (
        _with_template_placeholders(tokenizer, renderer.build_generation_prompt(messages))
        == expected
    )


@_IMAGE_MODES
def test_image_labels_count_per_message(tokenizer, renderer_name: str, hf_kwargs: dict):
    """A later message with one image has no label, whatever came before."""
    renderer = _renderer(renderer_name, tokenizer)
    messages = [
        Message(role="user", content=[_image(), _image(), _text("Compare.")]),
        Message(role="assistant", content="Both are black."),
        Message(role="user", content=[_image(), _text("And this one?")]),
    ]

    expected = _hf_generation_tokens(tokenizer, messages, **hf_kwargs)

    assert (
        _with_template_placeholders(tokenizer, renderer.build_generation_prompt(messages))
        == expected
    )


@_IMAGE_MODES
def test_image_in_tool_result_generation_matches_hf(tokenizer, renderer_name: str, hf_kwargs: dict):
    renderer = _renderer(renderer_name, tokenizer)
    messages = [
        Message(role="user", content="Plot the weather."),
        Message(
            role="assistant",
            content="",
            tool_calls=[_weather_call({"location": "Paris"}, "c1")],
        ),
        Message(role="tool", content=[_image(), _text("Chart.")], tool_call_id="c1"),
    ]
    hf_messages = [renderer.to_openai_message(m) for m in messages[:-1]] + [messages[-1]]

    expected = _hf_generation_tokens(tokenizer, hf_messages, **hf_kwargs)

    assert (
        _with_template_placeholders(tokenizer, renderer.build_generation_prompt(messages))
        == expected
    )


@_IMAGE_MODES
def test_image_in_system_message_generation_matches_hf(
    tokenizer, renderer_name: str, hf_kwargs: dict
):
    renderer = _renderer(renderer_name, tokenizer)
    messages = [
        Message(role="system", content=[_text("Reference: "), _image()]),
        Message(role="user", content="Describe the reference."),
    ]

    expected = _hf_generation_tokens(tokenizer, messages, **hf_kwargs)

    assert (
        _with_template_placeholders(tokenizer, renderer.build_generation_prompt(messages))
        == expected
    )


@_IMAGE_MODES
@pytest.mark.parametrize(
    "train_on_what",
    [TrainOnWhat.LAST_ASSISTANT_MESSAGE, TrainOnWhat.ALL_ASSISTANT_MESSAGES],
    ids=["last", "all"],
)
def test_images_supervised_matches_hf(
    tokenizer, renderer_name: str, hf_kwargs: dict, train_on_what: TrainOnWhat
):
    renderer = _renderer(renderer_name, tokenizer)
    messages = [
        Message(role="user", content=[_image(), _image(), _text("Compare.")]),
        Message(role="assistant", content="Both are black."),
        Message(role="user", content=[_text("And "), _image()]),
        Message(role="assistant", content="Also black."),
    ]

    model_input, _ = renderer.build_supervised_example(messages, train_on_what=train_on_what)
    expected = _hf_supervised_tokens(tokenizer, messages, **hf_kwargs)

    assert _with_template_placeholders(tokenizer, model_input) == expected


@pytest.mark.parametrize("num_images", [1, 2])
def test_image_sits_between_marker_tokens(tokenizer, num_images: int):
    """The HF processor expands each placeholder to the <img> and </img> special tokens around
    the image tokens, and the sampler looks for that pair."""
    renderer = _renderer("nemotron3_ultra", tokenizer)
    content: list = [_image() for _ in range(num_images)] + [_text("Describe.")]
    chunks = renderer.build_generation_prompt([Message(role="user", content=content)]).chunks

    def text_tokens(i: int) -> list[int]:
        chunk = chunks[i]
        assert isinstance(chunk, tinker.types.EncodedTextChunk)
        return list(chunk.tokens)

    positions = [i for i, c in enumerate(chunks) if isinstance(c, tinker.types.ImageChunk)]
    assert len(positions) == num_images
    assert [text_tokens(i - 1)[-1] for i in positions] == [
        tokenizer.convert_tokens_to_ids("<img>")
    ] * num_images
    assert [text_tokens(i + 1)[0] for i in positions] == [
        tokenizer.convert_tokens_to_ids("</img>")
    ] * num_images
