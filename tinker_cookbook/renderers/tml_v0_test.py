from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, cast

import pytest
import tinker
from PIL import Image
from tml_renderers import chat as tml_chat
from tml_renderers import v0 as public_tml_v0
from tml_renderers.tinker import token_spans_to_tinker_model_input

from tinker_cookbook.renderers import (
    AudioPart,
    ImagePart,
    Message,
    TextPart,
    ToolCall,
    TrainOnWhat,
    get_renderer,
    tml_v0,
)
from tinker_cookbook.renderers.tml import TmlRenderInput
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import SupportsTmlTokenizer, get_tokenizer


def _messages() -> list[Message]:
    return [
        Message(role="system", content="You are concise."),
        Message(role="user", content="Say hello."),
        Message(role="assistant", content="Hello."),
    ]


def _renderer() -> tml_v0.TmlV0Renderer:
    tokenizer = get_tokenizer("thinkingmachines/Inkling")
    renderer = get_renderer("tml_v0", tokenizer)
    assert isinstance(renderer, tml_v0.TmlV0Renderer)
    return renderer


def _input_len(model_input) -> int:
    return sum(int(chunk.length) for chunk in model_input.chunks)


@pytest.mark.parametrize("version", ["2.10.0", "2.12.0+cu130", "3.0.0"])
def test_validate_torch_version_accepts_supported_versions(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setattr(tml_v0.torch, "__version__", version)
    tml_v0._validate_torch_version()


def test_validate_torch_version_rejects_unsupported_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tml_v0.torch, "__version__", "2.9.1")
    with pytest.raises(RuntimeError, match=r"requires PyTorch 2\.10 or newer; found 2\.9\.1"):
        tml_v0._validate_torch_version()


def test_inkling_tokenizer_resolves_to_tml_adapter() -> None:
    tokenizer = get_tokenizer("thinkingmachines/Inkling")

    assert tokenizer.name_or_path == "thinkingmachines/Inkling"
    assert isinstance(tokenizer, SupportsTmlTokenizer)
    assert tokenizer.decode(tokenizer.encode("hello", add_special_tokens=False))


def test_build_supervised_example_returns_unshifted_input_and_weights() -> None:
    renderer = _renderer()

    model_input, weights = renderer.build_supervised_example(_messages())

    assert _input_len(model_input) == len(weights)
    assert float(weights.sum()) > 0


def test_stop_condition_is_a_flat_token_sequence() -> None:
    renderer = _renderer()

    stop = renderer.get_stop_sequences()

    assert stop
    assert all(isinstance(token, int) for token in stop)
    tinker.SamplingParams(stop=stop, max_tokens=16)


def test_build_generation_prompt_defaults_to_high_effort() -> None:
    renderer = _renderer()

    default_prompt = renderer.build_generation_prompt(_messages())
    high_prompt = renderer.build_generation_prompt(_messages(), effort=0.9)

    assert default_prompt.to_ints() == high_prompt.to_ints()
    assert "Thinking effort level: 0.9" in renderer.tokenizer.decode(default_prompt.to_ints())


def test_build_generation_prompt_effort_validates_range() -> None:
    renderer = _renderer()

    with pytest.raises(ValueError, match=r"thinking effort must be.*\[0, 1\)"):
        renderer.build_generation_prompt(_messages(), effort=1.0)


def test_build_supervised_example_defaults_to_high_effort() -> None:
    renderer = _renderer()

    default_input, default_weights = renderer.build_supervised_example(_messages())
    high_input, _ = renderer.build_supervised_example(_messages(), effort=0.9)

    assert default_input.to_ints() == high_input.to_ints()
    assert _input_len(default_input) == len(default_weights)
    assert "Thinking effort level: 0.9" in renderer.tokenizer.decode(default_input.to_ints())


def test_build_supervised_example_effort_validates_range() -> None:
    renderer = _renderer()

    with pytest.raises(ValueError, match=r"thinking effort must be.*\[0, 1\)"):
        renderer.build_supervised_example(_messages(), effort=1.0)


def test_generation_prompt_is_prefix_of_supervised_example() -> None:
    renderer = _renderer()
    prompt_messages = _messages()[:-1]

    prompt_ints = renderer.build_generation_prompt(prompt_messages, effort=0.6).to_ints()
    supervised_input, _ = renderer.build_supervised_example(_messages(), effort=0.6)
    supervised_ints = supervised_input.to_ints()

    assert supervised_ints[: len(prompt_ints)] == prompt_ints


def test_conversation_to_datum_uses_cookbook_shift() -> None:
    renderer = _renderer()

    datum = conversation_to_datum(_messages(), renderer, max_length=None)

    targets = datum.loss_fn_inputs["target_tokens"].to_numpy()
    weights = datum.loss_fn_inputs["weights"].to_numpy()
    assert len(targets) == len(weights)
    assert _input_len(datum.model_input) == len(targets)
    assert float(weights.sum()) > 0


def test_last_assistant_message_masks_earlier_assistant_messages() -> None:
    renderer = _renderer()
    messages = [
        Message(role="user", content="First."),
        Message(role="assistant", content="One."),
        Message(role="user", content="Second."),
        Message(role="assistant", content="Two."),
    ]

    _, all_weights = renderer.build_supervised_example(messages, TrainOnWhat.ALL_ASSISTANT_MESSAGES)
    _, last_weights = renderer.build_supervised_example(
        messages, TrainOnWhat.LAST_ASSISTANT_MESSAGE
    )

    assert 0 < float(last_weights.sum()) < float(all_weights.sum())


def test_unsupported_content_fails_loudly() -> None:
    renderer = _renderer()

    with pytest.raises(ValueError, match="Unsupported content part type"):
        renderer.build_supervised_example(
            [
                Message(
                    role="user",
                    content=cast(Any, [{"type": "video", "video": "gs://example"}]),
                ),
                Message(role="assistant", content="Nope."),
            ]
        )


def test_remote_image_url_fails_loudly() -> None:
    renderer = _renderer()

    with pytest.raises(ValueError, match="does not fetch remote image URLs"):
        renderer.build_supervised_example(
            [
                Message(
                    role="user",
                    content=[ImagePart(type="image", image="gs://example")],
                ),
                Message(role="assistant", content="Nope."),
            ]
        )


def test_image_path_builds_tinker_chunk() -> None:
    renderer = _renderer()
    image = Image.new("RGB", (64, 48), (30, 200, 120))
    model_input = renderer.build_generation_prompt(
        [
            Message(
                role="user",
                content=[
                    TextPart(type="text", text="Describe this image."),
                    ImagePart(type="image", image=image),
                ],
            )
        ]
    )

    image_chunks = [
        chunk for chunk in model_input.chunks if isinstance(chunk, tinker.types.ImageChunk)
    ]
    assert len(image_chunks) == 1
    assert image_chunks[0].format == "jpeg"
    assert image_chunks[0].data


def test_openai_audio_path_builds_tinker_chunk(tmp_path: Path) -> None:
    sample_rate = 16_000
    num_frames = sample_rate // 10
    audio_path = tmp_path / "tone.wav"
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * num_frames)

    renderer = _renderer()

    model_input = renderer.build_generation_prompt(
        [
            Message(
                role="user",
                content=[
                    TextPart(type="text", text="Describe this audio."),
                    AudioPart(type="audio", audio=str(audio_path)),
                ],
            )
        ]
    )
    dmel_chunks = [
        chunk for chunk in model_input.chunks if isinstance(chunk, tinker.types.DmelChunk)
    ]
    assert len(dmel_chunks) == 1
    assert dmel_chunks[0].dmel
    assert int(dmel_chunks[0].length) > 0


def test_partial_assistant_message_fails_loudly() -> None:
    renderer = _renderer()

    with pytest.raises(NotImplementedError, match="does not accept partial assistant messages"):
        renderer.build_generation_prompt(_messages(), prefill="answer:")


def test_empty_partial_assistant_message_fails_loudly() -> None:
    renderer = _renderer()

    with pytest.raises(NotImplementedError, match="does not accept partial assistant messages"):
        renderer.build_generation_prompt(_messages(), prefill="")


def test_tool_calls_are_accepted_through_oss_messages() -> None:
    renderer = _renderer()
    messages = [
        Message(role="user", content="What's the weather?"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call_weather",
                    function=ToolCall.FunctionBody(
                        name="get_weather", arguments='{"city": "San Francisco"}'
                    ),
                )
            ],
        ),
    ]

    model_input = renderer.build_generation_prompt(messages)

    assert _input_len(model_input) > 0


def test_parsed_tml_tool_call_returns_cookbook_tool_call_object() -> None:
    renderer = _renderer()
    tml_renderer = public_tml_v0.Renderer(renderer.tokenizer.tml_tokenizer)
    tool_message = tml_chat.Message(
        content=tml_chat.InvokeTool(
            tml_chat.StructuredToolCall(
                name="get_weather",
                args=[tml_chat.ToolArg("city", '"San Francisco"')],
                tool_call_id="call_weather",
            )
        ),
        author=tml_chat.Author(tml_chat.AuthorKind.Model),
        channel_enum=tml_chat.MessageChannel.Commentary,
    )

    stop_message = tml_chat.Message(
        content=tml_chat.ModelEndSampling(),
        author=tml_chat.Author(tml_chat.AuthorKind.Model),
        channel_enum=tml_chat.MessageChannel.Main,
    )
    spans, _ = tml_renderer.render_for_completion([tool_message, stop_message])
    model_input = token_spans_to_tinker_model_input(spans)
    message, termination = renderer.parse_response(model_input.to_ints())

    assert termination.is_clean
    tool_calls = message.get("tool_calls")
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert '"San Francisco"' in tool_calls[0].function.arguments


def test_tool_declarations_emit_tool_declare_prefix() -> None:
    renderer = _renderer()

    prefix = renderer.create_conversation_prefix_with_tools(
        [
            {
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        system_prompt="Use tools when needed.",
    )

    assert [message["role"] for message in prefix] == ["system", "tool_declare"]
    assert prefix[0]["content"] == "Use tools when needed."
    assert '"name":"get_weather"' in prefix[1]["content"]
    assert '"type":"function"' in prefix[1]["content"]

    model_input = renderer.build_generation_prompt(
        prefix + [Message(role="user", content="What's the weather in SF?")]
    )

    assert _input_len(model_input) > 0


def test_native_tml_renderers_inputs_are_accepted_and_terminated() -> None:
    renderer = _renderer()
    native_messages = [
        tml_chat.Message(
            content=tml_chat.Text("Say hello."),
            author=tml_chat.Author(tml_chat.AuthorKind.User),
            channel_enum=tml_chat.MessageChannel.Main,
        ),
        tml_chat.Message(
            content=tml_chat.Text("Hello."),
            author=tml_chat.Author(tml_chat.AuthorKind.Model),
            channel_enum=tml_chat.MessageChannel.Main,
        ),
    ]
    native_inputs: list[TmlRenderInput] = [
        native_messages,
        tml_chat.MessageList(native_messages),
        tml_chat.OpenAIMessage.from_oss_messages(_messages()),
    ]

    for native_input in native_inputs:
        model_input, weights = renderer.build_supervised_example(native_input)
        assert _input_len(model_input) == len(weights)
        assert float(weights.sum()) > 0

    stop = tml_chat.Message(
        content=tml_chat.ModelEndSampling(),
        author=tml_chat.Author(tml_chat.AuthorKind.Model),
    )
    bare_input, bare_weights = renderer.build_supervised_example(native_messages)
    explicit_input, explicit_weights = renderer.build_supervised_example([*native_messages, stop])

    # The cookbook terminates model turns automatically, so omitting the
    # explicit ModelEndSampling renders token-identically (including the
    # weighted stop token).
    assert bare_input.to_ints() == explicit_input.to_ints()
    assert bare_weights.tolist() == explicit_weights.tolist()
    assert float(bare_weights.sum()) > 0


def test_selective_sft_modes_require_cookbook_dict_messages_for_masking() -> None:
    renderer = _renderer()
    openai_messages = tml_chat.OpenAIMessage.from_oss_messages(_messages())

    with pytest.raises(NotImplementedError, match="selective train_on_what"):
        renderer.build_supervised_example(openai_messages, TrainOnWhat.LAST_ASSISTANT_MESSAGE)


def test_extension_property_holds_multiturn() -> None:
    """Prove the `has_extension_property=True` claim on a real multi-turn conversation."""
    renderer = _renderer()
    messages = [
        Message(role="system", content="You are concise."),
        Message(role="user", content="What is 2+2?"),
        Message(role="assistant", content="4."),
        Message(role="user", content="And 3+3?"),
        Message(role="assistant", content="6."),
    ]

    assert renderer.has_extension_property
    sequence_through_first_assistant = renderer.build_generation_prompt(messages[:3]).to_ints()
    prompt_before_second_assistant = renderer.build_generation_prompt(messages[:4]).to_ints()
    assert (
        prompt_before_second_assistant[: len(sequence_through_first_assistant)]
        == sequence_through_first_assistant
    )
