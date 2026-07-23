from __future__ import annotations

import base64
import wave
from pathlib import Path
from typing import Any, cast

import pytest
import tinker
from PIL import Image

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
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import ensure_tml_renderers_importable, get_tokenizer


class _FakeTmlRenderersChat:
    class AuthorKind:
        Model = "model"
        User = "user"

    class MessageList:
        pass

    class MessageMetadata:
        def __init__(self, training_metadata=None):
            self.training_metadata = training_metadata

    class TrainingMetadata:
        def __init__(self, weight: float, is_synthetic: bool):
            self.weight = weight
            self.is_synthetic = is_synthetic

    class _Author:
        def __init__(self, kind):
            self.kind = kind

    class Message:
        def __init__(self, kind):
            self.author = _FakeTmlRenderersChat._Author(kind)
            self.message_metadata = None

        def copy(self, message_metadata=None):
            copied = _FakeTmlRenderersChat.Message(self.author.kind)
            copied.message_metadata = message_metadata
            return copied

    class OpenAIMessage:
        source_messages: Any = None

        def __init__(self, role):
            self.role = role

        @staticmethod
        def from_oss_messages(messages):
            _FakeTmlRenderersChat.OpenAIMessage.source_messages = messages
            return [_FakeTmlRenderersChat.OpenAIMessage(message["role"]) for message in messages]

        def to_messages(self):
            if self.role == "assistant":
                return [_FakeTmlRenderersChat.Message(_FakeTmlRenderersChat.AuthorKind.Model)]
            return [_FakeTmlRenderersChat.Message(_FakeTmlRenderersChat.AuthorKind.User)]


@pytest.fixture
def mock_tml_renderers_chat(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTmlRenderersChat]:
    _FakeTmlRenderersChat.OpenAIMessage.source_messages = None
    monkeypatch.setattr(
        tml_v0,
        "import_module",
        lambda module="tml_renderers": _FakeTmlRenderersChat,
    )
    return _FakeTmlRenderersChat


def _require_tml_renderers() -> None:
    try:
        ensure_tml_renderers_importable()
        chat = tml_v0.import_module("tml_renderers.chat")
        __import__("tml_renderers.v0")
        __import__("tml_renderers.tinker")
        if not hasattr(chat, "OpenAIMessage"):
            pytest.skip("optional tml_renderers package does not expose OpenAIMessage yet")
    except Exception as exc:
        pytest.skip(f"optional tml_renderers package is not importable: {exc}")


def _messages() -> list[Message]:
    return [
        Message(role="system", content="You are concise."),
        Message(role="user", content="Say hello."),
        Message(role="assistant", content="Hello."),
    ]


def _renderer() -> tml_v0.TmlV0Renderer:
    _require_tml_renderers()
    tokenizer = get_tokenizer("thinkingmachines/Inkling")
    return cast(tml_v0.TmlV0Renderer, get_renderer("tml_v0", tokenizer))


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


def test_cookbook_dicts_are_converted_with_tool_calls_for_openai_message(
    mock_tml_renderers_chat: type[_FakeTmlRenderersChat],
) -> None:
    messages = [
        Message(role="user", content="Weather?"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call_weather",
                    function=ToolCall.FunctionBody(name="get_weather", arguments='{"city": "SF"}'),
                )
            ],
        ),
    ]

    rendered = cast(Any, tml_v0._messages_to_render_input(messages))

    assert all(isinstance(message, mock_tml_renderers_chat.OpenAIMessage) for message in rendered)
    source_messages = mock_tml_renderers_chat.OpenAIMessage.source_messages
    assert source_messages is not None
    assert source_messages[1]["tool_calls"][0]["id"] == "call_weather"
    assert isinstance(source_messages[1]["tool_calls"][0], dict)


def test_cookbook_audio_bytes_are_normalized_to_openai_input_audio(
    mock_tml_renderers_chat: type[_FakeTmlRenderersChat],
) -> None:
    audio = b"RIFF-test-wav-bytes"
    messages = [
        Message(
            role="user",
            content=[
                TextPart(type="text", text="Transcribe this."),
                AudioPart(type="audio", audio=audio, format="wav"),
            ],
        )
    ]

    tml_v0._messages_to_render_input(messages)

    source_messages = mock_tml_renderers_chat.OpenAIMessage.source_messages
    assert source_messages is not None
    audio_part = source_messages[0]["content"][1]
    assert audio_part["type"] == "input_audio"
    assert audio_part["input_audio"] == {
        "data": base64.b64encode(audio).decode("ascii"),
        "format": "wav",
    }


def test_cookbook_audio_local_path_and_metadata_are_normalized(
    tmp_path: Path,
    mock_tml_renderers_chat: type[_FakeTmlRenderersChat],
) -> None:
    audio = b"not-really-mp3"
    path = tmp_path / "clip.mp3"
    path.write_bytes(audio)
    messages = [
        Message(
            role="user",
            content=[
                AudioPart(
                    type="audio",
                    audio=str(path),
                    num_frames=48_000,
                    sample_rate=24_000,
                )
            ],
        )
    ]

    tml_v0._messages_to_render_input(messages)

    source_messages = mock_tml_renderers_chat.OpenAIMessage.source_messages
    assert source_messages is not None
    assert source_messages[0]["content"][0]["input_audio"] == {
        "data": base64.b64encode(audio).decode("ascii"),
        "format": "mp3",
        "num_frames": 48_000,
        "sample_rate": 24_000,
    }


def test_cookbook_audio_non_wav_requires_complete_metadata() -> None:
    with pytest.raises(ValueError, match="must provide num_frames and sample_rate"):
        tml_v0._audio_part_to_openai(
            AudioPart(type="audio", audio=b"mp3", format="mp3", num_frames=48_000)
        )


def test_cookbook_audio_metadata_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        tml_v0._audio_part_to_openai(
            AudioPart(
                type="audio",
                audio=b"mp3",
                format="mp3",
                num_frames=48_000,
                sample_rate=0,
            )
        )


def test_cookbook_audio_remote_url_is_rejected() -> None:
    part = AudioPart(type="audio", audio="https://example.com/clip.wav")

    with pytest.raises(ValueError, match="does not fetch remote audio URLs"):
        tml_v0._audio_part_to_openai(part)


def test_native_tml_renderers_inputs_pass_through(
    mock_tml_renderers_chat: type[_FakeTmlRenderersChat],
) -> None:
    messages = [
        mock_tml_renderers_chat.Message(mock_tml_renderers_chat.AuthorKind.User),
        mock_tml_renderers_chat.OpenAIMessage("user"),
    ]
    message_list = mock_tml_renderers_chat.MessageList()

    assert tml_v0._messages_to_render_input(cast(Any, messages)) is messages
    assert tml_v0._messages_to_render_input(cast(Any, message_list)) is message_list


def test_selective_sft_masking_sets_zero_training_metadata(
    mock_tml_renderers_chat: type[_FakeTmlRenderersChat],
) -> None:
    messages = [
        Message(role="user", content="First."),
        Message(role="assistant", content="One."),
        Message(role="user", content="Second."),
        Message(role="assistant", content="Two."),
    ]

    rendered = cast(
        Any, tml_v0._cookbook_messages_to_sft_input(messages, TrainOnWhat.LAST_ASSISTANT_MESSAGE)
    )

    assert rendered[1].message_metadata.training_metadata.weight == 0.0
    assert rendered[3].message_metadata is None


def test_customized_sft_masking_respects_trainable_flag(
    mock_tml_renderers_chat: type[_FakeTmlRenderersChat],
) -> None:
    messages = [
        Message(role="user", content="Prompt."),
        Message(role="assistant", content="Skip.", trainable=False),
        Message(role="assistant", content="Train.", trainable=True),
    ]

    rendered = cast(Any, tml_v0._cookbook_messages_to_sft_input(messages, TrainOnWhat.CUSTOMIZED))

    assert rendered[1].message_metadata.training_metadata.weight == 0.0
    assert rendered[2].message_metadata is None


def test_selective_sft_rejects_native_tml_renderers_inputs(
    mock_tml_renderers_chat: type[_FakeTmlRenderersChat],
) -> None:
    with pytest.raises(NotImplementedError, match="selective train_on_what"):
        tml_v0._cookbook_messages_to_sft_input(
            cast(Any, [mock_tml_renderers_chat.OpenAIMessage("assistant")]),
            TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )


def test_inkling_tokenizer_resolves_to_tml_adapter() -> None:
    _require_tml_renderers()
    tokenizer = get_tokenizer("thinkingmachines/Inkling")

    assert tokenizer.name_or_path == "thinkingmachines/Inkling"
    assert hasattr(tokenizer, "tml_tokenizer")
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

    with pytest.raises(Exception, match="Unsupported content part type"):
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

    with pytest.raises(Exception, match="does not fetch remote image URLs"):
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
    _require_tml_renderers()
    dmel_chunk_type = getattr(tinker.types, "DmelChunk", None)
    if dmel_chunk_type is None:
        pytest.skip("DmelChunk is unavailable; please upgrade the Tinker SDK")
    assert dmel_chunk_type is not None

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
    dmel_chunks = [chunk for chunk in model_input.chunks if isinstance(chunk, dmel_chunk_type)]
    assert len(dmel_chunks) == 1
    dmel_chunk = cast(Any, dmel_chunks[0])
    assert dmel_chunk.dmel
    assert int(dmel_chunk.length) > 0


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
    _require_tml_renderers()
    chat = cast(Any, tml_v0.import_module("tml_renderers.chat"))
    tml_v0_module = cast(Any, tml_v0.import_module("tml_renderers.v0"))

    renderer = _renderer()
    tml_renderer = tml_v0_module.Renderer(renderer.tokenizer.tml_tokenizer)
    tool_message = chat.Message(
        content=chat.InvokeTool(
            chat.StructuredToolCall(
                name="get_weather",
                args=[chat.ToolArg("city", '"San Francisco"')],
                tool_call_id="call_weather",
            )
        ),
        author=chat.Author(chat.AuthorKind.Model),
        channel_enum=chat.MessageChannel.Commentary,
    )

    stop_message = chat.Message(
        content=chat.ModelEndSampling(),
        author=chat.Author(chat.AuthorKind.Model),
        channel_enum=chat.MessageChannel.Main,
    )
    spans, _ = tml_renderer.render_for_completion([tool_message, stop_message])
    model_input = tml_v0.import_module("tml_renderers.tinker").token_spans_to_tinker_model_input(
        spans
    )
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


def test_native_tml_renderers_messages_are_accepted_directly() -> None:
    _require_tml_renderers()
    chat = cast(Any, tml_v0.import_module("tml_renderers.chat"))

    renderer = _renderer()
    messages = [
        chat.Message(
            content=chat.Text("Say hello."),
            author=chat.Author(chat.AuthorKind.User),
            channel_enum=chat.MessageChannel.Main,
        ),
        chat.Message(
            content=chat.Text("Hello."),
            author=chat.Author(chat.AuthorKind.Model),
            channel_enum=chat.MessageChannel.Main,
        ),
    ]

    model_input, weights = renderer.build_supervised_example(messages)

    assert _input_len(model_input) == len(weights)
    assert float(weights.sum()) > 0


def test_native_sft_input_gets_model_end_sampling_by_default() -> None:
    _require_tml_renderers()
    chat = cast(Any, tml_v0.import_module("tml_renderers.chat"))

    renderer = _renderer()
    native = [
        chat.Message(
            content=chat.Text("Say hello."),
            author=chat.Author(chat.AuthorKind.User),
            channel_enum=chat.MessageChannel.Main,
        ),
        chat.Message(
            content=chat.Text("Hello."),
            author=chat.Author(chat.AuthorKind.Model),
            channel_enum=chat.MessageChannel.Main,
        ),
    ]
    stop = chat.Message(
        content=chat.ModelEndSampling(),
        author=chat.Author(chat.AuthorKind.Model),
    )

    bare_input, bare_weights = renderer.build_supervised_example(native)
    explicit_input, explicit_weights = renderer.build_supervised_example(native + [stop])

    # The cookbook terminates model turns automatically, so omitting the
    # explicit ModelEndSampling renders token-identically (including the
    # weighted stop token).
    assert bare_input.to_ints() == explicit_input.to_ints()
    assert bare_weights.tolist() == explicit_weights.tolist()
    assert float(bare_weights.sum()) > 0


def test_native_tml_renderers_openai_messages_are_accepted_directly() -> None:
    _require_tml_renderers()
    chat = cast(Any, tml_v0.import_module("tml_renderers.chat"))

    renderer = _renderer()
    openai_messages = chat.OpenAIMessage.from_oss_messages(_messages())

    model_input, weights = renderer.build_supervised_example(openai_messages)

    assert _input_len(model_input) == len(weights)
    assert float(weights.sum()) > 0


def test_native_tml_renderers_message_list_is_accepted_directly() -> None:
    _require_tml_renderers()
    chat = cast(Any, tml_v0.import_module("tml_renderers.chat"))

    renderer = _renderer()
    messages = chat.MessageList(
        [
            chat.Message(
                content=chat.Text("Say hello."),
                author=chat.Author(chat.AuthorKind.User),
                channel_enum=chat.MessageChannel.Main,
            ),
            chat.Message(
                content=chat.Text("Hello."),
                author=chat.Author(chat.AuthorKind.Model),
                channel_enum=chat.MessageChannel.Main,
            ),
        ]
    )

    model_input, weights = renderer.build_supervised_example(messages)

    assert _input_len(model_input) == len(weights)
    assert float(weights.sum()) > 0


def test_selective_sft_modes_require_cookbook_dict_messages_for_masking() -> None:
    _require_tml_renderers()
    chat = cast(Any, tml_v0.import_module("tml_renderers.chat"))

    renderer = _renderer()
    openai_messages = chat.OpenAIMessage.from_oss_messages(_messages())

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
