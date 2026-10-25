"""Sample Inkling with audio using OpenAI-compatible or native chat messages.

OpenAI-compatible input (the default):

    uv run python -m tinker_cookbook.scripts.inkling.sample_audio

Native ``tml_renderers.chat`` input:

    uv run python -m tinker_cookbook.scripts.inkling.sample_audio message_format=chat

Provide a local WAV with ``audio_path=/path/to/audio.wav`` instead.
"""

from __future__ import annotations

import asyncio
import base64
import wave
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Literal, TypedDict, cast

import chz
import tinker

from tinker_cookbook import model_info
from tinker_cookbook.renderers import Message, get_renderer, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer

_SAMPLE_ID = "1272-128104-0000"
_SAMPLE_PATH = Path.home() / ".cache" / "tinker-cookbook" / f"{_SAMPLE_ID}.flac"
_SAMPLE_NUM_FRAMES = 93_680
_SAMPLE_RATE = 16_000
_REFERENCE_TRANSCRIPT = (
    "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL"
)
_PROMPT = "Transcribe this speech exactly. Return only the transcript."

AudioFormat = Literal["flac", "wav"]
MessageFormat = Literal["openai", "chat"]


@dataclass(frozen=True)
class AudioInput:
    path: Path
    format: AudioFormat
    num_frames: int
    sample_rate: int
    is_reference: bool


class _AudioData(TypedDict):
    bytes: bytes | None
    path: str | None


class _DatasetRow(TypedDict):
    id: str
    audio: _AudioData


def _download_sample() -> Path:
    if not _SAMPLE_PATH.exists():
        from datasets import Audio, load_dataset

        dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean",
            split="validation",
        ).cast_column("audio", Audio(decode=False))
        row = next(
            row for raw_row in dataset if (row := cast(_DatasetRow, raw_row))["id"] == _SAMPLE_ID
        )
        audio_bytes = row["audio"]["bytes"]
        if audio_bytes is None:
            raise ValueError(f"Hugging Face sample {_SAMPLE_ID!r} did not contain audio bytes")
        _SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SAMPLE_PATH.write_bytes(audio_bytes)
    return _SAMPLE_PATH


def _resolve_audio_input(configured_path: str | None) -> AudioInput:
    if configured_path is None:
        return AudioInput(
            path=_download_sample(),
            format="flac",
            num_frames=_SAMPLE_NUM_FRAMES,
            sample_rate=_SAMPLE_RATE,
            is_reference=True,
        )

    path = Path(configured_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Audio does not exist: {path}")
    with wave.open(str(path), "rb") as wav:
        return AudioInput(
            path=path,
            format="wav",
            num_frames=wav.getnframes(),
            sample_rate=wav.getframerate(),
            is_reference=False,
        )


def _build_openai_messages(audio: AudioInput) -> list[Message]:
    chat = import_module("tml_renderers.chat")
    audio_base64 = base64.b64encode(audio.path.read_bytes()).decode("ascii")
    return cast(
        list[Message],
        chat.OpenAIMessage.from_oss_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_base64,
                                "format": audio.format,
                                "num_frames": audio.num_frames,
                                "sample_rate": audio.sample_rate,
                            },
                        },
                    ],
                }
            ]
        ),
    )


def _build_chat_messages(audio: AudioInput) -> list[Message]:
    chat = import_module("tml_renderers.chat")
    user = chat.Author(chat.AuthorKind.User)
    return cast(
        list[Message],
        chat.MessageList(
            [
                chat.Message(content=chat.Text(_PROMPT), author=user),
                chat.Message(
                    content=chat.AudioPointer(
                        location=str(audio.path),
                        format=(
                            chat.AudioFormat.Flac
                            if audio.format == "flac"
                            else chat.AudioFormat.Wav
                        ),
                        num_frames=audio.num_frames,
                        sample_rate=audio.sample_rate,
                    ),
                    author=user,
                ),
            ]
        ),
    )


MessageBuilder = Callable[[AudioInput], list[Message]]
_MESSAGE_BUILDERS: dict[MessageFormat, MessageBuilder] = {
    "openai": _build_openai_messages,
    "chat": _build_chat_messages,
}


@chz.chz
class Config:
    audio_path: str | None = chz.field(
        default=None,
        doc="Optional local WAV path; defaults to a public Hugging Face LibriSpeech sample.",
    )
    model_name: str = chz.field(
        default="thinkingmachines/Inkling",
        doc="Audio-capable Tinker base model.",
    )
    base_url: str | None = chz.field(default=None, doc="Optional Tinker service URL.")
    message_format: MessageFormat = chz.field(
        default="openai",
        doc="Message representation to render: OpenAI-compatible dictionaries or native chat types.",
    )
    max_tokens: int = chz.field(default=128, doc="Maximum sampled response tokens.")
    temperature: float = chz.field(default=1.0, doc="Sampling temperature.")


async def async_main(cfg: Config) -> None:
    if not model_info.get_model_attributes(cfg.model_name).is_audio_in:
        raise ValueError(f"Audio input is not supported by {cfg.model_name!r}.")

    audio = _resolve_audio_input(cfg.audio_path)
    messages = _MESSAGE_BUILDERS[cfg.message_format](audio)

    renderer = get_renderer(
        model_info.get_recommended_renderer_name(cfg.model_name),
        get_tokenizer(cfg.model_name),
    )
    prompt = renderer.build_generation_prompt(messages)

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    sampling_client = await service_client.create_sampling_client_async(base_model=cfg.model_name)
    response = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            stop=renderer.get_stop_sequences(),
        ),
    )
    message, termination = renderer.parse_response(response.sequences[0].tokens)

    if audio.is_reference:
        print(f"Reference:  {_REFERENCE_TRANSCRIPT}")
    print(f"Message format: {cfg.message_format}")
    print(f"Prediction: {get_text_content(message)}")
    print(f"Termination: {termination.value}")


def cli_main(cfg: Config) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    cli_main(chz.entrypoint(Config))
