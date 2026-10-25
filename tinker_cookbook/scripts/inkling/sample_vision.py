"""Sample Inkling with an image using OpenAI-compatible or native chat messages.

OpenAI-compatible input (the default):

    uv run python -m tinker_cookbook.scripts.inkling.sample_vision

Native ``tml_renderers.chat`` input:

    uv run python -m tinker_cookbook.scripts.inkling.sample_vision message_format=chat

Provide a local PNG or JPEG with ``image_path=/path/to/image.png``.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Literal, TypedDict, cast

import chz
import tinker
from PIL import Image

from tinker_cookbook import model_info
from tinker_cookbook.renderers import Message, get_renderer, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer

_DATASET_ID = "dpdl-benchmark/caltech101"
_SAMPLE_LABEL = "stop_sign"
_SAMPLE_PATH = Path.home() / ".cache" / "tinker-cookbook" / "caltech101-stop-sign.png"
_REFERENCE_ANSWER = "Stop sign"
_DEFAULT_PROMPT = "What object is shown in this image? Answer with a short noun phrase."

ImageFormat = Literal["PNG", "JPEG"]
MessageFormat = Literal["openai", "chat"]


@dataclass(frozen=True)
class ImageInput:
    path: Path
    format: ImageFormat
    width: int
    height: int
    is_reference: bool


class _DatasetRow(TypedDict):
    image: object
    label: int


def _download_sample() -> Path:
    if not _SAMPLE_PATH.exists():
        from datasets import Dataset, load_dataset

        dataset = load_dataset(_DATASET_ID, split="test")
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected a Dataset from {_DATASET_ID!r}")
        label_feature = dataset.features["label"]
        rows = cast(Iterable[_DatasetRow], dataset)
        row = next(row for row in rows if label_feature.int2str(row["label"]) == _SAMPLE_LABEL)
        image = row["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected an image from {_DATASET_ID!r}")
        _SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(_SAMPLE_PATH, format="PNG")
    return _SAMPLE_PATH


def _resolve_image_input(configured_path: str | None) -> ImageInput:
    is_reference = configured_path is None
    path = _download_sample() if configured_path is None else Path(configured_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")

    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format
    if image_format not in {"PNG", "JPEG"}:
        raise ValueError(f"Unsupported image format {image_format!r}; provide a PNG or JPEG image.")

    return ImageInput(
        path=path,
        format=cast(ImageFormat, image_format),
        width=width,
        height=height,
        is_reference=is_reference,
    )


def _build_openai_messages(image: ImageInput, prompt: str) -> list[Message]:
    chat = import_module("tml_renderers.chat")
    mime = "image/png" if image.format == "PNG" else "image/jpeg"
    image_base64 = base64.b64encode(image.path.read_bytes()).decode("ascii")
    return cast(
        list[Message],
        chat.OpenAIMessage.from_oss_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{image_base64}",
                            },
                        },
                    ],
                }
            ]
        ),
    )


def _build_chat_messages(image: ImageInput, prompt: str) -> list[Message]:
    chat = import_module("tml_renderers.chat")
    user = chat.Author(chat.AuthorKind.User)
    return cast(
        list[Message],
        chat.MessageList(
            [
                chat.Message(content=chat.Text(prompt), author=user),
                chat.Message(
                    content=chat.ImagePointer(
                        location=str(image.path),
                        format=(
                            chat.ImageFormat.Png if image.format == "PNG" else chat.ImageFormat.Jpeg
                        ),
                        width=image.width,
                        height=image.height,
                    ),
                    author=user,
                ),
            ]
        ),
    )


MessageBuilder = Callable[[ImageInput, str], list[Message]]
_MESSAGE_BUILDERS: dict[MessageFormat, MessageBuilder] = {
    "openai": _build_openai_messages,
    "chat": _build_chat_messages,
}


@chz.chz
class Config:
    image_path: str | None = chz.field(
        default=None,
        doc="Optional local PNG or JPEG; defaults to a public Caltech101 image.",
    )
    prompt: str = chz.field(default=_DEFAULT_PROMPT, doc="Question to ask about the image.")
    model_name: str = chz.field(
        default="thinkingmachines/Inkling",
        doc="Vision-language Tinker base model.",
    )
    base_url: str | None = chz.field(default=None, doc="Optional Tinker service URL.")
    message_format: MessageFormat = chz.field(
        default="openai",
        doc="Message representation to render: OpenAI-compatible dictionaries or native chat types.",
    )
    max_tokens: int = chz.field(default=128, doc="Maximum sampled response tokens.")
    temperature: float = chz.field(default=1.0, doc="Sampling temperature.")


async def async_main(cfg: Config) -> None:
    if not model_info.get_model_attributes(cfg.model_name).is_vl:
        raise ValueError(f"Vision input is not supported by {cfg.model_name!r}.")

    image = _resolve_image_input(cfg.image_path)
    messages = _MESSAGE_BUILDERS[cfg.message_format](image, cfg.prompt)

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

    if image.is_reference:
        print(f"Reference:  {_REFERENCE_ANSWER}")
    print(f"Message format: {cfg.message_format}")
    print(f"Prediction: {get_text_content(message)}")
    print(f"Termination: {termination.value}")


def cli_main(cfg: Config) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    cli_main(chz.entrypoint(Config))
