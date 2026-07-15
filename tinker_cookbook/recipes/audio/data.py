"""Shared data plumbing for the ASR-style audio recipes (``asr/``,
``medical_asr/``): native-message prompt/conversation builders, supervised
datum construction, the reshuffled SFT dataset, and the local WAV cache.

Conversations are built from native ``tml_renderers.chat.Message`` objects
(accepted directly by the renderer) because audio has no cookbook/OpenAI dict
shape yet. Each model turn ends with a ``ModelEndSampling`` message carrying
loss weight, so SFT trains the stop token the sampler relies on.

``AudioPointer`` needs a local file path, so recipes decode HF audio rows once
into a WAV cache via ``cache_wav``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import tinker

from tinker_cookbook.renderers import Renderer
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import SupervisedDataset

if TYPE_CHECKING:
    from tml_renderers import chat  # pyright: ignore[reportMissingImports]

ASR_INSTRUCT_PROMPT = (
    "Transcribe the speech in the audio. Respond with only the transcription "
    "text and nothing else -- no preamble, no labels, no quotation marks, no "
    "explanation, and no markdown."
)


class Clip(TypedDict):
    """One cached utterance."""

    text: str
    path: str
    num_frames: int
    sample_rate: int


def prompt_messages(clip: Clip) -> list[chat.Message]:
    """The conditioning turn: instruction + AudioPointer (DMel-encoded at render time)."""
    from tml_renderers import chat  # pyright: ignore[reportMissingImports]

    user = chat.Author(chat.AuthorKind.User)
    return [
        chat.Message(content=chat.Text(ASR_INSTRUCT_PROMPT), author=user),
        chat.Message(
            content=chat.AudioPointer(
                location=clip["path"],
                format=chat.AudioFormat.Wav,
                num_frames=clip["num_frames"],
                sample_rate=clip["sample_rate"],
            ),
            author=user,
        ),
    ]


def asr_messages(clip: Clip) -> list[chat.Message]:
    """The full conversation. ``ModelEndSampling`` closes the model turn and is a
    loss target, so the model learns to emit its own stop token. The transcript's
    leading space matches the audio training data convention."""
    from tml_renderers import chat  # pyright: ignore[reportMissingImports]

    model = chat.Author(chat.AuthorKind.Model)
    return prompt_messages(clip) + [
        chat.Message(content=chat.Text(" " + clip["text"].strip()), author=model),
        # Either construct chat.OpenAIMessage turns (which append ModelEndSampling
        # automatically) or chat.Message lists with an explicit one. We do the latter.
        chat.Message(content=chat.ModelEndSampling(), author=model),
    ]


def clip_datum(renderer: Renderer, clip: Clip, max_length: int) -> tinker.Datum:
    """Render one clip's conversation into a supervised datum."""
    # Renderer returns unshifted (ModelInput, weights); the helper does the shift.
    model_input, weights = renderer.build_supervised_example(asr_messages(clip))
    return datum_from_model_input_weights(model_input, weights, max_length=max_length)


def cache_wav(path: Path, array: np.ndarray, sample_rate: int) -> None:
    """Write one decoded clip as a 16-bit PCM WAV (skips existing files)."""
    import soundfile as sf

    if not path.exists():
        sf.write(str(path), array, sample_rate, format="WAV", subtype="PCM_16")


class AudioASRDataset(SupervisedDataset):
    """A fixed list of ASR datums served in per-epoch reshuffled batches."""

    def __init__(self, datums: list[tinker.Datum], batch_size: int):
        self.datums = datums
        self.batch_size = batch_size
        self.order = list(range(len(datums)))

    def __len__(self) -> int:
        return len(self.datums) // self.batch_size  # drop-last

    def set_epoch(self, seed: int = 0):
        random.Random(seed).shuffle(self.order)

    def get_batch(self, index: int) -> list[tinker.Datum]:
        lo = index * self.batch_size
        return [self.datums[i] for i in self.order[lo : lo + self.batch_size]]
