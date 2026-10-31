"""Tests for audio helpers."""

from __future__ import annotations

import pytest

from tinker_cookbook.audio_processing_utils import audio_part_to_input_audio
from tinker_cookbook.renderers import AudioPart


def test_cookbook_audio_non_wav_requires_complete_metadata() -> None:
    with pytest.raises(ValueError, match="must provide num_frames and sample_rate"):
        audio_part_to_input_audio(
            AudioPart(type="audio", audio=b"mp3", format="mp3", num_frames=48_000)
        )


def test_cookbook_audio_metadata_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        audio_part_to_input_audio(
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
        audio_part_to_input_audio(part)
