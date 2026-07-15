"""Unit tests for the Expresso task layer: output parsing, scoring, and
manifest loading. No network; WER-dependent tests skip without ``jiwer``,
and prompt construction skips without ``tml_renderers``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinker_cookbook.recipes.audio.emotion.env import (
    STYLES,
    Clip,
    load_clips,
    parse_prediction,
    prompt_messages,
    score_response,
)


def _clip(
    id: str = "ex04_happy_00001",
    path: str = "/tmp/does-not-matter.wav",
) -> Clip:
    return Clip(
        id=id,
        text="hello world",
        emotion="happy",
        path=path,
        num_frames=16_000,
        sample_rate=16_000,
    )


def test_parse_prediction_valid() -> None:
    assert parse_prediction("[happy] hello world") == ("happy", "hello world")


def test_parse_prediction_normalizes_case_and_whitespace() -> None:
    assert parse_prediction("  [ Sad ]   first\nsecond  ") == ("sad", "first\nsecond")


def test_parse_prediction_rejects_missing_or_malformed_tag() -> None:
    assert parse_prediction("hello world") is None
    assert parse_prediction("(happy) hello") is None
    assert parse_prediction("[bad-style!] hello") is None


def test_styles_are_the_seven_read_base_styles() -> None:
    assert len(STYLES) == 7
    assert "default" in STYLES
    assert all(style == style.lower() for style in STYLES)


def test_score_response_correct() -> None:
    pytest.importorskip("jiwer")
    scored = score_response(_clip(), "[happy] Hello, world!")
    assert scored["format_valid"]
    assert scored["emotion_correct"]
    assert scored["wer"] == 0.0


def test_score_response_wrong_style_right_transcript() -> None:
    pytest.importorskip("jiwer")
    scored = score_response(_clip(), "[sad] hello world")
    assert scored["format_valid"]
    assert not scored["emotion_correct"]
    assert scored["wer"] == 0.0


def test_score_response_unparseable_falls_back_to_raw_text() -> None:
    pytest.importorskip("jiwer")
    scored = score_response(_clip(), "I think the speaker said hello world")
    assert not scored["format_valid"]
    assert not scored["emotion_correct"]
    assert scored["pred_emotion"] == ""
    assert scored["hyp"] == "I think the speaker said hello world"


def test_load_clips_missing_manifest_points_at_prepare_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        load_clips(str(tmp_path), "dev", None, seed=0)


def test_load_clips_resolves_paths_and_shuffles_deterministically(tmp_path: Path) -> None:
    clips = [_clip(id=f"clip_{i}", path=f"wav/clip_{i}.wav") for i in range(5)]
    manifest = tmp_path / "dev.jsonl"
    manifest.write_text("".join(json.dumps(c) + "\n" for c in clips))

    loaded = load_clips(str(tmp_path), "dev", None, seed=0)
    assert len(loaded) == 5
    assert all(clip["path"] == str(tmp_path / "wav" / f"{clip['id']}.wav") for clip in loaded)
    assert [c["id"] for c in loaded] != [c["id"] for c in clips]  # seed-0 shuffle reorders 5 items
    assert load_clips(str(tmp_path), "dev", 2, seed=0) == loaded[:2]


def test_prompt_messages_carries_instruction_and_audio_pointer() -> None:
    chat = pytest.importorskip("tml_renderers.chat")
    clip = _clip()
    messages = prompt_messages(clip)
    assert len(messages) == 2
    assert isinstance(messages[0].content, chat.Text)
    pointer = messages[1].content
    assert isinstance(pointer, chat.AudioPointer)
    assert pointer.location == clip["path"]
    assert pointer.num_frames == clip["num_frames"]
    assert pointer.sample_rate == clip["sample_rate"]
