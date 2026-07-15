"""Tests for the ASR recipe's data, SFT datum, env, and reward pipeline.

Skip when the optional ``tml_renderers`` package is not importable (install
the ``inkling`` extra), like ``renderers/tml_v0_test.py``.
"""

from __future__ import annotations

import asyncio
import math
import struct
import wave
from pathlib import Path
from typing import Any, cast

import pytest

from tinker_cookbook.recipes.audio.data import Clip


def _require_tml_renderers() -> Any:
    pytest.importorskip("tml_renderers")
    pytest.importorskip("tml_renderers.v0")
    pytest.importorskip("tml_renderers.tinker")
    return pytest.importorskip("tml_renderers.chat")


def _dmel_chunk_cls() -> type:
    import tinker

    cls = getattr(tinker.types, "DmelChunk", None)
    if cls is None:
        pytest.skip("installed tinker SDK predates tinker.types.DmelChunk")
    assert cls is not None
    return cls


def _renderer_and_tokenizer() -> tuple[Any, Any]:
    """The recipes' wiring: model name -> TML tokenizer adapter -> tml_v0."""
    _require_tml_renderers()
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer("thinkingmachines/Inkling")
    return get_renderer("tml_v0", tokenizer), tokenizer


def _write_test_wav(path: Path, seconds: float = 0.2, sample_rate: int = 16_000) -> int:
    """Write a mono 16-bit PCM sine wav; returns the number of frames."""
    num_frames = int(seconds * sample_rate)
    samples = (int(8000 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(num_frames))
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(b"".join(struct.pack("<h", s) for s in samples))
    return num_frames


def _test_clip(tmp_path: Path, text: str = "hello world") -> Clip:
    wav_path = tmp_path / "clip.wav"
    num_frames = _write_test_wav(wav_path)
    return Clip(
        text=text,
        path=str(wav_path),
        num_frames=num_frames,
        sample_rate=16_000,
    )


def _model_response_tokens(tokenizer: Any, text: str) -> list[int]:
    """A well-formed sampled stream: model text message closed by MES."""
    tml = tokenizer.tml_tokenizer
    return [
        int(tml.encode_special("message_model")),
        int(tml.encode_special("content_text")),
        *tml.encode_ordinary(text),
        int(tml.encode_special("end_message")),
        int(tml.encode_special("content_model_end_sampling")),
    ]


def test_asr_messages_are_native_public_types(tmp_path: Path) -> None:
    _require_tml_renderers()
    from tml_renderers import chat  # pyright: ignore[reportMissingImports]

    from tinker_cookbook.recipes.audio.data import asr_messages

    messages = asr_messages(_test_clip(tmp_path))

    assert len(messages) == 4
    assert isinstance(messages[0].content, chat.Text)
    assert isinstance(messages[1].content, chat.AudioPointer)
    assert messages[1].content.format == chat.AudioFormat.Wav
    assert isinstance(messages[2].content, chat.Text)
    assert isinstance(messages[3].content, chat.ModelEndSampling)
    assert messages[1].author.kind == chat.AuthorKind.User
    assert messages[3].author.kind == chat.AuthorKind.Model


def test_build_supervised_example_masks_audio_and_trains_transcript_and_stop(
    tmp_path: Path,
) -> None:
    dmel_chunk_cls = _dmel_chunk_cls()
    import tinker

    from tinker_cookbook.recipes.audio.data import asr_messages

    renderer, tokenizer = _renderer_and_tokenizer()
    model_input, weights = renderer.build_supervised_example(asr_messages(_test_clip(tmp_path)))

    dmel_chunks = [c for c in model_input.chunks if isinstance(c, dmel_chunk_cls)]
    assert len(dmel_chunks) == 1
    assert dmel_chunks[0].length > 0  # SDK header-based length

    total_len = sum(int(c.length) for c in model_input.chunks)
    assert total_len == len(weights)
    # Transcript + model framing + ModelEndSampling are targets; audio and user
    # text are conditioning only.
    assert 0.0 < float(weights.sum()) < len(weights)
    end_sampling = int(tokenizer.tml_tokenizer.encode_special("content_model_end_sampling"))
    last_chunk = model_input.chunks[-1]
    assert isinstance(last_chunk, tinker.types.EncodedTextChunk)
    assert last_chunk.tokens[-1] == end_sampling, "model turn ends with the stop token"
    assert float(weights[-1]) == 1.0, "the stop token is a loss target"

    # DMel positions are never targets.
    offset = 0
    for chunk in model_input.chunks:
        if isinstance(chunk, dmel_chunk_cls):
            assert float(weights[offset : offset + chunk.length].sum()) == 0.0
        offset += int(chunk.length)


def test_generation_prompt_ends_before_model_turn(tmp_path: Path) -> None:
    dmel_chunk_cls = _dmel_chunk_cls()
    import tinker

    from tinker_cookbook.recipes.audio.data import prompt_messages

    renderer, _tokenizer = _renderer_and_tokenizer()
    prompt = renderer.build_generation_prompt(prompt_messages(_test_clip(tmp_path)))

    assert any(isinstance(c, dmel_chunk_cls) for c in prompt.chunks)
    assert isinstance(prompt.chunks[-1], tinker.types.EncodedTextChunk)


def test_datum_pipeline_shifts_correctly(tmp_path: Path) -> None:
    from tinker_cookbook.recipes.audio.data import clip_datum

    renderer, _tokenizer = _renderer_and_tokenizer()

    datum = clip_datum(renderer, _test_clip(tmp_path), max_length=8192)

    targets = datum.loss_fn_inputs["target_tokens"].to_numpy()
    weights = datum.loss_fn_inputs["weights"].to_numpy()
    assert len(targets) == len(weights)
    assert datum.model_input.length == len(targets)  # inputs dropped the last token
    assert 0.0 < float(weights.sum()) < len(weights)


def test_group_builder_makes_group_of_single_clip_envs(tmp_path: Path) -> None:
    import tinker

    from tinker_cookbook.recipes.audio.asr.env import AudioASRGroupBuilder

    renderer, _tokenizer = _renderer_and_tokenizer()
    builder = AudioASRGroupBuilder(clip=_test_clip(tmp_path), renderer=renderer, group_size=3)

    envs = asyncio.run(builder.make_envs())
    assert len(envs) == 3

    result = asyncio.run(envs[0].initial_observation())
    assert isinstance(result, tuple), "a single short clip must not overflow the prompt"
    prompt, stop_condition = result
    assert stop_condition, "stop condition must be non-empty"
    # Prompt carries the audio and ends with text framing, before the model turn.
    assert prompt.length > 0
    assert isinstance(prompt.chunks[-1], tinker.types.EncodedTextChunk)


def test_step_rewards_perfect_and_garbage_transcripts(tmp_path: Path) -> None:
    pytest.importorskip("jiwer")
    from tinker_cookbook.recipes.audio.asr.env import AudioASREnv

    renderer, tokenizer = _renderer_and_tokenizer()
    clip = _test_clip(tmp_path, text="hello world")

    # Perfect transcript (leading space matches the SFT convention).
    env = AudioASREnv(clip, renderer)
    result = asyncio.run(env.step(_model_response_tokens(tokenizer, " hello world")))
    assert result.episode_done
    assert result.reward == 0.0
    assert result.metrics["wer"] == 0.0
    assert result.metrics["format"] == 1.0

    # Unframed ramble: MALFORMED parse, junk transcript, WER > 1 from
    # insertions -- the reward cap floors it at -1.
    env = AudioASREnv(clip, renderer)
    garbage = tokenizer.tml_tokenizer.encode_ordinary("complete and utter nonsense tokens")
    result = asyncio.run(env.step(list(garbage)))
    assert result.reward == -1.0
    assert result.metrics["wer"] > 1.0
    assert result.metrics["format"] == 0.0


def test_parse_response_text_handles_mid_character_truncation() -> None:
    """A rollout can end mid-character (byte-level BPE); grading must not crash."""
    from tinker_cookbook.recipes.audio.grading import parse_response_text
    from tinker_cookbook.renderers import ParseTermination

    renderer, tokenizer = _renderer_and_tokenizer()
    crab = list(tokenizer.tml_tokenizer.encode_ordinary("🦀"))
    assert len(crab) > 1, "multi-byte char should span several byte-level tokens"

    hyp, termination = parse_response_text(renderer, crab[:-1])
    # Depending on the tml-renderers build, decoding a mid-character cut either
    # raises (graded as "") or decodes lossily to U+FFFD; neither may crash and
    # neither yields transcript text.
    assert hyp.strip("�") == ""
    assert termination == ParseTermination.MALFORMED


def test_dataset_serves_one_pass_over_clips(tmp_path: Path) -> None:
    from tinker_cookbook.recipes.audio.asr.env import AudioASRGroupBuilder, AudioASRRLDataset

    renderer, _tokenizer = _renderer_and_tokenizer()
    clips = [_test_clip(tmp_path, text=f"clip number {i}") for i in range(8)]
    dataset = AudioASRRLDataset(
        clips=clips,
        renderer=renderer,
        group_size=4,
        groups_per_batch=3,
    )

    def batch(index: int) -> list[AudioASRGroupBuilder]:
        return cast("list[AudioASRGroupBuilder]", list(dataset.get_batch(index)))

    assert len(dataset) == 3  # ceil(8 / 3), last batch partial
    assert [len(batch(i)) for i in range(3)] == [3, 3, 2]
    assert all(b.group_size == 4 for b in batch(0))
    # One pass: every clip exactly once, in dataset order.
    texts = [b.clip["text"] for i in range(3) for b in batch(i)]
    assert texts == [c["text"] for c in clips]
    with pytest.raises(AssertionError):
        dataset.get_batch(3)
