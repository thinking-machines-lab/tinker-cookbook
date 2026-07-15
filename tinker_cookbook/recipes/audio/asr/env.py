"""LibriSpeech data, SL/RL datasets, envs, and the WER evaluator shared by the
ASR recipes (``sl_train``, ``rl_train``).

Message/datum construction and the reshuffled SFT dataset come from the shared
``audio/data.py``; response decoding and WER scoring from ``audio/grading.py``.
This module holds what is LibriSpeech-specific: streaming clips into the local
WAV cache, the dataset builders, the ``-WER`` RL environment, and the evaluator.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import chz
import datasets
import numpy as np
import tinker

from tinker_cookbook import model_info
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.recipes.audio.data import (
    AudioASRDataset,
    Clip,
    cache_wav,
    clip_datum,
    prompt_messages,
)
from tinker_cookbook.recipes.audio.grading import (
    clip_wer,
    corpus_wer,
    normalize_text,
    parse_response_text,
)
from tinker_cookbook.renderers import ParseTermination, Renderer, get_renderer
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

logger = logging.getLogger(__name__)


def _audio_renderer(model_name: str) -> Renderer:
    if not model_info.get_model_attributes(model_name).is_audio_in:
        raise ValueError(f"Audio input is not supported by {model_name!r}; use Inkling.")
    return get_renderer(
        model_info.get_recommended_renderer_name(model_name), get_tokenizer(model_name)
    )


@functools.cache
def load_clips(
    audio_cache_dir: str, split: str, n: int, seed: int, shuffle_buffer_size: int
) -> tuple[Clip, ...]:
    """Stream LibriSpeech clips into a local WAV cache (dataset builders and WER
    evaluator share one load via the cache)."""
    # AudioPointer needs a local file path, so cache each streamed clip as a WAV.
    wav_dir = Path(audio_cache_dir).expanduser() / split.replace(".", "_")
    wav_dir.mkdir(parents=True, exist_ok=True)
    ds = datasets.load_dataset("openslr/librispeech_asr", "other", split=split, streaming=True)
    ds = cast(datasets.IterableDataset, ds)
    # LibriSpeech is ordered by speaker; shuffle before taking a subset.
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    clips: list[Clip] = []
    for r in ds.take(n):
        audio = r["audio"]
        array = np.asarray(audio["array"], np.float32)
        # Key by utterance id so changing seed/n never reuses a stale file.
        path = wav_dir / f"{r['id']}.wav"
        cache_wav(path, array, int(audio["sampling_rate"]))
        clips.append(
            Clip(
                text=str(r["text"]),
                path=str(path),
                num_frames=len(array),
                sample_rate=int(audio["sampling_rate"]),
            )
        )
    return tuple(clips)


@chz.chz
class AudioASRDatasetBuilder(SupervisedDatasetBuilder):
    model_name: str
    batch_size: int
    n_train: int
    n_eval: int
    max_length: int
    seed: int
    shuffle_buffer_size: int
    audio_cache_dir: str

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        renderer = _audio_renderer(self.model_name)
        train_clips = load_clips(
            self.audio_cache_dir, "train.500", self.n_train, self.seed, self.shuffle_buffer_size
        )
        train_ds = AudioASRDataset(
            [clip_datum(renderer, c, self.max_length) for c in train_clips], self.batch_size
        )
        if not self.n_eval:
            return train_ds, None
        # The eval dataset feeds the auto-added NLLEvaluator ("test/nll").
        eval_clips = load_clips(
            self.audio_cache_dir, "validation", self.n_eval, self.seed, self.shuffle_buffer_size
        )
        eval_ds = AudioASRDataset(
            [clip_datum(renderer, c, self.max_length) for c in eval_clips], self.batch_size
        )
        return train_ds, eval_ds


class WEREvaluator(SamplingClientEvaluator):
    """Samples transcriptions on held-out clips and reports corpus WER
    (test/wer) over the parsed transcripts."""

    def __init__(self, config: WEREvaluatorBuilder):
        self.config = config
        self.renderer = _audio_renderer(config.model_name)
        self.eval_clips = list(
            load_clips(
                config.audio_cache_dir,
                "validation",
                config.n_eval,
                config.seed,
                config.shuffle_buffer_size,
            )
        )
        # Pre-render eval prompts once; DMel encoding is CPU work.
        self.eval_prompts = [
            self.renderer.build_generation_prompt(prompt_messages(c)) for c in self.eval_clips
        ]
        self._n_calls = 0

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        params = tinker.SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stop=self.renderer.get_stop_sequences(),
        )
        semaphore = asyncio.Semaphore(self.config.max_parallel_tasks)

        async def transcribe(prompt: tinker.ModelInput) -> tuple[str, ParseTermination]:
            async with semaphore:
                resp = await sampling_client.sample_async(
                    prompt=prompt, num_samples=1, sampling_params=params
                )
            return parse_response_text(self.renderer, list(resp.sequences[0].tokens))

        outs = await asyncio.gather(*[transcribe(p) for p in self.eval_prompts])
        refs = [c["text"] for c in self.eval_clips]
        hyps = [o[0] for o in outs]
        rollouts: list[dict[str, str | float]] = [
            {
                "ref": r,
                "hyp": h,
                "ref_norm": normalize_text(r),
                "hyp_norm": normalize_text(h),
                "wer": clip_wer(r, h),
                "termination": str(t),
            }
            for r, h, (_, t) in zip(refs, hyps, outs, strict=True)
        ]
        self._save_rollouts(rollouts)
        return {"test/wer": corpus_wer(refs, hyps)}

    def _save_rollouts(self, rollouts: list[dict[str, str | float]]) -> None:
        path = Path(self.config.log_path).expanduser() / f"eval_rollouts_{self._n_calls:03d}.jsonl"
        self._n_calls += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rollouts) + "\n")
        logger.info(f"Wrote {len(rollouts)} eval rollouts -> {path}")


@chz.chz
class WEREvaluatorBuilder:
    model_name: str
    log_path: str
    n_eval: int
    seed: int
    shuffle_buffer_size: int
    audio_cache_dir: str
    max_tokens: int = 8192
    temperature: float = 1.0
    max_parallel_tasks: int = 128

    def __call__(self) -> WEREvaluator:
        return WEREvaluator(self)


class AudioASREnv(Env):
    """Single-turn env: hear a clip, emit a transcript, get -WER as reward."""

    def __init__(self, clip: Clip, renderer: Renderer):
        self.clip = clip
        self.renderer = renderer

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        prompt = self.renderer.build_generation_prompt(prompt_messages(self.clip))
        return prompt, self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        hyp, termination = parse_response_text(self.renderer, action)
        wer = clip_wer(self.clip["text"], hyp)
        # Cap: a degenerate transcript (WER > 1 from insertions) must not
        # dominate group centering.
        reward = -min(wer, 1.0)

        logtree.table_from_dict(
            {
                "reference": self.clip["text"],
                "transcript": hyp,
                "termination": str(termination),
                "sampled_tokens": len(action),
                "wer": f"{wer:.3f}",
                "reward": f"{reward:.3f}",
            },
            caption="Transcription reward",
        )
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "wer": wer,
                "format": float(termination.is_stop_sequence),
            },
            logs={"transcript": hyp},
        )


@dataclass(frozen=True)
class AudioASRGroupBuilder(EnvGroupBuilder):
    """group_size rollouts of the same clip; GRPO centers rewards within it."""

    clip: Clip
    renderer: Renderer
    group_size: int

    async def make_envs(self) -> Sequence[Env]:
        return [AudioASREnv(self.clip, self.renderer) for _ in range(self.group_size)]

    def logging_tags(self) -> list[str]:
        return ["librispeech_asr"]


class AudioASRRLDataset(RLDataset):
    """One pass over the clip list (pre-shuffled by load_clips), like MathDataset."""

    def __init__(
        self,
        clips: Sequence[Clip],
        renderer: Renderer,
        group_size: int,
        groups_per_batch: int,
    ):
        self.clips = list(clips)
        self.renderer = renderer
        self.group_size = group_size
        self.groups_per_batch = groups_per_batch

    def __len__(self) -> int:
        return math.ceil(len(self.clips) / self.groups_per_batch)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        lo = index * self.groups_per_batch
        hi = min(lo + self.groups_per_batch, len(self.clips))
        assert lo < hi, "batch index past the end of the dataset"
        return [
            AudioASRGroupBuilder(clip, self.renderer, self.group_size) for clip in self.clips[lo:hi]
        ]


@chz.chz
class AudioASRRLDatasetBuilder(RLDatasetBuilder):
    model_name: str
    groups_per_batch: int
    group_size: int
    n_train: int
    seed: int
    shuffle_buffer_size: int
    audio_cache_dir: str

    async def __call__(self) -> tuple[AudioASRRLDataset, None]:
        renderer = _audio_renderer(self.model_name)
        clips = load_clips(
            self.audio_cache_dir, "train.500", self.n_train, self.seed, self.shuffle_buffer_size
        )
        return AudioASRRLDataset(
            clips=clips,
            renderer=renderer,
            group_size=self.group_size,
            groups_per_batch=self.groups_per_batch,
        ), None
