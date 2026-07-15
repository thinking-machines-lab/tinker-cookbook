"""Expresso data + RL environment for emotion classification + transcription.

The model's job is to emit ``[<style>] <transcription>`` for an expressive
speech clip. This module holds everything task-specific -- the prompt,
manifest-based clip loading, output parsing/scoring, and the RL environment
-- shared by ``sl_train.py``, ``rl_train.py``, and ``evaluate.py``. Everything
dataset-specific (download, 16 kHz transcode, split subsetting) happens once
in ``prepare_data.py``; at runtime a split is just a JSONL manifest of
ready-to-use clips.

RL episodes are single-turn: the observation is an instruction + audio clip,
the action is one sampled response, and the reward combines two terms:

    reward = emotion_coef * 1[style correct] + wer_coef * max(0, 1 - WER)

There is no format term: the base model already emits well-formed
``[<style>] <transcription>`` responses (see README), and an unparseable
response scores zero on both terms anyway. Format compliance is still
tracked as a metric. Group advantage centering (GRPO-style) happens in the
trainer.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import chz
import tinker

from tinker_cookbook.completers import StopCondition
from tinker_cookbook.recipes.audio.data import audio_renderer
from tinker_cookbook.recipes.audio.grading import clip_wer, parse_response_text
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
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
from tinker_cookbook.utils import logtree

if TYPE_CHECKING:
    from tml_renderers import chat  # pyright: ignore[reportMissingImports]

DEFAULT_DATA_DIR = "/tmp/tinker-examples/audio-data/expresso_16khz"

# Read-speech styles that have short "base" recordings with transcriptions.
# (narration exists only as longform; singing has no transcripts.)
STYLES = ("confused", "default", "enunciated", "happy", "laughing", "sad", "whisper")

# sft_train and rl_train are clip-disjoint subsets of the official train
# split (see prepare_data.py); dev and test are the official splits.
Split = Literal["sft_train", "rl_train", "dev", "test"]

INSTRUCT_PROMPT = (
    "Listen to the audio, then classify the speaking style and transcribe the "
    f"speech. The style must be one of: {', '.join(STYLES)}. Respond with a "
    "single line in exactly this format: [<style>] <transcription>. Output "
    "nothing else -- no preamble, no explanation, no markdown."
)

# `[<style>] <transcription>`; the transcription may span multiple lines.
_PREDICTION_RE = re.compile(r"\A\s*\[\s*([a-z_]+)\s*\]\s*(.*)\Z", re.IGNORECASE | re.DOTALL)


class Clip(TypedDict):
    """One Expresso read-speech utterance (one manifest line)."""

    id: str
    text: str
    emotion: str
    path: str
    num_frames: int
    sample_rate: int


def load_clips(data_dir: str, split: Split, n: int | None, seed: int) -> tuple[Clip, ...]:
    """Load one split's manifest, shuffled; ``n`` clips (``None`` = all)."""
    manifest = Path(data_dir).expanduser() / f"{split}.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} not found. Prepare the dataset first: uv run python -m "
            f"tinker_cookbook.recipes.audio.emotion.prepare_data data_path=<dir>"
        )
    clips: list[Clip] = [json.loads(line) for line in manifest.open(encoding="utf-8")]
    for clip in clips:
        clip["path"] = str(manifest.parent / clip["path"])
    random.Random(seed).shuffle(clips)
    return tuple(clips[:n] if n is not None else clips)


def prompt_messages(clip: Clip) -> list[chat.Message]:
    """The conditioning turn: instruction + AudioPointer (DMel-encoded at render time)."""
    from tml_renderers import chat  # pyright: ignore[reportMissingImports]

    user = chat.Author(chat.AuthorKind.User)
    return [
        chat.Message(content=chat.Text(INSTRUCT_PROMPT), author=user),
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


def parse_prediction(text: str) -> tuple[str, str] | None:
    """Parse ``[<style>] <transcription>`` into (style, transcription), or None."""
    match = _PREDICTION_RE.match(text)
    if match is None:
        return None
    return match.group(1).lower(), match.group(2).strip()


class ScoredResponse(TypedDict):
    """One model response graded against its clip (shared by the RL reward
    and the evaluator, so the two cannot drift)."""

    pred_emotion: str
    hyp: str
    format_valid: bool
    emotion_correct: bool
    wer: float


def score_response(clip: Clip, response: str) -> ScoredResponse:
    """Grade one response: parse ``[<style>] <transcription>`` (unparseable
    falls back to ``("", raw text)``), exact-match the style, per-clip WER."""
    parsed = parse_prediction(response)
    pred_emotion, hyp = parsed if parsed is not None else ("", response)
    return ScoredResponse(
        pred_emotion=pred_emotion,
        hyp=hyp,
        format_valid=parsed is not None,
        emotion_correct=pred_emotion == clip["emotion"],
        wer=clip_wer(clip["text"], hyp),
    )


class ExpressoEnv(Env):
    """One rollout on one clip. The prompt is pre-rendered by the group builder."""

    def __init__(
        self,
        clip: Clip,
        renderer: Renderer,
        prompt: tinker.ModelInput,
        emotion_coef: float,
        wer_coef: float,
    ):
        self.clip = clip
        self.renderer = renderer
        self.prompt = prompt
        self.emotion_coef = emotion_coef
        self.wer_coef = wer_coef

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return self.prompt, self.renderer.get_stop_sequences()

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        response, termination = parse_response_text(self.renderer, action)
        scored = score_response(self.clip, response)
        # The format *metric* additionally requires a clean stop-sequence stop;
        # the task rewards only care whether the response parsed.
        format_ok = float(termination.is_stop_sequence and scored["format_valid"])
        emotion_correct = float(scored["emotion_correct"])
        wer_reward = max(0.0, 1.0 - scored["wer"]) if scored["format_valid"] else 0.0

        reward = self.emotion_coef * emotion_correct + self.wer_coef * wer_reward

        with logtree.scope_header("Clip"):
            logtree.table_from_dict(
                {
                    "id": self.clip["id"],
                    "emotion": self.clip["emotion"],
                    "reference": self.clip["text"],
                }
            )
        with logtree.scope_header("Policy Response"):
            logtree.log_text(response)
        with logtree.scope_header("Reward"):
            logtree.table_from_dict(
                {
                    "format_valid": bool(format_ok),
                    "emotion_correct": bool(emotion_correct),
                    "wer": f"{scored['wer']:.3f}" if scored["format_valid"] else "n/a",
                    "reward": f"{reward:.3f}",
                },
                caption="Reward components",
            )

        metrics: dict[str, float | int] = {
            "format": format_ok,
            "emotion_correct": emotion_correct,
            "wer_reward": wer_reward,
        }
        if scored["format_valid"]:
            metrics["wer"] = scored["wer"]
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],  # single-turn: episode is always done
            metrics=metrics,
        )


@dataclass(frozen=True)
class ExpressoGroupBuilder(EnvGroupBuilder):
    """Builds ``num_envs`` rollouts of the same clip (the GRPO group)."""

    clip: Clip
    renderer: TmlV0Renderer
    num_envs: int
    emotion_coef: float
    wer_coef: float

    async def make_envs(self) -> Sequence[Env]:
        # Render once per group: DMel-encoding the audio is the expensive part,
        # and all envs in the group share the same prompt.
        prompt = self.renderer.build_generation_prompt(prompt_messages(self.clip))
        return [
            ExpressoEnv(
                clip=self.clip,
                renderer=self.renderer,
                prompt=prompt,
                emotion_coef=self.emotion_coef,
                wer_coef=self.wer_coef,
            )
            for _ in range(self.num_envs)
        ]

    def logging_tags(self) -> list[str]:
        # Per-style training metrics land under env/<style>/.
        return ["expresso", self.clip["emotion"]]


class ExpressoRLDataset(RLDataset):
    """One pass over the clips; each batch is ``batch_size`` clip groups."""

    def __init__(
        self,
        clips: Sequence[Clip],
        batch_size: int,
        group_size: int,
        renderer: TmlV0Renderer,
        emotion_coef: float,
        wer_coef: float,
    ):
        self.clips = list(clips)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.emotion_coef = emotion_coef
        self.wer_coef = wer_coef

    def __len__(self) -> int:
        return len(self.clips) // self.batch_size  # drop-last

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        lo = index * self.batch_size
        return [
            ExpressoGroupBuilder(
                clip=clip,
                renderer=self.renderer,
                num_envs=self.group_size,
                emotion_coef=self.emotion_coef,
                wer_coef=self.wer_coef,
            )
            for clip in self.clips[lo : lo + self.batch_size]
        ]


@chz.chz
class ExpressoRLDatasetBuilder(RLDatasetBuilder):
    model_name: str
    data_dir: str
    batch_size: int  # clip groups per training batch
    group_size: int  # rollouts per clip
    n_train: int | None  # None = the full split (800 rl_train clips)
    seed: int
    emotion_coef: float
    wer_coef: float
    # rl_train = the train clips the SFT stage does not use (see prepare_data.py),
    # so warm-started RL trains on data disjoint from the SFT stage.
    split: Split = "rl_train"

    async def __call__(self) -> tuple[ExpressoRLDataset, None]:
        renderer = audio_renderer(self.model_name)
        clips = load_clips(self.data_dir, self.split, self.n_train, self.seed)
        # Test metrics come from the ExpressoEvaluator (corpus WER, saved
        # rollouts), not from a test RLDataset.
        return ExpressoRLDataset(
            clips=clips,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            emotion_coef=self.emotion_coef,
            wer_coef=self.wer_coef,
        ), None
