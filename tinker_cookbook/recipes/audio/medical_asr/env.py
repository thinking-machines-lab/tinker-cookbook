"""EkaCare medical-ASR data, datasets, and the WER evaluator for a full SFT run.

Mirrors the LibriSpeech ``asr/`` recipe but on the EkaCare medical-ASR benchmark
(``ekacare/eka-medical-asr-evaluation-dataset``, ``en`` config): Indian-accented
English dictated prescriptions/consultations, dense with drug names and dosages.

The dataset ships a single ``test`` split, so we carve our own train/eval by a
seeded **random 80/20 shuffle** (``split=random_speaker_adapted``): speakers
appear on both sides, so this measures speaker-adapted WER, not unseen-speaker
generalization -- and the ~8% duplicate transcripts mean the number is
optimistic. It is the fast in-distribution signal for hill-climbing, not a
generalization claim.

Rendering, datum construction, and WER scoring are reused from the shared
``audio/data.py`` / ``audio/grading.py``; only the data source (embedded-audio
parquet + a medical-entity keyword-error metric) differs.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, cast

import chz
import datasets
import numpy as np
import tinker

from tinker_cookbook import model_info
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
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

_HF_DATASET = "ekacare/eka-medical-asr-evaluation-dataset"
_HF_CONFIG = "en"
_HF_SPLIT = "test"
_TRAIN_FRACTION = 0.8


def _audio_renderer(model_name: str) -> Renderer:
    if not model_info.get_model_attributes(model_name).is_audio_in:
        raise ValueError(f"Audio input is not supported by {model_name!r}; use Inkling.")
    return get_renderer(
        model_info.get_recommended_renderer_name(model_name), get_tokenizer(model_name)
    )


class MedClip(Clip):
    """An ASR ``Clip`` plus the reference's medical-entity phrases.

    ``entities`` are the surface strings annotated in ``medical_entities``
    (drugs / clinical_findings / advices / ...), used for keyword error rate.
    ``prompt_messages`` / ``asr_messages`` read only the base Clip fields, so a
    MedClip is a drop-in for the shared renderers.
    """

    entities: list[str]


def _parse_entities(medical_entities: str | None) -> list[str]:
    """Extract entity surface strings from the ``medical_entities`` JSON.

    Format: ``[[text, category, [[start, end], ...]], ...]``. Defensive: any
    parse failure yields no entities rather than crashing the run.
    """
    if not medical_entities:
        return []
    try:
        parsed = json.loads(medical_entities)
    except (json.JSONDecodeError, TypeError):
        return []
    out: list[str] = []
    for item in parsed:
        if isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
            out.append(item[0])
    return out


@functools.cache
def load_ekacare_split(
    audio_cache_dir: str, seed: int
) -> tuple[tuple[MedClip, ...], tuple[MedClip, ...]]:
    """Load the full EkaCare ``en`` set, cache each clip's audio as a local WAV,
    and return a seeded random 80/20 (train, eval) split.

    ``@functools.cache`` so the dataset builder and the WER evaluator share one
    load + one split for a given (cache dir, seed).
    """
    wav_dir = Path(audio_cache_dir).expanduser() / "ekacare_en"
    wav_dir.mkdir(parents=True, exist_ok=True)
    ds = datasets.load_dataset(_HF_DATASET, _HF_CONFIG, split=_HF_SPLIT)

    clips: list[MedClip] = []
    for row in cast("list[dict[str, Any]]", ds):
        audio = row["audio"]
        array = np.asarray(audio["array"], np.float32)
        sample_rate = int(audio["sampling_rate"])
        # md5_audio is unique per clip and stable across runs/seeds.
        key = row.get("md5_audio") or row.get("file_name") or f"row{len(clips)}"
        path = wav_dir / f"{key}.wav"
        cache_wav(path, array, sample_rate)
        clips.append(
            MedClip(
                text=str(row["text"]),
                path=str(path),
                num_frames=len(array),
                sample_rate=sample_rate,
                entities=_parse_entities(row.get("medical_entities")),
            )
        )

    order = list(range(len(clips)))
    random.Random(seed).shuffle(order)
    n_train = int(_TRAIN_FRACTION * len(clips))
    train = tuple(clips[i] for i in order[:n_train])
    eval_ = tuple(clips[i] for i in order[n_train:])
    logger.info(
        "EkaCare en: %d clips -> %d train / %d eval (seed=%d, random 80/20)",
        len(clips),
        len(train),
        len(eval_),
        seed,
    )
    return train, eval_


@chz.chz
class EkacareASRDatasetBuilder(SupervisedDatasetBuilder):
    model_name: str
    batch_size: int
    max_length: int
    seed: int
    audio_cache_dir: str

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        renderer = _audio_renderer(self.model_name)
        train_clips, eval_clips = load_ekacare_split(self.audio_cache_dir, self.seed)
        train_ds = AudioASRDataset(
            [clip_datum(renderer, c, self.max_length) for c in train_clips], self.batch_size
        )
        eval_ds = AudioASRDataset(
            [clip_datum(renderer, c, self.max_length) for c in eval_clips], self.batch_size
        )
        return train_ds, eval_ds


_ENTITY_HIT_TOLERANCE = 0.15  # CER at/below this counts the entity as correctly transcribed


def _norm_medical(text: str) -> str:
    """Normalization for entity matching, tighter than ``normalize_text``.

    Collapses number/unit spacing (``40 mg`` -> ``40mg``) and strips possessives
    (``disease's`` -> ``disease``) so pure-formatting differences don't count as
    entity errors, then applies the shared lower/punctuation/whitespace pass.
    """
    text = text.lower()
    text = re.sub(r"(\d)\s+(mg|mcg|ml|g|%|iu|units?)\b", r"\1\2", text)
    text = re.sub(r"\b(\w+?)'s\b", r"\1", text)
    return normalize_text(text)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _entity_cer(entity_norm: str, hyp_norm: str) -> float:
    """Character error rate of one reference entity against its best-matching
    window of the hypothesis (capped at 1.0).

    Graded, unlike a substring hit/miss: ``40 mg`` vs ``40mg`` scores ~0 (after
    ``_norm_medical``), a one-letter drug-name slip scores small, a wrong term
    scores ~1.0. Windows of the entity's word-length +/-1 are tried to absorb
    boundary slop; the metric is recall-oriented (it does not penalize
    hypothesis-only inserted entities, which would need NER on the output)."""
    e = entity_norm
    hyp_words = hyp_norm.split()
    if not e:
        return 0.0
    if not hyp_words:
        return 1.0
    n = len(e.split())
    best = 1.0
    for width in (n - 1, n, n + 1):
        if width < 1:
            continue
        for i in range(max(1, len(hyp_words) - width + 1)):
            window = " ".join(hyp_words[i : i + width])
            best = min(best, _levenshtein(e, window) / len(e))
            if best == 0.0:
                return 0.0
    return min(best, 1.0)


def _entity_scores(entities: list[str], hyp: str) -> tuple[float, int, int, int]:
    """(length-weighted CER sum, entity char-length sum, #hits, #entities) for one clip.

    A hit is CER <= ``_ENTITY_HIT_TOLERANCE``. Corpus ``test/entity_cer`` is the
    length-weighted mean (weighted_sum / length_sum); ``test/entity_hit_rate``
    is hits / entities.
    """
    hyp_norm = _norm_medical(hyp)
    weighted = 0.0
    length = hits = count = 0
    for raw in entities:
        e = _norm_medical(raw)
        if not e:
            continue
        count += 1
        cer = _entity_cer(e, hyp_norm)
        weighted += cer * len(e)
        length += len(e)
        if cer <= _ENTITY_HIT_TOLERANCE:
            hits += 1
    return weighted, length, hits, count


class EkacareWEREvaluator(SamplingClientEvaluator):
    """Samples transcriptions on the held-out EkaCare clips and reports corpus
    WER (``test/wer``) plus medical-entity metrics over the reference's annotated
    drug/finding/advice phrases: ``test/entity_cer`` (length-weighted character
    error rate of each entity vs its best hypothesis window) and
    ``test/entity_hit_rate`` (fraction transcribed within tolerance)."""

    def __init__(self, config: EkacareWEREvaluatorBuilder):
        self.config = config
        self.renderer = _audio_renderer(config.model_name)
        _train, eval_clips = load_ekacare_split(config.audio_cache_dir, config.seed)
        self.eval_clips: list[MedClip] = list(eval_clips)
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
            # A garbled sample (e.g. truncated mid-multibyte-character) parses as
            # MALFORMED with whatever text decodes; see grading.py.
            return parse_response_text(self.renderer, list(resp.sequences[0].tokens))

        outs = await asyncio.gather(*[transcribe(p) for p in self.eval_prompts])
        refs = [c["text"] for c in self.eval_clips]
        hyps = [o[0] for o in outs]

        cer_weighted = 0.0
        cer_length = ent_hits = ent_count = 0
        rollouts: list[dict[str, str | float]] = []
        for clip, ref, hyp, (_, term) in zip(self.eval_clips, refs, hyps, outs, strict=True):
            weighted, length, hits, count = _entity_scores(clip["entities"], hyp)
            cer_weighted += weighted
            cer_length += length
            ent_hits += hits
            ent_count += count
            rollouts.append(
                {
                    "ref": ref,
                    "hyp": hyp,
                    "ref_norm": normalize_text(ref),
                    "hyp_norm": normalize_text(hyp),
                    "wer": clip_wer(ref, hyp),
                    "entities": json.dumps(clip["entities"]),
                    "entity_cer": (weighted / length) if length else 0.0,
                    "entity_hits": hits,
                    "entities_total": count,
                    "termination": str(term),
                }
            )
        self._save_rollouts(rollouts)
        metrics = {"test/wer": corpus_wer(refs, hyps)}
        if cer_length:
            metrics["test/entity_cer"] = cer_weighted / cer_length
        if ent_count:
            metrics["test/entity_hit_rate"] = ent_hits / ent_count
        return metrics

    def _save_rollouts(self, rollouts: list[dict[str, str | float]]) -> None:
        path = Path(self.config.log_path).expanduser() / f"eval_rollouts_{self._n_calls:03d}.jsonl"
        self._n_calls += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rollouts) + "\n")
        logger.info("Wrote %d eval rollouts -> %s", len(rollouts), path)


@chz.chz
class EkacareWEREvaluatorBuilder:
    model_name: str
    log_path: str
    seed: int
    audio_cache_dir: str
    max_tokens: int = 8192
    temperature: float = 1.0
    max_parallel_tasks: int = 128

    def __call__(self) -> EkacareWEREvaluator:
        return EkacareWEREvaluator(self)
