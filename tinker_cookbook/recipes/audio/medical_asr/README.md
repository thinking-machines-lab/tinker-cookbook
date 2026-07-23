# Medical ASR (EkaCare)

This recipe fine-tunes Inkling on specialist medical vocabulary — the
domain-adaptation companion to the [`../asr/`](../asr/) LibriSpeech recipe.
Where LibriSpeech shows the pipeline, this shows a distribution the base model
actually struggles with: zero-shot, roughly one in five annotated medical
terms is transcribed wrong.

## Overview

A single SFT stage teaches both the medical vocabulary and the dictation task
shape (short clip → emit just the term), roughly halving WER and entity
misses. It even transcribes complex drugs absent from the train split
(`oxcarbazepine`, `somatropin`) — generalized transcription behavior, not just
memorized vocabulary.

Results (12 epochs over the ~2,895-clip train split):

| metric | Inkling | post-SFT |
|---|---:|---:|
| `test/wer` | 0.157 | **0.072** |
| `test/entity_hit_rate` | 0.806 | **0.915** |
| `test/entity_cer` | 0.102 | **0.037** |

Concretely, `test/entity_hit_rate` means the base model gets 80.6% of the
complex medical terms (drug names, findings, dosages) right, and SFT improves
that to 91.5% — from missing one term in five to fewer than one in eleven.

## Requirements

`tml_renderers` for Inkling rendering, plus `jiwer` and `soundfile` for
grading & the audio cache (installed by the `inkling` and `audio` extras:
`pip install "tinker_cookbook[audio,inkling]"`).

## Data

The [EkaCare medical ASR evaluation dataset](https://huggingface.co/datasets/ekacare/eka-medical-asr-evaluation-dataset)
(MIT, `en` config) is Indian-accented English dictated prescriptions and
consultations, dense with drug names and dosages, with the medical-entity
phrases annotated per clip. It downloads from Hugging Face on first run — no
separate preparation step. The dataset ships a single split, so the recipe
carves a seeded random 80/20 train/eval split (speakers appear on both sides,
making this an in-distribution signal).

## Train

```bash
uv run python -m tinker_cookbook.recipes.audio.medical_asr.train \
    wandb_project=<project>
```

Defaults run all train clips for 12 epochs (~2,172 steps at batch 16, LoRA
rank 32, lr 1e-3 linear decay).

## Evaluate

The evaluator runs on the full ~724-clip held-out set every `eval_every`
steps (aligned with checkpointing, so the best-eval checkpoint can be picked
post-hoc from the WER curve) and reports:

- `test/wer` — standard corpus WER, shared with the ASR recipe,
- `test/entity_cer` — length-weighted character error rate of each annotated
  entity against its best-matching window of the hypothesis. Graded, unlike a
  substring hit/miss: `40 mg` vs `40mg` scores ~0, a one-letter drug-name slip
  scores small, a wrong term scores ~1,
- `test/entity_hit_rate` — fraction of entities transcribed within
  CER ≤ 0.15,
- `test/nll` — teacher-forced transcript NLL (auto-added from the eval
  dataset).

Per-clip predictions land in `<log_path>/eval_rollouts_*.jsonl` for error
analysis.

## Files

- `env.py` — embedded-audio parquet loading + local WAV cache, the seeded
  80/20 split, and the medical-entity metrics (message/datum plumbing comes
  from [`../data.py`](../data.py), WER from [`../grading.py`](../grading.py))
- `env_test.py` — network-free tests for entity parsing + the CER metric
- `train.py` — SFT entrypoint → `supervised.train`
