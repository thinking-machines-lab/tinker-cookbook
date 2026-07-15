# Emotion + speech recognition (Expresso)

This recipe fine-tunes Inkling to classify speaking style and transcribe
speech, with the classic 2-stage SFT+RL setup. (New to audio input? Start
with the sampling example in [`../README.md`](../README.md).)

## Overview

The expected answer format is the speaking style and the transcript in one line:
```
[<style>] <transcription>
```

We perform the classic 2-stage SFT+RL training with shared prompt, output format, parser, and evaluator:

1. **SFT** (`sl_train.py`) — cross-entropy on subset of data with 100 steps.
2. **RL** (`rl_train.py`) — GRPO-style clip groups with reward
   `0.5 * 1[style correct] + 0.5 * max(0, 1 - WER)`, warm-started from the
   SFT checkpoint.

Results:

| metric | Inkling | post-SFT | post-RL |
|---|---:|---:|---:|
| `test/emotion_accuracy` | 0.361 | 0.757 | **0.852** |
| `test/emotion_macro_f1` | 0.383 | 0.758 | **0.861** |
| `test/wer` | 0.095 | 0.050 | **0.044** |


> Note: the goal is to showcase how to SFT + RL audio model with tinker, we did not optimize the configuration nor tune hyper-parameters for optimal performance on this benchmark.

## Requirements

`tml_renderers` for Inkling rendering and `jiwer` for reward & eval
(installed by the `inkling` and `audio` extras:
`pip install "tinker_cookbook[audio,inkling]"`); `ffmpeg` and `curl` for data
preparation.

## Data Preparation

[Expresso](https://arxiv.org/abs/2308.05725) (CC BY-NC 4.0) is a 48 kHz
expressive speech dataset. Only its `read` subset has transcriptions, so this
recipe uses the read "base" subset: short utterances in 7 styles —
`confused`, `default`, `enunciated`, `happy`, `laughing`, `sad`, `whisper`.

To download and prepare the dataset:

```bash
uv run python -m tinker_cookbook.recipes.audio.emotion.prepare_data data_path=<data-dir>
```

This downloads/extracts the raw dataset into `<data-dir>/expresso` and
produces the subsets `sft_train`, `rl_train`, `dev`, and `test` under
`<data-dir>/expresso_16khz` — pass the latter to the recipes as
`data_dir=<data-dir>/expresso_16khz`.

## Stage 1: SFT

`sl_train.py` fine-tunes on `[<style>] <transcription>` targets with the stop
token as a loss target, note that in this stage the model also learns to answer immediately
instead of spending its budget thinking.

```bash
uv run python -m tinker_cookbook.recipes.audio.emotion.sl_train
```

## Stage 2: RL

Warm-start from a SFT checkpoint:

```bash
uv run python -m tinker_cookbook.recipes.audio.emotion.rl_train \
    load_checkpoint_path="tinker://<sft-run-id>/weights/<step>"
```


Each training batch rolls out `group_size` samples per clip and advantages
are centered within the group. There is no format-compliance reward term since
the base model already emits well-formed responses.
Rewards are emotion classification accuracy and transcription word error rate.


## Evaluate

`evaluate.py` runs the shared evaluator on the held-out test split against
any sampler — use it for before/after comparisons:

```bash
# Before: the base model.
uv run python -m tinker_cookbook.recipes.audio.emotion.evaluate \
    log_path=/tmp/audio-rl-eval-before

# After: a sampler checkpoint saved by sl_train.py or rl_train.py (see checkpoints.jsonl).
uv run python -m tinker_cookbook.recipes.audio.emotion.evaluate \
    log_path=/tmp/audio-rl-eval-after \
    model_path="tinker://<run-id>/sampler_weights/<step>"
```

Each run prints `test/emotion_accuracy`, `test/emotion_macro_f1`,
`test/wer`, and `test/format_valid`, writes them to
`<log_path>/metrics.json`, and dumps per-clip predictions to
`<log_path>/eval_rollouts_test_000.jsonl` for error analysis.

## Files

- `env.py` — the task in one module: prompt, manifest-based clip loading,
  `[<style>] <transcription>` parsing/scoring, and the RL
  `Env` / `EnvGroupBuilder` / `RLDataset` with the combined reward
  (WER + response decoding come from the shared [`../grading.py`](../grading.py))
- `env_test.py` — unit tests for the parsing/scoring layer (no network)
- `sl_train.py` — Stage 1: supervised on `[<style>] <transcription>`
- `rl_train.py` — Stage 2: RL training entrypoint (wires `rl.train`)
- `evaluate.py` — shared evaluator (in-training + standalone test-split CLI)
- `prepare_data.py` — one-time data prep: download/extract, 16 kHz transcode,
  SFT/RL train subsets, per-split manifests
