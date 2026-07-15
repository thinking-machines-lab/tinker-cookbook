# Speech recognition (LibriSpeech)

Minimal audio-ASR fine-tuning on public [LibriSpeech](https://openslr.org/12)
(CC BY 4.0), with the SL and RL paths sharing one domain layer (`env.py`). This is the
"hello world" of audio fine-tuning: it streams clips straight from Hugging
Face, so there is no separate data-preparation step.

## SFT

`sl_train.py` builds native `chat.Message` conversations — instruction,
`AudioPointer` (DMel-encoded at render time), transcript, and a **trained
`ModelEndSampling` stop token** — through `renderer.build_supervised_example`,
with the loop, checkpointing, resume, and epoch reshuffling from
`supervised.train`:

```bash
# Smoke run (64 clips, 5 epochs, a few minutes)
uv run python -m tinker_cookbook.recipes.audio.asr.sl_train

# Reference-scale run (512 clips / 201 steps, ~30 min)
uv run python -m tinker_cookbook.recipes.audio.asr.sl_train \
    n_train=512 n_eval=64 num_epochs=7 max_steps=201 eval_every=25 \
    batch_size=16 lora_rank=32
```

## RL

`rl_train.py` is GRPO-style on the same clips: `group_size` transcripts
sampled per clip at T=1.0 (within-group WER variance is the gradient signal),
per-rollout reward `-WER` capped at −1, one shuffled pass over `n_train`
clips. Warm-start from an SFT checkpoint via `load_checkpoint_path`, or start
from the base model to exercise format acquisition:

```bash
uv run python -m tinker_cookbook.recipes.audio.asr.rl_train \
    n_train=2048 n_eval=64 groups_per_batch=32 learning_rate=2e-5 \
    eval_every=4 save_every=8
```

## Grading

Every rollout grades as whatever its parsed message said (the same lenient
decode all audio recipes share — see [`../grading.py`](../grading.py)):
a truncated or degenerate rollout's junk inflates its own WER, and the RL
*reward* is capped at −1 so it can't dominate GRPO group centering either.
Clean stop-sequence termination is tracked separately as the `format`
metric. `test/wer` is standard corpus WER and is unbounded under insertions
— evaluating an untrained model at temperature 1 can spike it well past 1.0
when rollouts ramble. Per-clip results are saved to `eval_rollouts_*.jsonl`,
so outliers can always be inspected or re-scored offline.

Eval samples at `temperature=1.0` / `max_tokens=8192`: the base model needs
that budget to finish thinking and emit a parseable transcript at step 0;
trained checkpoints stop after ~70 tokens, so the budget costs nothing later.

> Note: Inkling already transcribes clean LibriSpeech well (`test/wer` ~0.06
> zero-shot), so most GRPO groups are constant-reward and gains on this pairing
> are small. The recipe demonstrates the audio SFT/RL pipeline end to end; for
> a harder distribution with visible curves, see
> [`../medical_asr/`](../medical_asr/).

## Files

- `env.py` — LibriSpeech streaming + WAV cache, SL/RL dataset builders, the
  `-WER` env, and the WER evaluator (message/datum plumbing comes from
  [`../data.py`](../data.py), scoring from [`../grading.py`](../grading.py))
- `env_test.py` — datum masking / env reward / dataset tests (skip without
  the optional `tml_renderers` package, i.e. the `inkling` extra)
- `sl_train.py` — SFT entrypoint → `supervised.train`
- `rl_train.py` — RL entrypoint → `rl.train`
