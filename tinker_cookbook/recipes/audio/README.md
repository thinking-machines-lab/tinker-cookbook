# Audio

Recipes for fine-tuning audio-input models (Inkling) with Tinker using
supervised fine-tuning and reinforcement learning.

For minimal Inkling audio and vision sampling examples using `tml-renderers`,
see the [Inkling scripts](../../scripts/inkling/).

## Recipes

- **[Speech recognition](./asr/)**: the "hello world" of audio fine-tuning —
  SFT and RL on public LibriSpeech, streamed straight from Hugging Face, with
  a `-WER` reward.
- **[Emotion + speech recognition](./emotion/)**: classify speaking style and
  transcribe expressive speech ([Expresso](https://arxiv.org/abs/2308.05725)),
  as a two-stage SFT → RL pipeline with a shared WER + style-accuracy
  evaluator.
- **[Medical ASR domain adaptation](./medical_asr/)**: teach the model
  specialist vocabulary (EkaCare dictated prescriptions, dense with drug
  names) with SFT, graded by WER plus medical-entity recall metrics.

Each recipe lives in its own directory, named for the task it demonstrates,
with its own README, `env.py`, colocated tests, and training entrypoints
(`sl_train.py` / `rl_train.py`, or a single `train.py`). Helpers shared by
every audio recipe live at this
level: [`grading.py`](./grading.py) (renderer-safe response decoding,
transcript normalization, WER) and [`data.py`](./data.py) (native-message
prompt/conversation builders, supervised datum construction, the WAV cache).

## Requirements

These recipes render Inkling through `tml_renderers` (the `inkling` extra).
Grading needs `jiwer`, and the ASR recipes decode Hugging Face audio into a
local WAV cache via `soundfile` (the `audio` extra); install both:

```bash
pip install "tinker_cookbook[audio,inkling]"
```

Per-recipe data preparation may need more (e.g. `ffmpeg` — see each recipe's
README).
