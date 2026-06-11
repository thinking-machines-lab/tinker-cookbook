# Serving Tinker fine-tunes on Modal

Take a `tinker://` LoRA checkpoint to a live, OpenAI-compatible endpoint on
[Modal](https://modal.com) in two steps:

```
                         ┌──────────────────────────┐
 tinker://checkpoint ──▶ │         PREPARE          │ ────▶ servable artifact
                         │ download + adapter/merge │     ┌────────────────────┐
                         └──────────────────────────┘     │    Modal Volume    │
                                                          │ "tinker-artifacts" │
                         ┌──────────────────────────┐     └────────────────────┘
 Client ◀──────────────  │          SERVE           │◀──────────────┘
                         └──────────────────────────┘
```

- **`prepare.py`** wraps the cookbook's `weights.download` / `build_lora_adapter` /
  `build_hf_model` in a Modal Function that writes the result to a shared Volume.
- **`serve.py`** is Modal's low-latency SGLang example pointed at that artifact.
- **`common.py`** holds the app, images, the Volume, and the per-model registry.

## Setup

```bash
pip install "tinker-cookbook[modal]"
modal setup                                   # one-time Modal auth

# Secrets the Functions read (names matter):
modal secret create tinker      TINKER_API_KEY=tml-...
modal secret create huggingface HF_TOKEN=hf-...   # for gated base models
```

## Use

```bash
# 1. Prepare: tinker:// checkpoint -> artifact on the Volume
modal run prepare.py \
  --tinker-path tinker://<run-id>/sampler_weights/<name> \
  --base-model Qwen/Qwen3-8B --name my-finetune

# 2. Serve: a flash endpoint that scales to zero when idle
FINETUNE=my-finetune MODEL=Qwen/Qwen3-8B modal deploy serve.py

# 3. Use it: standard OpenAI client against the printed URL
```

```python
from openai import OpenAI
client = OpenAI(base_url="https://<...>.modal.direct/v1", api_key="x")
client.chat.completions.create(
    model="my-finetune",
    messages=[{"role": "user", "content": "hello"}],
)
```

`modal run serve.py` instead of `deploy` runs the built-in smoke test against a
fresh replica.

## Adapter vs. merge

Two ways to make a checkpoint servable:

| | Adapter (`build_lora_adapter`) | Merge (`build_hf_model`) |
|---|---|---|
| Artifact | PEFT LoRA, MBs | full model, GBs |
| Serve | base model + `--lora-paths` | model dir directly |
| Coverage | engines that LoRA-serve the arch | universal |

Each model in `MODEL_REGISTRY` carries a `lora_serving` flag — whether the
pinned SGLang version can LoRA-serve that architecture. The mode is chosen
automatically from it; pass `--mode merge` to override. Asking for `adapter` on
a model with `lora_serving=False` fails fast rather than at engine startup.

## Adding a model

Add a row to `MODEL_REGISTRY` in `common.py`:

```python
ModelConfig("org/Model", gpu="H100:2", tp=2, lora_serving=True)
```

## Scope

This recipe assumes you already have a trained checkpoint and serves it as-is
(no quantize/requantize step).
