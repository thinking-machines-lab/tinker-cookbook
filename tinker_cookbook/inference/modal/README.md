# Serving Tinker fine-tunes on Modal

Turn a `tinker://` LoRA checkpoint into an OpenAI-compatible endpoint on
[Modal](https://modal.com). `prepare.py` downloads the checkpoint and writes a
servable artifact (a PEFT adapter or a merged model) to a Volume; `serve.py`
runs it behind an [SGLang](https://github.com/sgl-project/sglang) endpoint.

## Setup

```bash
pip install "tinker-cookbook[modal]"
modal setup
modal secret create tinker      TINKER_API_KEY=tml-...
modal secret create huggingface HF_TOKEN=hf-...   # for gated base models
```

## Prepare

```bash
modal run -m tinker_cookbook.inference.modal.prepare \
  --tinker-path tinker://<run-id>/sampler_weights/<name> \
  --base-model Qwen/Qwen3-8B --name my-finetune
```

Writes the artifact to the `tinker-artifacts` Volume. Pass `--mode merge` to
merge the adapter into the base model instead; the default comes from the
model's `lora_serving` flag in `common.py`. Merge runs on a GPU.

## Serve

```bash
FINETUNE=my-finetune MODEL=Qwen/Qwen3-8B modal deploy -m tinker_cookbook.inference.modal.serve
```

`modal deploy` gives a persistent URL; `modal run` spins up a replica and runs a
smoke test. Hit it with any OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="https://<...>.modal.direct/v1", api_key="x")
client.chat.completions.create(
    model="my-finetune",
    messages=[{"role": "user", "content": "hello"}],
)
```

## Tested models

Parity is measured as mean per-token `|Δlogprob|` between Tinker's sampling
client and the Modal endpoint on the same checkpoint (lower is closer; ~0.01
is engine-level rounding, not a model difference).

| Model | Arch | mode | parity (Δlogprob) |
|---|---|---|---|
| Qwen/Qwen3-8B | dense | adapter or merge | 0.005 |
| Qwen/Qwen3.5-4B | GDN hybrid | merge | 0.003 |
| Qwen/Qwen3.6-35B-A3B | MoE + GDN | merge | 0.005 |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | Mamba hybrid | merge | 0.016 |

## Adding a model

Add a row to `MODEL_REGISTRY` in `common.py`:

```python
ModelConfig("org/Model", gpu="H100:2", tp=2, lora_serving=True)
```

Set `lora_serving=False` for models SGLang can't LoRA-serve; they get merged.
