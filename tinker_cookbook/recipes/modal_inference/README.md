# Serving Tinker fine-tunes on Modal

Turn a `tinker://` LoRA checkpoint into an OpenAI-compatible endpoint on
[Modal](https://modal.com). `prepare.py` downloads the checkpoint and writes a
servable artifact — a PEFT adapter or a merged model — to a Volume; `serve.py`
runs it behind an [SGLang](https://github.com/sgl-project/sglang) endpoint that
scales to zero.

## Setup

```bash
pip install "tinker-cookbook[modal]"
modal setup
modal secret create tinker      TINKER_API_KEY=tml-...
modal secret create huggingface HF_TOKEN=hf-...   # for gated base models
```

## Prepare

```bash
modal run prepare.py \
  --tinker-path tinker://<run-id>/sampler_weights/<name> \
  --base-model Qwen/Qwen3-8B --name my-finetune
```

Writes the artifact to the `tinker-artifacts` Volume. Pass `--mode merge` to
merge the adapter into the base model instead; the default comes from the
model's `lora_serving` flag in `common.py`.

## Serve

```bash
FINETUNE=my-finetune MODEL=Qwen/Qwen3-8B modal deploy serve.py
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

## Adding a model

Add a row to `MODEL_REGISTRY` in `common.py`:

```python
ModelConfig("org/Model", gpu="H100:2", tp=2, lora_serving=True)
```

Set `lora_serving=False` for models the pinned SGLang can't LoRA-serve — they
get merged automatically.
