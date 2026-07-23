# Serving Tinker fine-tunes on Modal

Turn a `tinker://` LoRA checkpoint into an OpenAI-compatible endpoint on
[Modal](https://modal.com). `prepare.py` merges the checkpoint into the base
model and writes it to a Volume; `serve.py` runs it behind an
[SGLang](https://github.com/sgl-project/sglang) endpoint.

## Setup

```bash
pip install "tinker-cookbook[modal]"
modal setup
export TINKER_API_KEY=tml-...   # required, used to download the checkpoint
export HF_TOKEN=hf-...          # optional, only for gated base models
```

`prepare`/`serve` read these from your local environment at deploy time; no
pre-created Modal secrets are needed.

## Prepare

```bash
modal run -m tinker_cookbook.inference.modal.prepare \
  --tinker-path tinker://<run-id>/sampler_weights/<name> \
  --base-model Qwen/Qwen3-8B --name my-finetune
```

Merges the adapter into the base model and writes the result to the
`tinker-artifacts` Volume. Runs on a GPU.

## Serve

```bash
FINETUNE=my-finetune MODEL=Qwen/Qwen3-8B modal deploy -m tinker_cookbook.inference.modal.serve
```

`modal deploy` gives a persistent URL; `modal run` spins up a replica and runs a
smoke test.

By default the endpoint requires a [Modal proxy auth
token](https://modal.com/docs/guide/webhook-urls#authentication). For a quick
open endpoint (no token), deploy with `UNAUTHENTICATED=1`. Then:

```bash
curl $URL/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "my-finetune",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

## Comparing against Tinker

`compare.py` samples the same prompts from the Tinker sampling client and the
Modal endpoint and reports how often they disagree:

```bash
FINETUNE=my-finetune MODEL=Qwen/Qwen3-8B \
  modal run -m tinker_cookbook.inference.modal.compare \
  --tinker-path tinker://<run-id>/sampler_weights/<name> --url $URL
```

## Tested models

Merge + serve verified end to end against the Tinker sampling client:

| Model | Arch |
|---|---|
| Qwen/Qwen3-8B | dense |
| Qwen/Qwen3.5-4B | GDN hybrid |
| Qwen/Qwen3.6-35B-A3B | MoE + GDN |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | Mamba hybrid |
| nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | Mamba hybrid |
| Qwen/Qwen3-235B-A22B-Instruct-2507 | MoE |

## Adding a model

Add a row to `MODEL_REGISTRY` in `common.py`:

```python
ModelConfig("org/Model", gpu="H100:2", tp=2)
```
