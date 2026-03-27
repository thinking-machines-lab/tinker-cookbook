---
name: tinker-weights
description: Guide for the weight lifecycle — downloading trained weights from Tinker, merging LoRA adapters into HuggingFace models, and publishing to HuggingFace Hub. Use when the user asks about exporting, downloading, merging, or publishing trained model weights.
---

# Weight Lifecycle

The `tinker_cookbook.weights` subpackage provides a standard pipeline for trained weight management: **download → build → publish**.

## Reference

Read these for details:
- `tinker_cookbook/weights/__init__.py` — API overview and workflow example
- `tinker_cookbook/weights/_download.py` — Download implementation
- `tinker_cookbook/weights/_export/` — LoRA merge implementation (full, quantized, sharded)
- `tinker_cookbook/weights/_publish.py` — HuggingFace Hub publish
- `docs/download-weights.mdx` — Download guide
- `docs/publish-weights.mdx` — Publishing guide
- `docs/save-load.mdx` — Checkpointing (save_weights_for_sampler vs save_state)

## Full workflow

```python
from tinker_cookbook import weights

# Step 1: Download adapter from Tinker
adapter_dir = weights.download(
    tinker_path="tinker://run-id/sampler_weights/final",
    output_dir="./adapter",
)

# Step 2: Merge LoRA adapter into base model
weights.build_hf_model(
    base_model="Qwen/Qwen3.5-35B-A3B",
    adapter_path=adapter_dir,
    output_path="./model",
    dtype="bfloat16",  # or "float16", "float32"
)

# Step 3: Publish to HuggingFace Hub
url = weights.publish_to_hf_hub(
    model_path="./model",
    repo_id="user/my-finetuned-model",
    private=True,
)
```

## API reference

### `weights.download()`
Downloads and extracts a checkpoint archive from Tinker.

```python
adapter_dir = weights.download(
    tinker_path="tinker://run-id/sampler_weights/final",  # Tinker checkpoint path
    output_dir="./adapter",      # Local directory to extract to
    base_url=None,               # Optional custom Tinker API URL
)
# Returns: path to extracted directory
```

### `weights.build_hf_model()`
Merges a LoRA adapter into a base model, producing a full HuggingFace model.

```python
weights.build_hf_model(
    base_model="Qwen/Qwen3-8B",     # HF model name or local path
    adapter_path="./adapter",        # Directory with adapter_model.safetensors
    output_path="./model",           # Where to save merged model
    dtype="bfloat16",                # Weight dtype
    trust_remote_code=None,          # Override HF_TRUST_REMOTE_CODE
)
```

### `weights.publish_to_hf_hub()`
Pushes a local model directory to HuggingFace Hub.

```python
url = weights.publish_to_hf_hub(
    model_path="./model",                    # Local model directory
    repo_id="user/my-finetuned-model",       # HF repo ID
    private=True,                            # Private repo
    token=None,                              # HF token (uses HF_TOKEN env var if None)
)
# Returns: URL to published repo
```

### `weights.build_lora_adapter()`
Convert a Tinker LoRA adapter to standard PEFT format for serving with vLLM (`--lora-modules`) or SGLang (`--lora-paths`). Unlike `build_hf_model()`, this does **not** merge weights into the base model — it produces a lightweight adapter directory.

```python
weights.build_lora_adapter(
    base_model="Qwen/Qwen3-8B",     # HF model name (for config metadata)
    adapter_path="./adapter",        # Directory with adapter_model.safetensors
    output_path="./peft_adapter",    # Where to save PEFT-format adapter
    trust_remote_code=None,          # Override HF_TRUST_REMOTE_CODE
)
```

## Checkpoint types (during training)

During training, there are two types of checkpoints:

- **`save_state()`** — Full state (weights + optimizer). Used for **resuming** training.
- **`save_weights_for_sampler()`** — Weights only. Used for **sampling** and **export**.

The `weights.download()` function works with sampler weights (`save_weights_for_sampler` checkpoints).

## Common pitfalls
- `download()` expects a `tinker://` path from `save_weights_for_sampler`, not `save_state`
- `build_hf_model()` requires the base model to be downloadable from HuggingFace
- Set `HF_TOKEN` environment variable for private models and publishing
- `dtype="bfloat16"` is recommended for most models
