# Weight Lifecycle

Complete reference for downloading, merging, and publishing trained weights.

## Reference

- `tinker_cookbook/weights/__init__.py` — API overview
- `tinker_cookbook/weights/_download.py` — Download implementation
- `tinker_cookbook/weights/_export/` — LoRA merge (full, quantized, sharded)
- `tinker_cookbook/weights/_publish.py` — HuggingFace Hub publish

## Full workflow

```python
from tinker_cookbook import weights

# Step 1: Download adapter
adapter_dir = weights.download(
    tinker_path="tinker://run-id/sampler_weights/final",
    output_dir="./adapter",
)

# Step 2: Merge LoRA into base model
weights.build_hf_model(
    base_model="Qwen/Qwen3.5-35B-A3B",
    adapter_path=adapter_dir,
    output_path="./model",
    dtype="bfloat16",
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

```python
adapter_dir = weights.download(
    tinker_path="tinker://run-id/sampler_weights/final",
    output_dir="./adapter",
    base_url=None,
)
```

### `weights.build_hf_model()`

Merges LoRA adapter into base model:
```python
weights.build_hf_model(
    base_model="Qwen/Qwen3-8B",
    adapter_path="./adapter",
    output_path="./model",
    dtype="bfloat16",
    trust_remote_code=None,
)
```

### `weights.build_lora_adapter()`

Convert to PEFT format for vLLM/SGLang serving (does not merge):
```python
weights.build_lora_adapter(
    base_model="Qwen/Qwen3-8B",
    adapter_path="./adapter",
    output_path="./peft_adapter",
    trust_remote_code=None,
)
```

### `weights.publish_to_hf_hub()`

```python
url = weights.publish_to_hf_hub(
    model_path="./model",
    repo_id="user/my-finetuned-model",
    private=True,
    token=None,  # Uses HF_TOKEN env var
)
```

## Pitfalls

- `download()` expects `tinker://` path from `save_weights_for_sampler`, not `save_state`
- `build_hf_model()` requires the base model to be downloadable from HuggingFace
- Set `HF_TOKEN` for private models and publishing
- `dtype="bfloat16"` is recommended for most models
