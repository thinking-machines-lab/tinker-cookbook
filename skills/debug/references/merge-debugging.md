# Merge Debugging Reference

Technical reference for debugging weight merge issues between Tinker adapters and HuggingFace models. Read this when the user has output mismatches between Tinker sampling and external inference engines.

## Tinker adapter weight naming

Tinker adapters store LoRA A/B matrices with these naming conventions:

```
base_model.model.layers.{N}.mlp.gate_proj.lora_A.weight    # w1 = gate
base_model.model.layers.{N}.mlp.down_proj.lora_A.weight    # w2 = down
base_model.model.layers.{N}.mlp.up_proj.lora_A.weight      # w3 = up
base_model.model.layers.{N}.self_attn.q_proj.lora_A.weight
base_model.model.layers.{N}.self_attn.k_proj.lora_A.weight
base_model.model.layers.{N}.self_attn.v_proj.lora_A.weight
base_model.model.layers.{N}.self_attn.o_proj.lora_A.weight
```

For MoE models, expert weights are 3D tensors: `(num_experts, rank, dim)`.

## MoE expert weight layouts

Different model families fuse gate and up projections differently in HuggingFace format. Getting this wrong is the #1 cause of merge-related output mismatches.

### Separate layout (Qwen3 MoE, DeepSeek, Kimi)

Each expert has individual weight files:
```
model.layers.{N}.mlp.experts.{E}.gate_proj.weight   # (out_dim, in_dim)
model.layers.{N}.mlp.experts.{E}.down_proj.weight
model.layers.{N}.mlp.experts.{E}.up_proj.weight
```

Merge is straightforward — apply LoRA delta to each expert independently.

### Fused concatenated layout (Qwen3.5, Qwen3-VL)

Gate and up projections are concatenated into a single tensor:
```
model.layers.{N}.mlp.experts.gate_up_proj   # (num_experts, out_dim*2, in_dim) or (num_experts, in_dim, out_dim*2)
```

The first half (along the fused dimension) is `gate`, the second half is `up`:
```python
# Splitting:
gate = gate_up_proj[..., :out_dim]
up   = gate_up_proj[..., out_dim:]

# Merging LoRA delta for gate (w1):
gate_up_proj[..., :out_dim] += lora_B_gate @ lora_A_gate

# Merging LoRA delta for up (w3):
gate_up_proj[..., out_dim:] += lora_B_up @ lora_A_up
```

**Important:** The fused dimension can be dim 1 or dim 2 depending on the model:
- Qwen3-VL: `(num_experts, in_dim, out_dim*2)` — fused on last dim
- Qwen3.5: `(num_experts, out_dim*2, in_dim)` — fused on middle dim (delta must be transposed)

The cookbook's `_detect_fused_axis()` handles this automatically.

### Fused interleaved layout (GPT-OSS)

Gate and up elements alternate:
```
model.layers.{N}.mlp.experts.gate_up_proj
# Layout: [g0, u0, g1, u1, g2, u2, ...]
```

```python
# Splitting:
gate = gate_up_proj[..., 0::2]   # even indices
up   = gate_up_proj[..., 1::2]   # odd indices

# Merging LoRA delta for gate:
gate_up_proj[..., 0::2] += delta_gate

# Merging LoRA delta for up:
gate_up_proj[..., 1::2] += delta_up
```

**Using concatenation when the model expects interleaving (or vice versa) silently corrupts the weights.** The model will still generate text, but outputs will be degraded or invalid.

## Vision-language model considerations

VL models add a `model.language_model.*` prefix to language model weights:
```
# Standard model:
model.layers.0.mlp.gate_proj.weight

# VL model:
model.language_model.layers.0.mlp.gate_proj.weight
```

The adapter weights don't have this prefix, so the merge code must add it. The cookbook handles this via `has_language_model_prefix` in the merge profile.

## Split QKV projections (Qwen3.5)

Qwen3.5 models with hybrid attention (some layers use linear attention) fuse Q/K/V into a single `in_proj_qkv`:
```
model.layers.{N}.self_attn.in_proj_qkv.weight   # (q_dim + k_dim + v_dim, hidden)
```

Tinker trains separate Q/K/V adapters. During merge, the adapter deltas must be written to the correct row range:
```python
q_rows = lora_B_q.shape[0]  # e.g., 4096
k_rows = lora_B_k.shape[0]  # e.g., 512
v_rows = lora_B_v.shape[0]  # e.g., 512

# Q delta goes to rows [0 : q_rows]
# K delta goes to rows [q_rows : q_rows + k_rows]
# V delta goes to rows [q_rows + k_rows : q_rows + k_rows + v_rows]
```

## Diagnostic: comparing merge outputs

To verify a merge is correct, compare against Tinker's sampling output on a known input:

```python
import torch
from safetensors.torch import load_file

# Load a few expert weights from both merges
cookbook = load_file("cookbook_merge/model-00003-of-00010.safetensors")
custom = load_file("custom_merge/model-00003-of-00010.safetensors")

# Focus on MLP expert weights (where fusion bugs manifest)
for key in sorted(cookbook.keys()):
    if "experts" in key and "mlp" in key:
        if key in custom:
            diff = (cookbook[key].float() - custom[key].float()).abs()
            if diff.max() > 1e-4:
                print(f"MISMATCH {key}: max={diff.max():.6f} mean={diff.mean():.6f}")
            else:
                print(f"OK       {key}: max={diff.max():.6f}")
```

If mismatches cluster in `gate_up_proj` weights, the fusion convention is wrong.

## Diagnostic: end-to-end output comparison

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./merged_model", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./merged_model")

# Use the exact same tokens as Tinker
input_ids = torch.tensor([tinker_token_ids], device=model.device)
with torch.no_grad():
    out = model.generate(input_ids, max_new_tokens=100, temperature=0, do_sample=False)

print(tokenizer.decode(out[0]))
# Compare with Tinker sampling output
```

## Using PEFT adapter as a workaround

If merge issues persist and time is critical, serve the unmerged adapter via PEFT:

```python
from tinker_cookbook import weights

weights.build_lora_adapter(
    base_model="Qwen/Qwen3-8B",
    adapter_path="./adapter",
    output_path="./peft_adapter",
)
```

Then serve with vLLM:
```bash
vllm serve Qwen/Qwen3-8B --lora-modules my_adapter=./peft_adapter
```

This avoids merge entirely — the engine applies the LoRA at inference time. It's slightly slower but eliminates merge-related bugs.
