# Weights: Merge & Adapter Serving Support

Two build paths for trained LoRA adapters:

- **`build_hf_model`** — merge LoRA into full HF model weights (for direct deployment)
- **`build_lora_adapter`** — convert to PEFT format for serving with vLLM / SGLang (lightweight, hot-swappable)

## Table of Contents

- [Model Support Matrix](#model-support-matrix)
- [Recommended Settings](#recommended-settings)
- [Architecture Details](#architecture-details)
- [Unsupported Models — Adapter Serving](#unsupported-models--adapter-serving)
- [vLLM Serving Compatibility](#vllm-serving-compatibility)
- [Testing](#testing)

## Model Support Matrix

| Model Family | Merge | Adapter | Quantized Checkpoint | Notes |
|---|:---:|:---:|---|---|
| **Qwen3 dense** (4B–32B) | ✅ | ✅ | bf16 | Standard attention LoRA |
| **Qwen3 MoE** (30B-A3B) | ✅ | ✅ | bf16 | Fused gate_up_proj experts; vLLM MoE LoRA experimental |
| **Qwen3-VL MoE** (30B-A3B) | ✅ | ✅ | bf16 | Vision prefix (`model.language_model.*`) + fused experts |
| **Qwen3.5 dense** (4B, 27B) | ✅ | ✅ | bf16 | Split QKV (`in_proj_q/k/v`); tied embeddings |
| **Qwen3.5 MoE** (35B-A3B, 397B-A17B) | ✅ | ✅ | bf16 | Split QKV + fused experts; vLLM MoE LoRA experimental |
| **GPT-OSS** (20B, 120B) | ✅ | ✅ | bf16 | `.attn` → `.self_attn` remap; interleaved expert layout |
| **Kimi-K2** (~1T-A32B) | ✅ | ✅ | INT4 pack-quantized | DeepSeek arch; separate experts; shard-by-shard dequant/merge/requant |
| **Kimi-K2.5** (~1T-A32B) | ✅ | ✅ | INT4 pack-quantized | VL model; `language_model.model.*` prefix; vLLM LoRA not yet supported |
| **DeepSeek V3 / V3.1** | ✅ | ❌ | bf16 or native FP8 | vLLM/SGLang don't support DeepSeek LoRA |
| **Nemotron-3** (Nano 30B, Super 120B) | ✅ | ✅ | bf16 | `backbone.*` weight prefix (handled automatically) |

### Legend

- **Merge** = `build_hf_model` support
- **Adapter** = `build_lora_adapter` support (PEFT format for vLLM/SGLang serving)
- **Quantized Checkpoint** = format of the HuggingFace checkpoint weights

## Recommended Settings

### Learning Rate

Use `hyperparam_utils.get_lr(model_name)` when available. LoRA training typically uses ~10x higher LR than full fine-tuning.

| Model Family | LoRA LR | Full FT LR | Notes |
|---|---|---|---|
| **Qwen3** (all sizes) | `get_lr(model_name)` | `get_lr(model_name, is_lora=False)` | Calibrated; scales with hidden size |
| **Llama 3.x** (all sizes) | `get_lr(model_name)` | `get_lr(model_name, is_lora=False)` | Calibrated; scales with hidden size |
| **Kimi-K2** | ~5e-4 | ~5e-5 | Not yet calibrated in `get_lr`; start with 5e-4 for LoRA |
| **Kimi-K2.5** | ~5e-4 | ~5e-5 | Same as K2 (same hidden size / architecture) |
| **DeepSeek V3 / V3.1** | ~5e-4 | ~5e-5 | Not yet calibrated; similar architecture to Kimi |
| **GPT-OSS** | ~5e-4 | ~5e-5 | Not yet calibrated |
| **Nemotron-3** | ~5e-4 | ~5e-5 | Not yet calibrated |
| **DPO** (all models) | ~1e-5 | — | Start with `dpo_beta=0.1` |

### LoRA Rank and Alpha

Default settings that work well across models:

| Setting | Default | Notes |
|---|---|---|
| `lora_rank` (`r`) | 32 | Higher rank = more capacity but more memory. 16–64 is typical. |
| `lora_alpha` | 32 | Common to set `alpha = rank`. Effective scaling = `alpha / rank`. |
| Merge scaling | `alpha / rank` | Applied automatically during `build_hf_model`. |

### Merge Strategy

| Strategy | When to use | Peak memory |
|---|---|---|
| `"auto"` / `"shard"` (default) | Large models, quantized checkpoints | ~1 shard (~10 GB) |
| `"full"` | Small models, need specific output dtype | Full model size |

For Kimi K2/K2.5 (595 GB–1 TB), always use the default shard strategy. The shard path automatically handles INT4 dequant → LoRA merge → INT4 requant for packed expert weights.

Wall time depends on I/O bandwidth and service load (multiple concurrent jobs share the storage). Expect ~30–60 minutes for a 600 GB model on NFS, faster on local SSD.

## Architecture Details

### Expert Weight Layouts

MoE models store expert weights in different layouts. The merge/adapter code handles all three:

| Layout | Models | Structure |
|---|---|---|
| **Separate** | DeepSeek, Kimi-K2, Kimi-K2.5, Qwen3 MoE (transformers 4.x) | Individual `experts.{i}.gate_proj.weight` per expert |
| **Fused concatenated** | Qwen3 MoE (transformers 5.x), Qwen3.5 MoE, Qwen3-VL MoE | Single `experts.gate_up_proj` with `[gate \| up]` layout |
| **Fused interleaved** | GPT-OSS | Single `experts.gate_up_proj` with `[g0, u0, g1, u1, ...]` layout |

### INT4 Pack-Quantized Experts (Kimi K2, K2.5)

Kimi models on HuggingFace use compressed-tensors `pack-quantized` format for routed expert weights:

| Key suffix | Dtype | Contents |
|---|---|---|
| `.weight_packed` | I32 | 8 signed INT4 values packed per int32 |
| `.weight_scale` | BF16 | Per-group dequantization scale (group_size=32) |
| `.weight_shape` | I32 | Original `[out_dim, in_dim]` dimensions |

Dense layers, attention, shared experts, and embeddings remain in bf16.

During merge, the shard export transparently:
1. Dequantizes INT4 → bf16
2. Applies LoRA delta in float32
3. Re-quantizes bf16 → INT4

Detection is config-based (`quantization_config.format == "pack-quantized"`), not key-suffix heuristics. Other quantized formats (Nemotron NVFP4, DeepSeek native FP8) are not affected.

### Qwen3.5 Split QKV

Qwen3.5 linear attention layers use a fused `in_proj_qkv` weight, but Tinker trains separate `in_proj_q/k/v` LoRA adapters (which may have unequal dimensions). The merge path fuses them back; the adapter path keeps them split (vLLM handles this via `packed_modules_mapping`).

### Key Remapping

| Remap | Applies to | Why |
|---|---|---|
| `base_model.model.` prefix strip | All models | Tinker adapter prefix → HF parameter names |
| `unembed_tokens` → `lm_head` (or `embed_tokens`) | All models | Tinker naming → HF naming; tied embeddings use `embed_tokens` |
| `model.*` → `model.language_model.*` | Qwen3.5 VL, Qwen3-VL | Inner `language_model` prefix for standard VL models |
| `model.*` → `language_model.model.*` | Kimi-K2.5 | Outer `language_model.` prefix (different from Qwen pattern) |
| `.attn` → `.self_attn` | GPT-OSS | Tinker internal naming → HF naming |
| `w1/w2/w3` → `gate_proj/down_proj/up_proj` | MoE expert keys | Tinker expert naming → HF naming |

## Unsupported Models — Adapter Serving

### DeepSeek V3/V3.1 (`model_type=deepseek_v3`)

**Status:** Blocked — raises `WeightsAdapterError`.

**Reason:** vLLM and SGLang do not support LoRA inference for the DeepSeek V3 architecture. Use `build_hf_model` to merge the adapter into a full model instead.

**Unblock when:** vLLM adds DeepSeek V3 LoRA support.

### Kimi-K2.5 (`model_type=kimi_k25`)

**Status:** Adapter conversion works, but vLLM 0.18 does not implement `SupportsLoRA` for `KimiK25ForConditionalGeneration`.

**Workaround:** Use `build_hf_model` to merge the adapter into a full model, then serve the merged model directly.

### ~~Nemotron-3 (`model_type=nemotron_h`)~~ — Resolved

Nemotron HF checkpoints use `backbone.*` instead of `model.*` as the weight prefix. vLLM remaps `backbone.*` → `model.*` internally via `WeightsMapper`. The adapter conversion applies the same remap via `_SERVING_PREFIX_REMAPS` so PEFT keys match vLLM's internal parameter names.

## vLLM Serving Compatibility

Adapter serving has been verified end-to-end with vLLM 0.18:

| Model | vLLM LoRA | Status |
|---|---|---|
| Qwen3-8B (dense) | Full support | ✅ Verified |
| Qwen3-30B-A3B (MoE) | Experimental | ✅ Verified (requires all 3 expert projections) |
| Qwen3.5-4B (split QKV) | Full support | ✅ Verified (split in_proj_q/k/v works) |
| GPT-OSS-20B | Full support | ⚠️ Conversion verified; serving blocked by mxfp4+LoRA incompatibility |
| Kimi-K2 | Supported (DeepSeekV2) | ⚠️ Conversion verified; model too large (~1TB) for routine e2e testing |
| Kimi-K2.5 | Not supported | ⚠️ Conversion verified; vLLM 0.18 lacks LoRA for `KimiK25ForConditionalGeneration` |
| DeepSeek V3/V3.1 | Not supported | ❌ Adapter conversion blocked |
| Nemotron-3-Nano (30B-A3B) | Full support (vLLM) | ✅ Verified (`backbone.*` → `model.*` remap, TP=2) |
| Nemotron-3-Super (120B-A12B) | Full support (vLLM) | ✅ Verified (`backbone.*` → `model.*` remap, TP=4) |

See `tests/weights/vllm_serving/` for the serving test suite.

## Testing

```bash
# Unit tests (no network, no API key)
pytest tinker_cookbook/weights/ -v

# E2E tests (downloads HF configs, no API key)
pytest tests/weights/ -v

# Profile detection against real HF model configs
pytest tests/weights/test_profile_kimi.py -v

# vLLM serving tests (requires GPU + isolated venv)
bash tests/weights/vllm_serving/setup_env.sh
/tmp/vllm-test-env/bin/python -m pytest tests/weights/vllm_serving/ -v -s
```
