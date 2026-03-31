# Weights: Merge & Adapter Serving Support

Two build paths for trained LoRA adapters:

- **`build_hf_model`** — merge LoRA into full HF model weights (for direct deployment)
- **`build_lora_adapter`** — convert to PEFT format for serving with vLLM / SGLang (lightweight, hot-swappable)

## Model Support Matrix

| Model Family | Merge | Adapter | Transformers | Notes |
|---|:---:|:---:|---|---|
| **Qwen3 dense** (4B–32B) | ✅ | ✅ | 4.x, 5.x | Standard attention LoRA |
| **Qwen3 MoE** (30B-A3B) | ✅ | ✅ | 4.x, 5.x | Fused gate_up_proj experts; vLLM MoE LoRA experimental |
| **Qwen3-VL MoE** (30B-A3B) | ✅ | ✅ | 4.x, 5.x | Vision prefix (`model.language_model.*`) + fused experts |
| **Qwen3.5 dense** (4B, 27B) | ✅ | ✅ | 5.x only | Split QKV (`in_proj_q/k/v`); tied embeddings |
| **Qwen3.5 MoE** (35B-A3B, 397B-A17B) | ✅ | ✅ | 5.x only | Split QKV + fused experts; vLLM MoE LoRA experimental |
| **GPT-OSS** (20B, 120B) | ✅ | ✅ | 4.x, 5.x | `.attn` → `.self_attn` remap; interleaved expert layout |
| **Kimi-K2** | ✅ | ✅ | 4.x, 5.x | DeepSeek architecture, separate experts; ~1TB bf16 |
| **Kimi-K2.5** | ✅ | ✅ | 4.x, 5.x | VL model (`kimi_k25`); INT4 packed experts; vLLM LoRA not yet supported |
| **DeepSeek V3 / V3.1** | ✅ | ❌ | 4.x (custom code), 5.x (native) | vLLM/SGLang don't support DeepSeek LoRA |
| **Nemotron-3** (Nano 30B, Super 120B) | ✅ | ✅ | 4.x, 5.x | `backbone.*` weight prefix (handled automatically) |

### Legend

- **Merge** = `build_hf_model` support
- **Adapter** = `build_lora_adapter` support (PEFT format for vLLM/SGLang serving)
- **Transformers** = supported `transformers` library versions

## Architecture Details

### Expert Weight Layouts

MoE models store expert weights in different layouts. The merge/adapter code handles all three:

| Layout | Models | Structure |
|---|---|---|
| **Separate** | DeepSeek, Kimi-K2, Kimi-K2.5, Qwen3 MoE (transformers 4.x) | Individual `experts.{i}.gate_proj.weight` per expert |
| **Fused concatenated** | Qwen3 MoE (transformers 5.x), Qwen3.5 MoE, Qwen3-VL MoE | Single `experts.gate_up_proj` with `[gate \| up]` layout |
| **Fused interleaved** | GPT-OSS | Single `experts.gate_up_proj` with `[g0, u0, g1, u1, ...]` layout |

### Qwen3.5 Split QKV

Qwen3.5 linear attention layers use a fused `in_proj_qkv` weight, but Tinker trains separate `in_proj_q/k/v` LoRA adapters (which may have unequal dimensions). The merge path fuses them back; the adapter path keeps them split (vLLM handles this via `packed_modules_mapping`).

### Key Remapping

| Remap | Applies to | Why |
|---|---|---|
| `base_model.model.` prefix strip | All models | Tinker adapter prefix → HF parameter names |
| `unembed_tokens` → `lm_head` (or `embed_tokens`) | All models | Tinker naming → HF naming; tied embeddings use `embed_tokens` |
| `model.*` → `model.language_model.*` | Vision models | Vision models nest language model under `language_model` |
| `.attn` → `.self_attn` | GPT-OSS | Tinker internal naming → HF naming |
| `w1/w2/w3` → `gate_proj/down_proj/up_proj` | MoE expert keys | Tinker expert naming → HF naming |

## Unsupported Models — Adapter Serving

### DeepSeek V3/V3.1 (`model_type=deepseek_v3`)

**Status:** Blocked — raises `WeightsAdapterError`.

**Reason:** vLLM and SGLang do not support LoRA inference for the DeepSeek V3 architecture. Use `build_hf_model` to merge the adapter into a full model instead.

**Unblock when:** vLLM adds DeepSeek V3 LoRA support.

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

# vLLM serving tests (requires GPU + isolated venv)
bash tests/weights/vllm_serving/setup_env.sh
/tmp/vllm-test-env/bin/python -m pytest tests/weights/vllm_serving/ -v -s
```
