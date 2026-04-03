# GPU Weight Export Tests

End-to-end tests for the full weight export pipeline with **real models and real Tinker adapters**. Each test file covers one model family across all applicable dimensions: merge, adapter export, quantization, GPU acceleration, and vLLM serving.

## Setup

### 1. Model weights on NFS

Cache HuggingFace model weights on NFS so they persist across sessions:

```bash
export HF_HUB_CACHE=~/huggingface/hub
```

Models are downloaded automatically on first run but are large (up to 1TB for Kimi). Pre-download recommended:

```bash
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --cache-dir ~/huggingface/hub
```

### 2. Tinker API key

```bash
export TINKER_API_KEY=<your-key>
```

### 3. vLLM environment (optional, for serving tests)

vLLM pins its own torch/transformers versions, so it needs a separate venv:

```bash
bash tests/weights/vllm_serving/setup_env.sh
```

This creates `/tmp/vllm-test-env/`. The `TestVllmServing` classes are skipped automatically when vLLM is not installed.

## Running

```bash
# All tests (merge + adapter + quant, no vLLM)
HF_HUB_CACHE=~/huggingface/hub pytest tests/weights/gpu/ -v --timeout=3600

# Single model family
HF_HUB_CACHE=~/huggingface/hub pytest tests/weights/gpu/test_qwen3.py -v

# Skip large models (DeepSeek 671B, Kimi 1TB)
HF_HUB_CACHE=~/huggingface/hub pytest tests/weights/gpu/ -v \
  --ignore=tests/weights/gpu/test_deepseek.py \
  --ignore=tests/weights/gpu/test_kimi.py \
  --ignore=tests/weights/gpu/test_kimi_k25.py

# vLLM serving tests only (requires vLLM venv)
HF_HUB_CACHE=~/huggingface/hub /tmp/vllm-test-env/bin/python -m pytest \
  tests/weights/gpu/ -v -s -k vllm

# Parallel: different models on different GPUs
CUDA_VISIBLE_DEVICES=0 HF_HUB_CACHE=~/huggingface/hub pytest tests/weights/gpu/test_qwen3.py -v &
CUDA_VISIBLE_DEVICES=1 HF_HUB_CACHE=~/huggingface/hub pytest tests/weights/gpu/test_qwen3_5.py -v &
```

## Coverage matrix

### Test dimensions

- **Merge**: `build_hf_model` with real adapter → verify output is valid HF model
- **Adapter**: `build_lora_adapter` → verify PEFT format output
- **FP8 CPU/GPU**: `quantize="experts-fp8"` on CPU and GPU → verify routed experts in FP8
- **MXFP4 CPU/GPU**: Shard merge with MXFP4 dequant/requant → verify blocks format preserved
- **INT4**: Shard merge with INT4 dequant/requant → verify packed format preserved
- **CPU/GPU equiv**: Bitwise identical output from CPU and GPU paths
- **vLLM**: Export adapter → load in vLLM → generate text with LoRA

### Per-model coverage

| File | Model | Type | Merge | Adapter | FP8 CPU | FP8 GPU | MXFP4 | MXFP4 GPU | INT4 | CPU/GPU equiv | vLLM |
|------|-------|------|:-----:|:-------:|:-------:|:-------:|:-----:|:---------:|:----:|:-------------:|:----:|
| test_qwen3.py | Qwen3-4B | Dense | x | x | | | | | | | x |
| test_qwen3_5.py | Qwen3.5-35B-A3B | MoE | x | x | x | x | | | | | x |
| test_qwen3_vl.py | Qwen3-VL-30B-A3B | VL MoE | x | x | x | | | | | | x |
| test_deepseek.py | DeepSeek-V3.1 | FP8 MoE | | | x | x | | | | x | - |
| test_kimi.py | Kimi-K2 | INT4 MoE | x | x | | | | | x | | - |
| test_kimi_k25.py | Kimi-K2.5 | VL+INT4 MoE | x | x | | | | | x | | - |
| test_gpt_oss.py | GPT-OSS-20B | MXFP4 MoE | x | x | | | x | x | | | - |
| test_nemotron.py | Nemotron-30B | Mamba+MoE | x | x | | | | | | | x |

**`-`** = not applicable (DeepSeek uses FP8 merge not adapters; GPT-OSS MXFP4+LoRA unsupported in vLLM; Kimi too large for single-node vLLM)

### Tinker model lineup coverage

All 36 models on [Tinker](https://tinker-docs.thinkingmachines.ai/tinker/models/) map to one of the 8 configurations above. Scaled-up variants (e.g. Qwen3-32B vs Qwen3-4B) share the same merge profile and code path — shard-by-shard processing is size-independent.

| Config | Models | Tested by |
|--------|--------|-----------|
| Qwen3 dense (tied) | Qwen3-0.6B, 1.7B, 4B + variants | test_qwen3.py |
| Qwen3 dense (untied) | Qwen3-8B, 14B, 32B + variants | test_qwen3.py (same path) |
| Qwen3 MoE | Qwen3-30B-A3B, 235B-A22B + variants | test_qwen3_5.py (same merge profile) |
| Qwen3-VL MoE | Qwen3-VL-30B-A3B, 235B-A22B | test_qwen3_vl.py |
| Qwen3.5 (dense + MoE, VL) | Qwen3.5-4B, 27B, 35B-A3B, 397B-A17B | test_qwen3_5.py |
| DeepSeek V3 (native FP8) | DeepSeek-V3.1, V3.1-Base | test_deepseek.py |
| Kimi K2 (INT4) | Kimi-K2-Thinking | test_kimi.py |
| Kimi K2.5 (VL + INT4) | Kimi-K2.5 | test_kimi_k25.py |
| GPT-OSS (MXFP4) | gpt-oss-20b, 120b | test_gpt_oss.py |
| Nemotron (fused proj) | Nemotron-Nano-30B, Super-120B | test_nemotron.py |
| Llama 3.x | Llama-3.1/3.2/3.3 (8 models) | Not tested (gated on HF) — uses default merge profile, same path as Qwen3 dense |

## Last validated

Tested on 8xH200 (2026-04-03):

| Test | Status | Time |
|------|--------|------|
| test_qwen3.py (merge + adapter) | PASSED | ~4min |
| test_qwen3.py (vLLM serving) | PASSED | ~1min |
| test_qwen3_5.py (merge + adapter + FP8) | PASSED | ~16min |
| test_gpt_oss.py (MXFP4 merge CPU + GPU) | PASSED | ~2min |
| test_nemotron.py (merge + adapter) | PASSED | ~1min |
| test_kimi.py (INT4 merge + adapter) | PASSED | ~3min |
| test_deepseek.py (FP8 CPU) | PASSED | ~45min |

## Adding a new model

1. Create `test_<model>.py` following the pattern in existing files
2. Add test classes as applicable:
   - `TestMerge` — shard merge with output verification
   - `TestAdapter` — PEFT export with config verification
   - `TestQuantized` — quantized merge with format verification (FP8/MXFP4/INT4)
   - `TestVllmServing` — adapter export → vLLM load → generate (use `@skip_no_vllm`)
3. Use fixtures from `conftest.py`: `train_one_step`, `download_adapter`, `verify_merged_model`, `verify_fp8_output`, `vllm_generate`
4. Update the coverage matrices above
5. Run and record results in "Last validated" section
