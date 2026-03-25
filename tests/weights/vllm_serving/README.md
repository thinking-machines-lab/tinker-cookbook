# vLLM Adapter Serving Tests

E2E tests verifying that PEFT adapters from `build_lora_adapter` can be loaded and served by vLLM.

## Why a separate environment?

vLLM pins its own `torch` and `transformers` versions which conflict with the main project's dependencies. These tests run in an isolated venv defined by [`requirements.txt`](requirements.txt).

## Setup

```bash
bash tests/weights/vllm_serving/setup_env.sh
```

Or manually:
```bash
python3 -m venv /tmp/vllm-test-env
/tmp/vllm-test-env/bin/pip install -r tests/weights/vllm_serving/requirements.txt
/tmp/vllm-test-env/bin/pip install -e .
```

Override the venv path with `VLLM_TEST_ENV`:
```bash
VLLM_TEST_ENV=~/my-vllm-env bash tests/weights/vllm_serving/setup_env.sh
```

## Running

```bash
# All models
/tmp/vllm-test-env/bin/python -m pytest tests/weights/vllm_serving/ -v -s

# Single model family
/tmp/vllm-test-env/bin/python -m pytest tests/weights/vllm_serving/test_qwen3.py -v -s

# Specific GPUs
CUDA_VISIBLE_DEVICES=0 /tmp/vllm-test-env/bin/python -m pytest tests/weights/vllm_serving/test_qwen3.py -v -s

# Parallel (different model families on different GPUs)
CUDA_VISIBLE_DEVICES=0   /tmp/vllm-test-env/bin/python -m pytest tests/weights/vllm_serving/test_qwen3.py::TestQwen3Dense -v -s &
CUDA_VISIBLE_DEVICES=2,3 /tmp/vllm-test-env/bin/python -m pytest tests/weights/vllm_serving/test_qwen3.py::TestQwen3Moe -v -s &
CUDA_VISIBLE_DEVICES=4   /tmp/vllm-test-env/bin/python -m pytest tests/weights/vllm_serving/test_qwen3_5.py -v -s &
```

## Model status

| Model | File | vLLM Serving | Notes |
|-------|------|:---:|-------|
| Qwen3-8B (dense) | `test_qwen3.py` | Tested | q_proj + v_proj LoRA |
| Qwen3-30B-A3B (MoE) | `test_qwen3.py` | Tested | Expert expansion, TP=2 |
| Qwen3.5-4B (split QKV) | `test_qwen3_5.py` | Tested | Split in_proj_q/k/v + full_attention |
| GPT-OSS-20B | `test_gpt_oss.py` | Conversion only | mxfp4+LoRA not supported in vLLM |
| Kimi-K2 | `test_kimi.py` | Placeholder | Model too large for routine testing |
| DeepSeek V3/V3.1 | `test_deepseek.py` | Placeholder | Intentionally unsupported |
| Nemotron-3-Nano-30B-A3B | `test_nemotron.py` | Tested | `backbone.*` → `model.*` remap, TP=2 |

## Adding a new model

1. Create `test_<model_family>.py`
2. Use helpers from `conftest.py` (`save_tinker_adapter`, `convert_and_load`, `generate`)
3. Create synthetic adapter weights matching the model's actual dimensions
4. Verify both conversion correctness and vLLM serving
