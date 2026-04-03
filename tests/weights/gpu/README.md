# GPU Weight Export Tests

End-to-end tests that exercise the full weight export pipeline with real models.
Each file covers one model family and tests all applicable capabilities.

## Requirements

- **GPU**: CUDA-capable GPU (H100/H200 recommended)
- **TINKER_API_KEY**: Set in environment for training
- **Model weights**: Cached on NFS at `~/huggingface/hub/`
## Running

```bash
# All GPU tests
HF_HUB_CACHE=~/huggingface/hub pytest tests/weights/gpu/ -v --timeout=3600

# Single model family
HF_HUB_CACHE=~/huggingface/hub pytest tests/weights/gpu/test_qwen3.py -v

# Skip slow models (DeepSeek 671B)
HF_HUB_CACHE=~/huggingface/hub pytest tests/weights/gpu/ -v --ignore=tests/weights/gpu/test_deepseek.py
```

## Coverage matrix

| File | Model | Merge | Adapter | FP8 CPU | FP8 GPU | MXFP4 | MXFP4 GPU | INT4 | CPU/GPU equiv |
|------|-------|:-----:|:-------:|:-------:|:-------:|:-----:|:---------:|:----:|:-------------:|
| test_qwen3.py | Qwen3-4B | x | x | | | | | | |
| test_qwen3_5.py | Qwen3.5-35B-A3B | x | x | x | x | | | | |
| test_deepseek.py | DeepSeek-V3.1 | | | x | x | | | | x |
| test_kimi.py | Kimi-K2 | x | x | | | | | x | |
| test_gpt_oss.py | GPT-OSS-20B | x | x | | | x | x | | |
| test_nemotron.py | Nemotron-30B | x | x | | | | | | |
| test_qwen3_vl.py | Qwen3-VL-30B-A3B | x | x | x | | | | | |

## Adding a new model

1. Create `test_<model>.py` following the pattern in existing files
2. Add appropriate test classes: `TestMerge`, `TestAdapter`, `TestQuantized`
3. Use fixtures from `conftest.py`: `train_one_step`, `download_adapter`, etc.
4. Update the coverage matrix above
