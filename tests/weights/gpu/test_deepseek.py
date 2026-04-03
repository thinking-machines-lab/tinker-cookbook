"""GPU e2e tests for DeepSeek V3.1 models.

Covers: FP8 quantized merge (CPU + GPU), CPU/GPU equivalence.
Model: deepseek-ai/DeepSeek-V3.1 (native FP8 checkpoint, separate experts).

DeepSeek V3.1 is a 671B model — these tests require significant disk space
and time for model weight downloads. The merge itself is shard-by-shard so
memory usage is bounded.
"""

from __future__ import annotations

import pytest
import torch

from tests.weights.gpu.conftest import (
    download_adapter,
    load_all_tensors,
    train_one_step,
    verify_fp8_output,
    verify_merged_model,
)
from tinker_cookbook.weights import build_hf_model

MODEL = "deepseek-ai/DeepSeek-V3.1"
RENDERER = "deepseekv3"


@pytest.fixture(scope="module")
def adapter_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("deepseek")
    tinker_path = train_one_step(MODEL, RENDERER, "deepseek_gpu_e2e")
    return download_adapter(tinker_path, root / "adapter")


class TestQuantized:
    def test_fp8_experts_cpu(self, adapter_dir, tmp_path):
        """FP8 quantized merge on CPU — baseline correctness."""
        output = tmp_path / "merged_cpu"
        build_hf_model(
            base_model=MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
            quantize="experts-fp8",
            serving_format="vllm",
            device="cpu",
        )
        verify_merged_model(output, expect_config_key="compression_config")
        verify_fp8_output(output)

    def test_fp8_experts_gpu(self, adapter_dir, tmp_path):
        """FP8 quantized merge on GPU."""
        output = tmp_path / "merged_gpu"
        build_hf_model(
            base_model=MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
            quantize="experts-fp8",
            serving_format="vllm",
            device="cuda",
        )
        verify_merged_model(output, expect_config_key="compression_config")
        verify_fp8_output(output)

    def test_cpu_gpu_equivalence(self, adapter_dir, tmp_path):
        """CPU and GPU produce identical FP8 output."""
        cpu_out = tmp_path / "cpu"
        gpu_out = tmp_path / "gpu"

        build_hf_model(
            base_model=MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(cpu_out),
            quantize="experts-fp8",
            serving_format="vllm",
            device="cpu",
        )
        build_hf_model(
            base_model=MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(gpu_out),
            quantize="experts-fp8",
            serving_format="vllm",
            device="cuda",
        )

        cpu_tensors = load_all_tensors(cpu_out)
        gpu_tensors = load_all_tensors(gpu_out)

        assert set(cpu_tensors.keys()) == set(gpu_tensors.keys()), "Key sets differ"

        for key in cpu_tensors:
            cpu_t = cpu_tensors[key]
            gpu_t = gpu_tensors[key]
            assert cpu_t.shape == gpu_t.shape, f"Shape mismatch: {key}"
            assert cpu_t.dtype == gpu_t.dtype, f"Dtype mismatch: {key}"
            if cpu_t.dtype == torch.float8_e4m3fn:
                assert torch.equal(cpu_t.to(torch.float32), gpu_t.to(torch.float32)), (
                    f"FP8 mismatch: {key}"
                )
            else:
                assert torch.equal(cpu_t, gpu_t), f"Tensor mismatch: {key}"


# vLLM serving: DeepSeek V3/V3.1 LoRA adapter serving is intentionally
# unsupported — the model uses adapter-free FP8 quantized merge instead.
