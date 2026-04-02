"""GPU e2e tests for DeepSeek V3.1 models.

Covers: FP8 quantized merge (CPU + GPU), CPU/GPU equivalence.
Model: deepseek-ai/DeepSeek-V3.1 (native FP8 checkpoint, separate experts).

DeepSeek V3.1 is a 671B model — these tests require significant disk space
and time for model weight downloads. The merge itself is shard-by-shard so
memory usage is bounded.
"""

from __future__ import annotations

import json
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Quantized merge (FP8 experts)
# ---------------------------------------------------------------------------


def _build_hf_model_with_device(device: str | None = None, **kwargs) -> None:
    """Call build_hf_model, passing device only if the API supports it."""
    import inspect

    sig = inspect.signature(build_hf_model)
    if "device" in sig.parameters and device is not None:
        kwargs["device"] = device
    build_hf_model(**kwargs)


class TestQuantized:
    def test_fp8_experts(self, adapter_dir, tmp_path):
        """FP8 quantized merge — baseline correctness."""
        output = tmp_path / "merged"
        _build_hf_model_with_device(
            base_model=MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
            quantize="experts-fp8",
            serving_format="vllm",
        )
        verify_merged_model(output, expect_config_key="compression_config")
        verify_fp8_output(output)

    def test_fp8_experts_gpu(self, adapter_dir, tmp_path):
        """FP8 quantized merge on GPU — verify GPU path works."""
        output = tmp_path / "merged_gpu"
        _build_hf_model_with_device(
            device="cuda",
            base_model=MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
            quantize="experts-fp8",
            serving_format="vllm",
        )
        verify_merged_model(output, expect_config_key="compression_config")
        verify_fp8_output(output)

    def test_cpu_gpu_equivalence(self, adapter_dir, tmp_path):
        """CPU and GPU produce identical FP8 output."""
        cpu_out = tmp_path / "cpu"
        gpu_out = tmp_path / "gpu"

        _build_hf_model_with_device(
            device="cpu",
            base_model=MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(cpu_out),
            quantize="experts-fp8",
            serving_format="vllm",
        )
        _build_hf_model_with_device(
            device="cuda",
            base_model=MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(gpu_out),
            quantize="experts-fp8",
            serving_format="vllm",
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
