"""GPU e2e tests for Qwen3.5 MoE models.

Covers: shard merge, PEFT adapter export, FP8 experts quantized merge.
Model: Qwen/Qwen3.5-35B-A3B (MoE with fused concatenated gate_up_proj).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tests.weights.gpu.conftest import (
    LORA_RANK,
    download_adapter,
    load_all_tensors,
    train_one_step,
    verify_fp8_output,
    verify_merged_model,
)
from tinker_cookbook.weights import build_hf_model, build_lora_adapter

MODEL = "Qwen/Qwen3.5-35B-A3B"
RENDERER = "qwen3_5"


@pytest.fixture(scope="module")
def adapter_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("qwen3_5")
    tinker_path = train_one_step(MODEL, RENDERER, "qwen35_gpu_e2e")
    return download_adapter(tinker_path, root / "adapter")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestMerge:
    def test_shard_merge(self, adapter_dir, tmp_path):
        output = tmp_path / "merged"
        build_hf_model(
            base_model=MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
            merge_strategy="shard",
        )
        verify_merged_model(output)

        tensors = load_all_tensors(output)
        expert_keys = [k for k in tensors if ".experts." in k]
        assert len(expert_keys) > 0, "Expected expert weights in MoE model"


# ---------------------------------------------------------------------------
# Adapter export
# ---------------------------------------------------------------------------


class TestAdapter:
    def test_lora_adapter_export(self, adapter_dir, tmp_path):
        output = tmp_path / "peft"
        build_lora_adapter(
            base_model=MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
        )
        assert (output / "adapter_config.json").exists()
        config = json.loads((output / "adapter_config.json").read_text())
        assert "peft_type" in config


# ---------------------------------------------------------------------------
# Quantized merge (FP8 experts)
# ---------------------------------------------------------------------------


class TestQuantized:
    def test_fp8_experts(self, adapter_dir, tmp_path):
        """FP8 quantized merge for MoE routed experts."""
        output = tmp_path / "merged_fp8"
        build_hf_model(
            base_model=MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
            quantize="experts-fp8",
            serving_format="vllm",
        )
        verify_merged_model(output, expect_config_key="compression_config")
        verify_fp8_output(output)
