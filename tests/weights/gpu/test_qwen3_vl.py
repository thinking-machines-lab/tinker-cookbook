"""GPU e2e tests for Qwen3-VL MoE models (vision-language).

Covers: shard merge, PEFT adapter export, FP8 experts quantized merge.
Model: Qwen/Qwen3-VL-30B-A3B-Instruct (VL MoE with language_model prefix).

Tests the vision model prefix path (model.language_model.*) which requires
special handling in adapter key remapping.
"""

from __future__ import annotations

import json

import pytest

from tests.weights.gpu.conftest import (
    download_adapter,
    load_all_tensors,
    train_one_step,
    verify_fp8_output,
    verify_merged_model,
)
from tinker_cookbook.weights import build_hf_model, build_lora_adapter

MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
RENDERER = "qwen3_vl_instruct"


@pytest.fixture(scope="module")
def adapter_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("qwen3_vl")
    tinker_path = train_one_step(MODEL, RENDERER, "qwen3_vl_gpu_e2e")
    return download_adapter(tinker_path, root / "adapter")


class TestMerge:
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_shard_merge(self, adapter_dir, tmp_path, device):
        output = tmp_path / f"merged_{device}"
        build_hf_model(
            base_model=MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
            merge_strategy="shard",
            device=device,
        )
        verify_merged_model(output)

        tensors = load_all_tensors(output)
        # VL model should have language_model prefix keys
        lm_keys = [k for k in tensors if "language_model" in k]
        assert len(lm_keys) > 0, "Expected language_model prefix keys in VL model"
        # Should also have visual encoder keys
        visual_keys = [k for k in tensors if "visual" in k]
        assert len(visual_keys) > 0, "Expected visual encoder keys in VL model"
        # MoE: should have expert keys
        expert_keys = [k for k in tensors if ".experts." in k]
        assert len(expert_keys) > 0, "Expected expert weights in MoE VL model"


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


class TestQuantized:
    def test_fp8_experts(self, adapter_dir, tmp_path):
        """FP8 quantized merge for VL MoE routed experts."""
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
