"""GPU e2e tests for Kimi K2.5 models (VL + INT4).

Covers: shard merge with INT4 dequant/requant, PEFT adapter export.
Model: moonshotai/Kimi-K2.5 (VL + INT4 pack-quantized experts).

Kimi K2.5 combines the vision-language prefix (model.language_model.*)
with INT4 compressed-tensors pack-quantized expert weights — a unique
combination that exercises both the VL key remapping and INT4 hooks.
"""

from __future__ import annotations

import json

import pytest
import torch

from tests.weights.gpu.conftest import (
    LORA_RANK,
    download_adapter,
    load_all_tensors,
    train_one_step,
    verify_merged_model,
)
from tinker_cookbook.weights import build_hf_model, build_lora_adapter

MODEL = "moonshotai/Kimi-K2.5"
RENDERER = "kimi_k25"


@pytest.fixture(scope="module")
def adapter_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("kimi_k25")
    tinker_path = train_one_step(MODEL, RENDERER, "kimi_k25_gpu_e2e")
    return download_adapter(tinker_path, root / "adapter")


class TestMerge:
    def test_shard_merge(self, adapter_dir, tmp_path):
        """Shard merge with VL prefix + INT4 packed expert weights."""
        output = tmp_path / "merged"
        build_hf_model(
            base_model=MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
            merge_strategy="shard",
            trust_remote_code=True,
        )
        verify_merged_model(output)

        tensors = load_all_tensors(output)

        # VL model should have language_model prefix keys
        lm_keys = [k for k in tensors if "language_model" in k]
        assert len(lm_keys) > 0, "Expected language_model prefix keys in VL model"

        # Expert weights should remain in packed INT4 format
        packed_keys = [k for k in tensors if k.endswith(".weight_packed")]
        assert len(packed_keys) > 0, "Expected packed INT4 expert weights"

        for key in packed_keys:
            base = key.removesuffix(".weight_packed")
            assert f"{base}.weight_scale" in tensors, f"Missing scale for {key}"
            assert f"{base}.weight_shape" in tensors, f"Missing shape for {key}"
            assert tensors[key].dtype == torch.int32


class TestAdapter:
    def test_lora_adapter_export(self, adapter_dir, tmp_path):
        output = tmp_path / "peft"
        build_lora_adapter(
            base_model=MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
            trust_remote_code=True,
        )
        assert (output / "adapter_config.json").exists()
        config = json.loads((output / "adapter_config.json").read_text())
        assert "peft_type" in config
        assert config.get("r") == LORA_RANK


# vLLM serving: vLLM 0.18 lacks LoRA support for KimiK25ForConditionalGeneration.
