"""GPU e2e tests for Kimi K2 models.

Covers: shard merge with INT4 dequant/requant, PEFT adapter export.
Model: moonshotai/Kimi-K2-Thinking (INT4 pack-quantized experts).

Kimi K2 uses compressed-tensors pack-quantized format where routed expert
weights are stored as INT4 packed into int32 with per-group scales. The
merge pipeline must dequantize → apply LoRA → requantize atomically.
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

MODEL = "moonshotai/Kimi-K2-Thinking"
RENDERER = "kimi_k2"


@pytest.fixture(scope="module")
def adapter_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("kimi")
    tinker_path = train_one_step(MODEL, RENDERER, "kimi_gpu_e2e")
    return download_adapter(tinker_path, root / "adapter")


# ---------------------------------------------------------------------------
# Merge (INT4 dequant → merge → requant)
# ---------------------------------------------------------------------------


class TestMerge:
    def test_shard_merge(self, adapter_dir, tmp_path):
        """Shard merge with INT4 packed expert weights."""
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

        # Expert weights should remain in packed INT4 format after merge
        packed_keys = [k for k in tensors if k.endswith(".weight_packed")]
        assert len(packed_keys) > 0, "Expected packed INT4 expert weights"

        # Each packed weight should have scale and shape tensors
        for key in packed_keys:
            base = key.removesuffix(".weight_packed")
            assert f"{base}.weight_scale" in tensors, f"Missing scale for {key}"
            assert f"{base}.weight_shape" in tensors, f"Missing shape for {key}"

        # Packed tensors should be int32
        for key in packed_keys:
            assert tensors[key].dtype == torch.int32, (
                f"Expected int32 for packed weight {key}, got {tensors[key].dtype}"
            )


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
        assert config.get("r") == LORA_RANK
