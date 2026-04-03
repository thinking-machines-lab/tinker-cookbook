"""GPU e2e tests for Qwen3 dense models.

Covers: shard merge, PEFT adapter export.
Model: Qwen/Qwen3-4B-Instruct-2507 (dense transformer, qwen3 renderer).
"""

from __future__ import annotations

import json

import pytest
import torch

from tests.weights.gpu.conftest import (
    LORA_RANK,
    download_adapter,
    load_all_tensors,
    skip_no_vllm,
    train_one_step,
    verify_merged_model,
    vllm_generate,
)
from tinker_cookbook.weights import build_hf_model, build_lora_adapter

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
RENDERER = "qwen3_instruct"


@pytest.fixture(scope="module")
def adapter_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("qwen3")
    tinker_path = train_one_step(MODEL, RENDERER, "qwen3_gpu_e2e")
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
        weight_keys = [k for k in tensors if k.endswith(".weight")]
        assert all(tensors[k].dtype in (torch.bfloat16, torch.float32) for k in weight_keys), (
            "Expected all weights to be BF16/FP32 for dense model"
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


# ---------------------------------------------------------------------------
# vLLM serving
# ---------------------------------------------------------------------------


class TestVllmServing:
    @skip_no_vllm
    def test_adapter_serves_in_vllm(self, adapter_dir, tmp_path):
        """Export adapter to PEFT, load in vLLM, generate text."""
        peft_dir = tmp_path / "peft"
        build_lora_adapter(
            base_model=MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(peft_dir),
        )
        vllm_generate(MODEL, peft_dir)
