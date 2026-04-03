"""GPU e2e tests for GPT-OSS models.

Covers: shard merge (MXFP4 dequant/requant), PEFT adapter export.
Model: openai/gpt-oss-20b (MXFP4 block-quantized experts, interleaved gate_up_proj).
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

MODEL = "openai/gpt-oss-20b"
RENDERER = "gpt_oss_no_sysprompt"


@pytest.fixture(scope="module")
def adapter_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("gpt_oss")
    tinker_path = train_one_step(MODEL, RENDERER, "gpt_oss_gpu_e2e")
    return download_adapter(tinker_path, root / "adapter")


class TestMerge:
    def test_shard_merge(self, adapter_dir, tmp_path):
        """Shard merge with MXFP4 block-quantized expert weights."""
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

        # Expert weights should remain in MXFP4 blocks format
        blocks_keys = [k for k in tensors if k.endswith("_blocks")]
        assert len(blocks_keys) > 0, "Expected MXFP4 _blocks keys in output"

        # Each blocks key should have a matching scales key
        for key in blocks_keys:
            scales_key = key.replace("_blocks", "_scales")
            assert scales_key in tensors, f"Missing scales for {key}"
            assert tensors[key].dtype == torch.uint8
            assert tensors[scales_key].dtype == torch.uint8

    def test_shard_merge_gpu(self, adapter_dir, tmp_path):
        """Shard merge with GPU-accelerated MXFP4 dequant/requant."""
        output = tmp_path / "merged_gpu"
        build_hf_model(
            base_model=MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
            merge_strategy="shard",
            trust_remote_code=True,
            device="cuda",
        )
        verify_merged_model(output)

        tensors = load_all_tensors(output)
        blocks_keys = [k for k in tensors if k.endswith("_blocks")]
        assert len(blocks_keys) > 0, "Expected MXFP4 _blocks keys in output"


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


# vLLM serving: MXFP4+LoRA is not yet supported in vLLM.
# Adapter conversion works, but vLLM cannot load the base model with LoRA adapters.
