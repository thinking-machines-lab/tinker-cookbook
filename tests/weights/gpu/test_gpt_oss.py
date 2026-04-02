"""GPU e2e tests for GPT-OSS models.

Covers: shard merge, PEFT adapter export.
Model: openai/gpt-oss-20b (interleaved fused gate_up_proj experts).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    @pytest.mark.xfail(reason="GPT-OSS expert key mapping issue with real adapters — separate experts vs fused gate_up_proj")
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
        expert_keys = [k for k in tensors if ".experts." in k or "gate_up_proj" in k]
        assert len(expert_keys) > 0, "Expected expert/fused weights in GPT-OSS"


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
