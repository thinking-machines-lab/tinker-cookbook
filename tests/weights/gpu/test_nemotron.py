"""GPU e2e tests for Nemotron models.

Covers: shard merge, PEFT adapter export.
Model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B (fused projections, Mamba layers).
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

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER = "nemotron3"
# Nemotron uses custom model code that requires trust_remote_code
TRUST_REMOTE_CODE = True


@pytest.fixture(scope="module")
def adapter_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("nemotron")
    tinker_path = train_one_step(MODEL, RENDERER, "nemotron_gpu_e2e")
    return download_adapter(tinker_path, root / "adapter")


class TestMerge:
    @pytest.mark.xfail(reason="load_config_dict doesn't forward trust_remote_code to AutoConfig — needs fix")
    def test_shard_merge(self, adapter_dir, tmp_path):
        output = tmp_path / "merged"
        build_hf_model(
            base_model=MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
            merge_strategy="shard",
            trust_remote_code=True,
        )
        verify_merged_model(output)


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
