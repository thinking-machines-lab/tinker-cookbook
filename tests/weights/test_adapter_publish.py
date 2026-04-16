"""E2E test: build_lora_adapter → publish_to_hf_hub → download and verify.

Builds a synthetic PEFT adapter from a tiny model, publishes it to a
temporary private HuggingFace Hub repo, downloads the files back, and
verifies they match the local originals.

Requires HF authentication (HF_TOKEN env var or ``hf auth login``).
No TINKER_API_KEY needed — uses synthetic adapter data.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, PretrainedConfig

from tests.weights.conftest import (
    save_dense_adapter,
    save_model_to_disk,
)
from tests.weights.test_publish import _managed_hf_repo
from tinker_cookbook.weights import build_lora_adapter, publish_to_hf_hub


def _make_tiny_qwen3_dense_config() -> PretrainedConfig:
    """Create a minimal Qwen3 config for a single-layer dense model."""
    config = AutoConfig.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    config.num_hidden_layers = 1
    config.hidden_size = 64
    config.intermediate_size = 64
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    return config


@pytest.mark.integration
class TestAdapterPublishE2E:
    def test_build_publish_download_adapter(self) -> None:
        """Full round-trip: build PEFT adapter → publish to Hub → download and compare."""
        from huggingface_hub import hf_hub_download

        with _managed_hf_repo() as (repo_id, api):
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                model_path = root / "model"
                adapter_path = root / "adapter"
                peft_path = root / "peft"

                # 1. Create tiny model and synthetic LoRA adapter.
                config = _make_tiny_qwen3_dense_config()
                save_model_to_disk(config, model_path, tokenizer_name="Qwen/Qwen3-8B")

                orig_tensors = load_file(str(model_path / "model.safetensors"))
                gate_shape = orig_tensors["model.layers.0.mlp.gate_proj.weight"].shape
                out_dim, in_dim = gate_shape

                save_dense_adapter(adapter_path, in_dim=in_dim, out_dim=out_dim)

                # 2. Convert to PEFT format.
                build_lora_adapter(
                    base_model=str(model_path),
                    adapter_path=str(adapter_path),
                    output_path=str(peft_path),
                )

                # 3. Snapshot local PEFT output.
                local_weights = load_file(str(peft_path / "adapter_model.safetensors"))
                with open(peft_path / "adapter_config.json") as f:
                    local_config = json.load(f)

                # 4. Publish to Hub.
                url = publish_to_hf_hub(
                    model_path=str(peft_path),
                    repo_id=repo_id,
                    private=True,
                )
                assert url == f"https://huggingface.co/{repo_id}"

                # 5. Verify files exist on Hub.
                files = api.list_repo_files(repo_id=repo_id, repo_type="model")
                assert "adapter_model.safetensors" in files
                assert "adapter_config.json" in files

                # 6. Download and compare adapter_config.json.
                downloaded_config_path = hf_hub_download(
                    repo_id=repo_id, filename="adapter_config.json"
                )
                with open(downloaded_config_path) as f:
                    downloaded_config = json.load(f)
                assert downloaded_config == local_config

                # 7. Download and compare adapter_model.safetensors.
                downloaded_weights_path = hf_hub_download(
                    repo_id=repo_id, filename="adapter_model.safetensors"
                )
                downloaded_weights = load_file(downloaded_weights_path)
                assert downloaded_weights.keys() == local_weights.keys()
                for key in local_weights:
                    assert torch.equal(local_weights[key], downloaded_weights[key]), (
                        f"Tensor mismatch for {key}"
                    )
