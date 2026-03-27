"""E2e export tests for GPT-OSS: fused interleaved gate_up_proj."""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from tests.weights.conftest import (
    FILL_A,
    FILL_B,
    run_build_and_reload,
    save_expert_adapter,
    save_model_to_disk,
)


def _make_tiny_gpt_oss_config() -> PretrainedConfig:
    config = AutoConfig.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
    config.num_hidden_layers = 1
    config.num_local_experts = 2
    config.hidden_size = 64
    config.intermediate_size = 64
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    config.layer_types = ["full_attention"]
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    return config


class TestGptOssFusedInterleaved:
    """GPT-OSS: gate_up_proj with interleaved layout [g0, u0, g1, u1, ...]."""

    FUSED_KEY = "model.layers.0.mlp.experts.gate_up_proj"

    def test_gate_and_up_deltas_in_correct_interleaved_slots(self):
        config = _make_tiny_gpt_oss_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "merged",
            )

            save_model_to_disk(config, model_path, tokenizer_name="openai/gpt-oss-20b")
            orig = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_fused = orig.state_dict()[self.FUSED_KEY].clone()
            num_experts, in_dim, fused_dim = orig_fused.shape

            save_expert_adapter(
                adapter_path, num_experts=num_experts, in_dim=in_dim, out_dim=fused_dim // 2
            )
            merged_sd = run_build_and_reload(model_path, adapter_path, output_path)

            delta = merged_sd[self.FUSED_KEY] - orig_fused
            gate_delta = delta[:, :, 0::2]
            up_delta = delta[:, :, 1::2]

            assert torch.allclose(gate_delta, torch.full_like(gate_delta, FILL_A), atol=1e-3)
            assert torch.allclose(up_delta, torch.full_like(up_delta, FILL_B), atol=1e-3)

    def test_up_only_does_not_modify_gate_slots(self):
        config = _make_tiny_gpt_oss_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "merged",
            )

            save_model_to_disk(config, model_path, tokenizer_name="openai/gpt-oss-20b")
            orig = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_gate = orig.state_dict()[self.FUSED_KEY][:, :, 0::2].clone()
            num_experts, in_dim, fused_dim = orig.state_dict()[self.FUSED_KEY].shape

            # Save only w3 (up) adapter
            prefix = "base_model.model.model.layers.0.mlp.experts"
            rank = 1
            up_only = {
                f"{prefix}.w3.lora_A.weight": torch.ones(1, rank, in_dim) * FILL_B,
                f"{prefix}.w3.lora_B.weight": torch.ones(num_experts, fused_dim // 2, rank),
            }
            adapter_path.mkdir(parents=True)
            save_file(up_only, str(adapter_path / "adapter_model.safetensors"))
            (adapter_path / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 1, "r": rank})
            )

            merged_sd = run_build_and_reload(model_path, adapter_path, output_path)
            merged_gate = merged_sd[self.FUSED_KEY][:, :, 0::2]

            assert torch.allclose(merged_gate, orig_gate, atol=1e-3), (
                "up adapter modified gate slots"
            )
