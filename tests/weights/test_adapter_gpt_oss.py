"""E2e adapter tests for GPT-OSS: .attn -> .self_attn remap + expert expansion."""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import (
    AutoConfig,
    PretrainedConfig,
)

from tests.weights.conftest import (
    FILL_A,
    run_build_adapter,
    save_model_to_disk,
)

# ---------------------------------------------------------------------------
# GPT-OSS — .attn -> .self_attn remap + expert expansion
# ---------------------------------------------------------------------------


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


class TestGptOssAdapter:
    """GPT-OSS: .attn -> .self_attn remap in PEFT output."""

    def test_attn_to_self_attn_and_expert_expansion(self) -> None:
        config = _make_tiny_gpt_oss_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "peft",
            )

            save_model_to_disk(config, model_path, tokenizer_name="openai/gpt-oss-20b")
            orig_tensors = load_file(str(model_path / "model.safetensors"))

            # Get attention dims
            q_shape = orig_tensors["model.layers.0.self_attn.q_proj.weight"].shape
            attn_out_dim, attn_in_dim = q_shape

            # Adapter with .attn naming (Tinker internal) + expert weights
            rank = 1
            fused_key = "model.layers.0.mlp.experts.gate_up_proj"
            num_experts, expert_in_dim, fused_dim = orig_tensors[fused_key].shape
            expert_out_dim = fused_dim // 2

            weights: dict[str, torch.Tensor] = {
                # Attention with .attn naming
                "base_model.model.model.layers.0.attn.q_proj.lora_A.weight": (
                    torch.ones(rank, attn_in_dim) * FILL_A
                ),
                "base_model.model.model.layers.0.attn.q_proj.lora_B.weight": torch.ones(
                    attn_out_dim, rank
                ),
                # Expert weights (broadcast pattern matches real Tinker adapters)
                "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight": (
                    torch.ones(1, rank, expert_in_dim) * FILL_A
                ),
                "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight": torch.ones(
                    num_experts, expert_out_dim, rank
                ),
            }
            adapter_path.mkdir(parents=True)
            save_file(weights, str(adapter_path / "adapter_model.safetensors"))
            (adapter_path / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 1, "r": rank})
            )

            peft_weights, peft_config = run_build_adapter(model_path, adapter_path, output_path)

            # .attn should be remapped to .self_attn
            assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in peft_weights
            assert not any(".attn." in k for k in peft_weights)

            # Expert keys should be expanded
            for i in range(num_experts):
                assert (
                    f"base_model.model.model.layers.0.mlp.experts.{i}.gate_proj.lora_A.weight"
                    in peft_weights
                )

            assert "q_proj" in peft_config["target_modules"]
            assert "gate_proj" in peft_config["target_modules"]
