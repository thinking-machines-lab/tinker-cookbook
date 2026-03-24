"""E2e adapter tests for Qwen3 family: dense, MoE, and VL MoE."""

import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    PretrainedConfig,
)

from tests.weights.conftest import (
    run_build_adapter,
    save_dense_adapter,
    save_expert_adapter,
    save_model_to_disk,
)
from tinker_cookbook.weights import build_hf_model

# ---------------------------------------------------------------------------
# Qwen3 dense — standard linear layers
# ---------------------------------------------------------------------------


def _make_tiny_qwen3_dense_config() -> PretrainedConfig:
    config = AutoConfig.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    config.num_hidden_layers = 1
    config.hidden_size = 64
    config.intermediate_size = 64
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    return config


class TestQwen3DenseAdapter:
    """Qwen3 dense: standard linear layers, no special remapping."""

    def test_peft_keys_match_hf_model_params(self) -> None:
        config = _make_tiny_qwen3_dense_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "peft",
            )

            save_model_to_disk(config, model_path, tokenizer_name="Qwen/Qwen3-8B")

            # Read actual model dims
            orig_tensors = load_file(str(model_path / "model.safetensors"))
            gate_shape = orig_tensors["model.layers.0.mlp.gate_proj.weight"].shape
            out_dim, in_dim = gate_shape

            save_dense_adapter(adapter_path, in_dim=in_dim, out_dim=out_dim)
            peft_weights, peft_config = run_build_adapter(model_path, adapter_path, output_path)

            # PEFT keys should reference actual model parameter paths
            assert "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight" in peft_weights
            assert "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight" in peft_weights
            assert peft_config["peft_type"] == "LORA"
            assert "gate_proj" in peft_config["target_modules"]

    def test_mathematical_equivalence_with_merge(self) -> None:
        """Verify adapter conversion is lossless: merge path == adapter + manual delta."""
        config = _make_tiny_qwen3_dense_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path = root / "model"
            adapter_path = root / "adapter"
            merged_path = root / "merged"
            peft_path = root / "peft"

            save_model_to_disk(config, model_path, tokenizer_name="Qwen/Qwen3-8B")
            orig_tensors = load_file(str(model_path / "model.safetensors"))
            gate_shape = orig_tensors["model.layers.0.mlp.gate_proj.weight"].shape
            out_dim, in_dim = gate_shape

            save_dense_adapter(adapter_path, in_dim=in_dim, out_dim=out_dim)

            # Path 1: merge into base model
            build_hf_model(
                base_model=str(model_path),
                adapter_path=str(adapter_path),
                output_path=str(merged_path),
            )
            merged_tensors = load_file(str(merged_path / "model.safetensors"))

            # Path 2: build PEFT adapter, manually apply delta
            peft_weights, peft_config = run_build_adapter(model_path, adapter_path, peft_path)
            alpha = peft_config["lora_alpha"]
            rank = peft_config["r"]
            scaling = alpha / rank

            lora_A = peft_weights["base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight"]
            lora_B = peft_weights["base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight"]
            delta = (lora_B.float() @ lora_A.float()) * scaling
            manual_merged = orig_tensors["model.layers.0.mlp.gate_proj.weight"].float() + delta

            assert torch.allclose(
                merged_tensors["model.layers.0.mlp.gate_proj.weight"].float(),
                manual_merged,
                atol=1e-3,
            ), "Merge path and adapter+manual delta path should produce identical results"


# ---------------------------------------------------------------------------
# Qwen3 MoE — separate per-expert expansion
# ---------------------------------------------------------------------------


def _make_tiny_qwen3_moe_config() -> PretrainedConfig:
    config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)
    config.num_hidden_layers = 1
    config.num_experts = 2
    config.num_experts_per_tok = 1
    config.hidden_size = 64
    config.intermediate_size = 64
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    return config


class TestQwen3MoeAdapter:
    """Qwen3 MoE: 3D expert tensors expanded to per-expert 2D PEFT keys."""

    def test_expert_expansion(self) -> None:
        config = _make_tiny_qwen3_moe_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "peft",
            )

            save_model_to_disk(config, model_path, tokenizer_name="Qwen/Qwen3-30B-A3B")
            orig_tensors = load_file(str(model_path / "model.safetensors"))
            gate_shape = orig_tensors["model.layers.0.mlp.experts.0.gate_proj.weight"].shape
            expert_out_dim, expert_in_dim = gate_shape
            num_experts = 2

            save_expert_adapter(
                adapter_path, num_experts=num_experts, in_dim=expert_in_dim, out_dim=expert_out_dim
            )
            peft_weights, peft_config = run_build_adapter(model_path, adapter_path, output_path)

            # Per-expert keys should exist
            for i in range(num_experts):
                for proj in ("gate_proj", "up_proj"):
                    key_a = f"base_model.model.model.layers.0.mlp.experts.{i}.{proj}.lora_A.weight"
                    key_b = f"base_model.model.model.layers.0.mlp.experts.{i}.{proj}.lora_B.weight"
                    assert key_a in peft_weights, f"Missing {key_a}"
                    assert key_b in peft_weights, f"Missing {key_b}"
                    assert peft_weights[key_a].ndim == 2
                    assert peft_weights[key_b].ndim == 2

            assert "gate_proj" in peft_config["target_modules"]
            assert "up_proj" in peft_config["target_modules"]


# ---------------------------------------------------------------------------
# Qwen3-VL MoE — fused concatenated gate_up_proj + vision prefix
# ---------------------------------------------------------------------------


def _make_tiny_qwen3_vl_moe_config() -> PretrainedConfig:
    config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct", trust_remote_code=True)
    tc = config.text_config
    tc.num_hidden_layers = 1
    tc.num_experts = 2
    tc.num_experts_per_tok = 1
    tc.hidden_size = 64
    tc.intermediate_size = 64
    tc.num_attention_heads = 2
    tc.num_key_value_heads = 2
    config.vision_config.num_hidden_layers = 1
    config.vision_config.hidden_size = 64
    config.vision_config.intermediate_size = 64
    config.vision_config.num_attention_heads = 2
    return config


class TestQwen3VlMoeAdapter:
    """Qwen3-VL MoE: vision prefix + expert expansion."""

    def test_vision_prefix_and_expert_expansion(self) -> None:
        config = _make_tiny_qwen3_vl_moe_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "peft",
            )

            save_model_to_disk(
                config,
                model_path,
                tokenizer_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                is_vision=True,
            )
            orig_tensors = load_file(str(model_path / "model.safetensors"))

            # Transformers 5.x saves fused expert keys (gate_up_proj).
            # Read dims from the fused key.
            fused_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
            if fused_key in orig_tensors:
                num_experts, fused_dim, expert_in_dim = orig_tensors[fused_key].shape
                expert_out_dim = fused_dim // 2
            else:
                # Older transformers: per-expert keys
                gate_key = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
                gate_shape = orig_tensors[gate_key].shape
                expert_out_dim, expert_in_dim = gate_shape
                num_experts = 2

            # Adapter uses model.layers... (without language_model prefix)
            save_expert_adapter(
                adapter_path, num_experts=num_experts, in_dim=expert_in_dim, out_dim=expert_out_dim
            )
            peft_weights, peft_config = run_build_adapter(model_path, adapter_path, output_path)

            # PEFT keys should have model.language_model prefix
            for i in range(num_experts):
                key = f"base_model.model.model.language_model.layers.0.mlp.experts.{i}.gate_proj.lora_A.weight"
                assert key in peft_weights, f"Missing vision-prefixed expert key: {key}"

            assert peft_config["peft_type"] == "LORA"
