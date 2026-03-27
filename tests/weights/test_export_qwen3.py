"""E2e export tests for Qwen3 family: dense, MoE, and VL MoE."""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PretrainedConfig,
)

from tests.weights.conftest import (
    FILL_A,
    FILL_B,
    run_build_and_reload,
    save_dense_adapter,
    save_expert_adapter,
    save_model_to_disk,
)
from tinker_cookbook.weights import build_hf_model

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
    # Use asymmetric moe_intermediate_size (≠ hidden_size) to catch
    # transposition bugs — real Qwen3-VL has hidden=2048, moe_inter=768.
    tc.moe_intermediate_size = 48
    tc.num_attention_heads = 2
    tc.num_key_value_heads = 2
    config.vision_config.num_hidden_layers = 1
    config.vision_config.hidden_size = 64
    config.vision_config.intermediate_size = 64
    config.vision_config.num_attention_heads = 2
    return config


class TestQwen3VlMoeFusedConcatenated:
    """Qwen3-VL MoE: gate_up_proj with concatenated layout [gate | up].

    Also tests the vision model language_model prefix remapping.
    """

    FUSED_KEY = "model.language_model.layers.0.mlp.experts.gate_up_proj"

    def test_gate_and_up_deltas_in_correct_halves(self):
        config = _make_tiny_qwen3_vl_moe_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "merged",
            )

            save_model_to_disk(
                config,
                model_path,
                tokenizer_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                is_vision=True,
            )
            orig = AutoModelForImageTextToText.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_fused = orig.state_dict()[self.FUSED_KEY].clone()
            num_experts, in_dim, fused_dim = orig_fused.shape
            sz = fused_dim // 2

            save_expert_adapter(
                adapter_path,
                num_experts=num_experts,
                in_dim=in_dim,
                out_dim=sz,
            )

            merged_sd = run_build_and_reload(model_path, adapter_path, output_path, is_vision=True)

            delta = merged_sd[self.FUSED_KEY] - orig_fused
            gate_half = delta[:, :, :sz]
            up_half = delta[:, :, sz:]

            assert torch.allclose(gate_half, torch.full_like(gate_half, FILL_A), atol=1e-3)
            assert torch.allclose(up_half, torch.full_like(up_half, FILL_B), atol=1e-3)

    def test_up_only_does_not_modify_gate_half(self):
        config = _make_tiny_qwen3_vl_moe_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "merged",
            )

            save_model_to_disk(
                config,
                model_path,
                tokenizer_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                is_vision=True,
            )
            orig = AutoModelForImageTextToText.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_fused = orig.state_dict()[self.FUSED_KEY].clone()
            num_experts, in_dim, fused_dim = orig_fused.shape
            sz = fused_dim // 2

            # Only w3 (up) adapter
            prefix = "base_model.model.model.layers.0.mlp.experts"
            rank = 1
            weights = {
                f"{prefix}.w3.lora_A.weight": torch.ones(num_experts, rank, in_dim) * FILL_B,
                f"{prefix}.w3.lora_B.weight": torch.ones(num_experts, sz, rank),
            }
            adapter_path.mkdir(parents=True)
            save_file(weights, str(adapter_path / "adapter_model.safetensors"))
            (adapter_path / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 1, "r": rank})
            )

            merged_sd = run_build_and_reload(model_path, adapter_path, output_path, is_vision=True)

            orig_gate = orig_fused[:, :, :sz]
            merged_gate = merged_sd[self.FUSED_KEY][:, :, :sz]

            assert torch.allclose(merged_gate, orig_gate, atol=1e-3), (
                "up adapter modified gate half"
            )

    def test_down_proj_with_broadcast_w2(self):
        """w2 (down_proj) uses reversed broadcast: A per-expert, B shared."""
        config = _make_tiny_qwen3_vl_moe_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "merged",
            )

            save_model_to_disk(
                config,
                model_path,
                tokenizer_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                is_vision=True,
            )
            down_key = "model.language_model.layers.0.mlp.experts.down_proj"
            orig = AutoModelForImageTextToText.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_down = orig.state_dict()[down_key].clone()

            save_expert_adapter(
                adapter_path,
                num_experts=orig_down.shape[0],
                in_dim=orig_down.shape[1],
                out_dim=orig_down.shape[2],
                down_fill=0.03,
                layer_prefix="base_model.model.model.layers.0.mlp.experts",
            )

            merged_sd = run_build_and_reload(model_path, adapter_path, output_path, is_vision=True)
            delta = merged_sd[down_key] - orig_down
            # gate/up fill defaults produce gate+up deltas too; check down separately
            assert delta.abs().sum() > 0, "down_proj delta not applied"


# ---------------------------------------------------------------------------
# Qwen3 MoE — expert weights
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


class TestQwen3MoeExperts:
    """Qwen3 MoE: expert weight merge.

    HuggingFace safetensors stores separate per-expert weights
    (experts.0.gate_proj.weight), but transformers 5.x may return fused
    keys (experts.gate_up_proj) from state_dict(). save_pretrained()
    always writes separate keys regardless of version.

    This test reads dims from the saved safetensors (stable format) and
    verifies deltas via the saved files (not state_dict).
    """

    def test_expert_weights_updated(self):
        config = _make_tiny_qwen3_moe_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "merged",
            )

            save_model_to_disk(config, model_path, tokenizer_name="Qwen/Qwen3-30B-A3B")

            # Read dims from saved safetensors (stable separate format)
            orig_tensors = load_file(str(model_path / "model.safetensors"))
            gate_shape = orig_tensors["model.layers.0.mlp.experts.0.gate_proj.weight"].shape
            expert_out_dim, expert_in_dim = gate_shape
            num_experts = 2

            save_expert_adapter(
                adapter_path, num_experts=num_experts, in_dim=expert_in_dim, out_dim=expert_out_dim
            )

            build_hf_model(
                base_model=str(model_path),
                adapter_path=str(adapter_path),
                output_path=str(output_path),
            )

            # Compare saved safetensors directly (avoids state_dict format issues)
            merged_tensors = load_file(str(output_path / "model.safetensors"))

            for i in range(num_experts):
                gate_key = f"model.layers.0.mlp.experts.{i}.gate_proj.weight"
                up_key = f"model.layers.0.mlp.experts.{i}.up_proj.weight"

                gate_delta = (merged_tensors[gate_key] - orig_tensors[gate_key]).abs().sum()
                up_delta = (merged_tensors[up_key] - orig_tensors[up_key]).abs().sum()

                assert gate_delta > 0, f"Expert {i} gate_proj not updated"
                assert up_delta > 0, f"Expert {i} up_proj not updated"


# ---------------------------------------------------------------------------
# Qwen3 dense — standard linear layers (no experts)
# ---------------------------------------------------------------------------


def _make_tiny_qwen3_dense_config() -> PretrainedConfig:
    config = AutoConfig.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    config.num_hidden_layers = 1
    config.hidden_size = 64
    config.intermediate_size = 64
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    if hasattr(config, "layer_types") and config.layer_types is not None:
        config.layer_types = config.layer_types[:1]
    return config


class TestQwen3Dense:
    """Qwen3 dense: standard MLP with gate_proj/up_proj (no experts)."""

    def test_dense_linear_merge(self):
        config = _make_tiny_qwen3_dense_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "merged",
            )

            save_model_to_disk(config, model_path, tokenizer_name="Qwen/Qwen3-8B")
            orig = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_gate = orig.state_dict()["model.layers.0.mlp.gate_proj.weight"].clone()

            save_dense_adapter(adapter_path, in_dim=64, out_dim=64, fill=FILL_A)
            merged_sd = run_build_and_reload(model_path, adapter_path, output_path)

            delta = (merged_sd["model.layers.0.mlp.gate_proj.weight"] - orig_gate).abs().sum()
            assert delta > 0, "Dense gate_proj not updated"
