"""E2e adapter tests for Kimi-K2: separate per-expert expansion (DeepSeek architecture)."""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, DeepseekV3Config

from tests.weights.conftest import (
    run_build_adapter,
    save_expert_adapter,
)

# ---------------------------------------------------------------------------
# Kimi-K2 — DeepSeek architecture with model_type=kimi_k2
# ---------------------------------------------------------------------------


def _make_tiny_kimi_k2_config() -> DeepseekV3Config:
    """Create a tiny Kimi-K2 config.

    Kimi-K2 uses DeepseekV3ForCausalLM architecture but with model_type
    "kimi_k2". We create a DeepseekV3Config directly (the HF repo requires
    trust_remote_code and the custom code has import compat issues).
    """
    return DeepseekV3Config(
        model_type="kimi_k2",
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        first_k_dense_replace=0,
        vocab_size=256,
    )


def _save_kimi_model_to_disk(config: DeepseekV3Config, path: Path) -> None:
    """Save a tiny Kimi-K2 model, patching model_type to kimi_k2.

    DeepseekV3Config.save_pretrained writes model_type="deepseek_v3",
    but real Kimi-K2 checkpoints have model_type="kimi_k2". We patch the
    saved config.json so that detect_merge_profile routes correctly
    (falls through to default, not DeepSeek which is blocked).
    """
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model.save_pretrained(path)

    # Patch model_type in saved config.json
    config_path = path / "config.json"
    config_dict = json.loads(config_path.read_text())
    config_dict["model_type"] = "kimi_k2"
    config_path.write_text(json.dumps(config_dict, indent=2) + "\n")

    # Save a minimal tokenizer so tests don't need HF downloads
    (path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"})
    )
    (path / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
                "added_tokens": [],
            }
        )
    )


class TestKimiK2Adapter:
    """Kimi-K2: separate per-expert expansion with DeepSeek architecture.

    Kimi-K2 has model_type "kimi_k2" (not "deepseek_v3"), so it falls
    through to the default merge profile rather than being blocked as
    unsupported DeepSeek.
    """

    def test_expert_expansion_with_kimi_model_type(self) -> None:
        config = _make_tiny_kimi_k2_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "peft",
            )

            _save_kimi_model_to_disk(config, model_path)

            # Read expert dims from saved model
            orig_tensors = load_file(str(model_path / "model.safetensors"))
            gate_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
            if gate_key in orig_tensors:
                expert_out_dim, expert_in_dim = orig_tensors[gate_key].shape
            else:
                # Transformers 5.x may save fused keys
                fused_key = "model.layers.0.mlp.experts.gate_up_proj"
                num_exp, fused_dim, expert_in_dim = orig_tensors[fused_key].shape
                expert_out_dim = fused_dim // 2
            num_experts = 2

            save_expert_adapter(
                adapter_path, num_experts=num_experts, in_dim=expert_in_dim, out_dim=expert_out_dim
            )
            peft_weights, peft_config = run_build_adapter(model_path, adapter_path, output_path)

            # Verify per-expert PEFT keys exist
            for i in range(num_experts):
                for proj in ("gate_proj", "up_proj"):
                    key = f"base_model.model.model.layers.0.mlp.experts.{i}.{proj}.lora_A.weight"
                    assert key in peft_weights, f"Missing {key}"
                    assert peft_weights[key].ndim == 2

            assert peft_config["peft_type"] == "LORA"
            assert "gate_proj" in peft_config["target_modules"]

    def test_kimi_not_blocked_as_deepseek(self) -> None:
        """Kimi-K2 (model_type=kimi_k2) should NOT be blocked by the
        DeepSeek unsupported check (model_family=deepseek)."""
        config = _make_tiny_kimi_k2_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "peft",
            )

            _save_kimi_model_to_disk(config, model_path)
            orig_tensors = load_file(str(model_path / "model.safetensors"))

            # Use a dense adapter (attention-only, simpler)
            q_key = "model.layers.0.self_attn.q_a_proj.weight"
            out_dim, in_dim = orig_tensors[q_key].shape
            rank = 1
            from safetensors.torch import save_file

            weights = {
                "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight": torch.ones(
                    rank, in_dim
                ),
                "base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.weight": torch.ones(
                    out_dim, rank
                ),
            }
            adapter_path.mkdir(parents=True)
            save_file(weights, str(adapter_path / "adapter_model.safetensors"))
            (adapter_path / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 1, "r": rank})
            )

            # Should succeed (not raise WeightsAdapterError)
            peft_weights, peft_config = run_build_adapter(model_path, adapter_path, output_path)
            assert "q_a_proj" in peft_config["target_modules"]
