"""E2e adapter tests for Qwen3.5: split QKV, tied embeddings, MoE expert expansion."""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    PretrainedConfig,
)

from tests.weights.conftest import (
    FILL_A,
    FILL_B,
    run_build_adapter,
    save_model_to_disk,
)
from tinker_cookbook.weights import build_hf_model

# ---------------------------------------------------------------------------
# Qwen3.5 dense — split QKV + vision prefix
# ---------------------------------------------------------------------------


def _make_tiny_qwen3_5_dense_config() -> PretrainedConfig:
    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    tc = config.text_config
    tc.num_hidden_layers = 1
    tc.layer_types = ["linear_attention"]  # single linear_attn layer
    tc.linear_num_key_heads = 2
    tc.linear_num_value_heads = 4
    tc.linear_key_head_dim = 8
    tc.linear_value_head_dim = 8
    tc.hidden_size = 64
    tc.intermediate_size = 64
    tc.num_attention_heads = 2
    tc.num_key_value_heads = 2
    tc.head_dim = 32
    tc.vocab_size = 256
    tc.mtp_num_hidden_layers = 0
    config.vision_config.num_hidden_layers = 1
    config.vision_config.hidden_size = 64
    config.vision_config.intermediate_size = 64
    config.vision_config.num_attention_heads = 2
    return config


def _save_qkv_adapter(
    path: Path,
    *,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    in_dim: int,
    q_fill: float = FILL_A,
    k_fill: float = FILL_A,
    v_fill: float = FILL_B,
    layer_prefix: str = "base_model.model.model.layers.0.linear_attn",
) -> None:
    """Save a LoRA adapter for split in_proj_q/k/v projections."""
    rank = 1
    weights: dict[str, torch.Tensor] = {
        f"{layer_prefix}.in_proj_q.lora_A.weight": torch.ones(rank, in_dim) * q_fill,
        f"{layer_prefix}.in_proj_q.lora_B.weight": torch.ones(q_dim, rank),
        f"{layer_prefix}.in_proj_k.lora_A.weight": torch.ones(rank, in_dim) * k_fill,
        f"{layer_prefix}.in_proj_k.lora_B.weight": torch.ones(k_dim, rank),
        f"{layer_prefix}.in_proj_v.lora_A.weight": torch.ones(rank, in_dim) * v_fill,
        f"{layer_prefix}.in_proj_v.lora_B.weight": torch.ones(v_dim, rank),
    }
    path.mkdir(parents=True)
    save_file(weights, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 1, "r": rank}))


class TestQwen35DenseAdapter:
    """Qwen3.5 dense: split in_proj_q/k/v preserved as separate PEFT keys.

    Uses the same tiny config as the merge e2e test (test_export.py) to
    ensure we get a linear_attention layer with in_proj_qkv.
    """

    FUSED_KEY = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"

    def test_split_qkv_keys_in_peft_output(self) -> None:
        config = _make_tiny_qwen3_5_dense_config()
        tc = config.text_config
        q_dim = tc.linear_num_key_heads * tc.linear_key_head_dim
        k_dim = q_dim
        v_dim = tc.linear_num_value_heads * tc.linear_value_head_dim
        in_dim = tc.hidden_size

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "peft",
            )

            save_model_to_disk(config, model_path, tokenizer_name="Qwen/Qwen3.5-4B", is_vision=True)
            _save_qkv_adapter(adapter_path, q_dim=q_dim, k_dim=k_dim, v_dim=v_dim, in_dim=in_dim)

            peft_weights, peft_config = run_build_adapter(model_path, adapter_path, output_path)

            # Split QKV should be preserved as separate PEFT keys with vision prefix.
            for proj in ("in_proj_q", "in_proj_k", "in_proj_v"):
                matching = [k for k in peft_weights if proj in k]
                assert matching, f"Missing {proj} in PEFT output"
                assert all("language_model" in k for k in matching), (
                    f"{proj} keys missing language_model prefix"
                )
                assert proj in peft_config["target_modules"]

            # All tensors should be 2D and unscaled.
            for key, tensor in peft_weights.items():
                assert tensor.ndim == 2, f"{key} is {tensor.ndim}D"

    def test_mathematical_equivalence_with_merge(self) -> None:
        """Verify the PEFT adapter produces the same delta as the merge path.

        The merge path computes: fused_qkv += [B_q@A_q; B_k@A_k; B_v@A_v] * (alpha/r)
        The adapter path should produce separate in_proj_q/k/v tensors that,
        when manually applied with the same formula, yield the identical delta.
        """
        config = _make_tiny_qwen3_5_dense_config()
        tc = config.text_config
        q_dim = tc.linear_num_key_heads * tc.linear_key_head_dim
        k_dim = q_dim
        v_dim = tc.linear_num_value_heads * tc.linear_value_head_dim
        in_dim = tc.hidden_size

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path = root / "model"
            adapter_path = root / "adapter"
            merged_path = root / "merged"
            peft_path = root / "peft"

            save_model_to_disk(config, model_path, tokenizer_name="Qwen/Qwen3.5-4B", is_vision=True)
            orig = AutoModelForImageTextToText.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_fused = orig.state_dict()[self.FUSED_KEY].clone()
            del orig

            _save_qkv_adapter(adapter_path, q_dim=q_dim, k_dim=k_dim, v_dim=v_dim, in_dim=in_dim)

            # Path 1: merge into base model
            build_hf_model(
                base_model=str(model_path),
                adapter_path=str(adapter_path),
                output_path=str(merged_path),
            )
            merged_model = AutoModelForImageTextToText.from_pretrained(
                merged_path, trust_remote_code=True, dtype=torch.float32
            )
            merged_fused = merged_model.state_dict()[self.FUSED_KEY].clone()
            del merged_model
            merge_delta = merged_fused - orig_fused

            # Path 2: build PEFT adapter, manually compute delta
            peft_weights, peft_config = run_build_adapter(model_path, adapter_path, peft_path)
            alpha = peft_config["lora_alpha"]
            rank = peft_config["r"]
            scaling = alpha / rank

            prefix = "base_model.model.model.language_model.layers.0.linear_attn"
            q_A = peft_weights[f"{prefix}.in_proj_q.lora_A.weight"].float()
            q_B = peft_weights[f"{prefix}.in_proj_q.lora_B.weight"].float()
            k_A = peft_weights[f"{prefix}.in_proj_k.lora_A.weight"].float()
            k_B = peft_weights[f"{prefix}.in_proj_k.lora_B.weight"].float()
            v_A = peft_weights[f"{prefix}.in_proj_v.lora_A.weight"].float()
            v_B = peft_weights[f"{prefix}.in_proj_v.lora_B.weight"].float()

            # Manual delta: concatenate [B_q@A_q, B_k@A_k, B_v@A_v] * scaling
            peft_delta = torch.cat([q_B @ q_A, k_B @ k_A, v_B @ v_A], dim=0) * scaling

            assert torch.allclose(merge_delta, peft_delta, atol=1e-3), (
                "Merge path and PEFT adapter + manual delta should produce identical results "
                f"for Qwen3.5 split QKV. Max diff: {(merge_delta - peft_delta).abs().max()}"
            )

    def test_tied_embeddings_remapped(self) -> None:
        """With tie_word_embeddings=True (Qwen3.5-4B), unembed_tokens should
        be remapped to embed_tokens in the PEFT output."""
        config = _make_tiny_qwen3_5_dense_config()
        tc = config.text_config
        assert tc.tie_word_embeddings is True

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "peft",
            )

            save_model_to_disk(config, model_path, tokenizer_name="Qwen/Qwen3.5-4B", is_vision=True)
            orig_tensors = load_file(str(model_path / "model.safetensors"))
            embed_shape = orig_tensors["model.language_model.embed_tokens.weight"].shape
            vocab_size, hidden = embed_shape

            rank = 1
            weights = {
                "base_model.model.model.unembed_tokens.lora_A.weight": (
                    torch.ones(rank, hidden) * FILL_A
                ),
                "base_model.model.model.unembed_tokens.lora_B.weight": torch.ones(vocab_size, rank),
            }
            adapter_path.mkdir(parents=True)
            save_file(weights, str(adapter_path / "adapter_model.safetensors"))
            (adapter_path / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 1, "r": rank})
            )

            peft_weights, _ = run_build_adapter(model_path, adapter_path, output_path)

            # Should be remapped to embed_tokens (not lm_head) due to tied embeddings.
            embed_keys = [k for k in peft_weights if "embed_tokens" in k]
            lm_head_keys = [k for k in peft_weights if "lm_head" in k]
            assert embed_keys, "unembed_tokens should map to embed_tokens for tied embeddings"
            assert not lm_head_keys, "Should not have lm_head keys for tied embeddings"


# ---------------------------------------------------------------------------
# Qwen3.5 MoE — expert expansion with broadcast shapes + vision prefix
# ---------------------------------------------------------------------------


def _make_tiny_qwen3_5_moe_config() -> PretrainedConfig:
    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B", trust_remote_code=True)
    tc = config.text_config
    tc.num_hidden_layers = 1
    tc.layer_types = ["linear_attention"]
    tc.linear_num_key_heads = 2
    tc.linear_num_value_heads = 4
    tc.linear_key_head_dim = 8
    tc.linear_value_head_dim = 8
    tc.hidden_size = 64
    tc.intermediate_size = 64
    tc.moe_intermediate_size = 48
    tc.num_attention_heads = 2
    tc.num_key_value_heads = 2
    tc.head_dim = 32
    tc.vocab_size = 256
    tc.mtp_num_hidden_layers = 0
    tc.num_experts = 2
    tc.num_experts_per_tok = 1
    tc.shared_expert_intermediate_size = 64
    config.vision_config.num_hidden_layers = 1
    config.vision_config.hidden_size = 64
    config.vision_config.intermediate_size = 64
    config.vision_config.num_attention_heads = 2
    return config


class TestQwen35MoeAdapterExport:
    """Qwen3.5 MoE: expert expansion with broadcast shapes + vision prefix.

    Verifies that build_lora_adapter handles:
    - 3D expert LoRA with broadcast (1, rank, dim) lora_A
    - Vision model language_model prefix
    - Per-expert 2D PEFT key generation
    """

    def test_expert_expansion_with_broadcast(self):
        config = _make_tiny_qwen3_5_moe_config()
        num_experts = config.text_config.num_experts

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
                tokenizer_name="Qwen/Qwen3.5-35B-A3B",
                is_vision=True,
            )

            # Read expert dims from saved model
            saved = load_file(str(model_path / "model.safetensors"))
            gate_key = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
            expert_out_dim, expert_in_dim = saved[gate_key].shape
            down_key = "model.language_model.layers.0.mlp.experts.0.down_proj.weight"
            down_out_dim, down_in_dim = saved[down_key].shape

            # Broadcast adapter: w1/w3 A shared, w2 B shared
            rank = 1
            prefix = "base_model.model.model.layers.0.mlp.experts"
            weights = {
                f"{prefix}.w1.lora_A.weight": torch.ones(1, rank, expert_in_dim) * FILL_A,
                f"{prefix}.w1.lora_B.weight": torch.ones(num_experts, expert_out_dim, rank),
                f"{prefix}.w3.lora_A.weight": torch.ones(1, rank, expert_in_dim) * FILL_B,
                f"{prefix}.w3.lora_B.weight": torch.ones(num_experts, expert_out_dim, rank),
                f"{prefix}.w2.lora_A.weight": torch.ones(num_experts, rank, down_in_dim) * FILL_A,
                f"{prefix}.w2.lora_B.weight": torch.ones(1, down_out_dim, rank),
            }
            adapter_path.mkdir(parents=True)
            save_file(weights, str(adapter_path / "adapter_model.safetensors"))
            (adapter_path / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 1, "r": rank})
            )

            peft_weights, peft_config = run_build_adapter(model_path, adapter_path, output_path)

            # Should have per-expert keys with vision prefix
            for e in range(num_experts):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    a_key = f"base_model.model.model.language_model.layers.0.mlp.experts.{e}.{proj}.lora_A.weight"
                    assert a_key in peft_weights, f"Missing {a_key}"
                    assert peft_weights[a_key].ndim == 2

            # Verify target_modules
            assert sorted(peft_config["target_modules"]) == [
                "down_proj",
                "gate_proj",
                "up_proj",
            ]
