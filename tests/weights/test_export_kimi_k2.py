"""E2E export tests for Kimi K2: shard-by-shard merge with INT4 packed experts.

Creates a synthetic model directory mimicking Kimi K2's weight layout:
- Standard model.layers.* prefix (no VL nesting, unlike K2.5)
- INT4 group-quantized routed expert weights (weight_packed / weight_scale / weight_shape)
- bf16 dense layers, attention, shared experts, embeddings
- model_type=kimi_k2 (falls through to default merge profile, NOT blocked as deepseek_v3)
"""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from tests.weights.conftest import FILL_A, FILL_B
from tinker_cookbook.weights import build_hf_model
from tinker_cookbook.weights._packed_int4 import (
    dequantize_int4_group,
    quantize_int4_group,
)

# Tiny model dimensions (matching K2's DeepSeek architecture)
HIDDEN = 64
MLP_DIM = 128  # dense MLP intermediate
EXPERT_DIM = 32  # per-expert MoE intermediate (must be divisible by group_size)
NUM_EXPERTS = 2
GROUP_SIZE = 32
VOCAB = 128
RANK = 1


def _make_kimi_k2_config() -> dict:
    """Create a config.json dict mimicking Kimi K2."""
    return {
        "model_type": "kimi_k2",
        "architectures": ["DeepseekV3ForCausalLM"],
        "hidden_size": HIDDEN,
        "intermediate_size": MLP_DIM,
        "moe_intermediate_size": EXPERT_DIM,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "n_routed_experts": NUM_EXPERTS,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "first_k_dense_replace": 1,
        "vocab_size": VOCAB,
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "quantization_status": "compressed",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": GROUP_SIZE,
                    },
                },
            },
            "ignore": [
                "lm_head",
                "re:.*self_attn.*",
                "re:.*shared_experts.*",
                "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
            ],
        },
    }


def _build_synthetic_model(model_dir: Path) -> None:
    """Build a synthetic K2 model directory with INT4 packed experts."""
    tensors: dict[str, torch.Tensor] = {}

    # Dense layer 0 (bf16)
    tensors["model.layers.0.self_attn.q_a_proj.weight"] = torch.randn(
        HIDDEN, HIDDEN, dtype=torch.bfloat16
    )
    tensors["model.layers.0.self_attn.o_proj.weight"] = torch.randn(
        HIDDEN, HIDDEN, dtype=torch.bfloat16
    )
    tensors["model.layers.0.mlp.gate_proj.weight"] = torch.randn(
        MLP_DIM, HIDDEN, dtype=torch.bfloat16
    )
    tensors["model.layers.0.mlp.up_proj.weight"] = torch.randn(
        MLP_DIM, HIDDEN, dtype=torch.bfloat16
    )
    tensors["model.layers.0.mlp.down_proj.weight"] = torch.randn(
        HIDDEN, MLP_DIM, dtype=torch.bfloat16
    )
    tensors["model.layers.0.input_layernorm.weight"] = torch.ones(HIDDEN, dtype=torch.bfloat16)
    tensors["model.layers.0.post_attention_layernorm.weight"] = torch.ones(
        HIDDEN, dtype=torch.bfloat16
    )

    # MoE layer 1 — attention (bf16)
    tensors["model.layers.1.self_attn.q_a_proj.weight"] = torch.randn(
        HIDDEN, HIDDEN, dtype=torch.bfloat16
    )
    tensors["model.layers.1.self_attn.o_proj.weight"] = torch.randn(
        HIDDEN, HIDDEN, dtype=torch.bfloat16
    )
    tensors["model.layers.1.input_layernorm.weight"] = torch.ones(HIDDEN, dtype=torch.bfloat16)
    tensors["model.layers.1.post_attention_layernorm.weight"] = torch.ones(
        HIDDEN, dtype=torch.bfloat16
    )

    # MoE layer 1 — shared experts (bf16)
    tensors["model.layers.1.mlp.shared_experts.gate_proj.weight"] = torch.randn(
        EXPERT_DIM, HIDDEN, dtype=torch.bfloat16
    )
    tensors["model.layers.1.mlp.shared_experts.up_proj.weight"] = torch.randn(
        EXPERT_DIM, HIDDEN, dtype=torch.bfloat16
    )
    tensors["model.layers.1.mlp.shared_experts.down_proj.weight"] = torch.randn(
        HIDDEN, EXPERT_DIM, dtype=torch.bfloat16
    )

    # MoE layer 1 — routed experts (INT4 quantized)
    for i in range(NUM_EXPERTS):
        for proj, shape in [
            ("gate_proj", (EXPERT_DIM, HIDDEN)),
            ("up_proj", (EXPERT_DIM, HIDDEN)),
            ("down_proj", (HIDDEN, EXPERT_DIM)),
        ]:
            prefix = f"model.layers.1.mlp.experts.{i}.{proj}"
            bf16_weight = torch.randn(*shape, dtype=torch.bfloat16)
            packed, scale = quantize_int4_group(bf16_weight, GROUP_SIZE)
            tensors[f"{prefix}.weight_packed"] = packed
            tensors[f"{prefix}.weight_scale"] = scale
            tensors[f"{prefix}.weight_shape"] = torch.tensor(shape, dtype=torch.int32)

    # Embeddings and lm_head (bf16)
    tensors["model.embed_tokens.weight"] = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)
    tensors["lm_head.weight"] = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)

    # Save as two shards
    shard1_keys = [k for k in tensors if "layers.0" in k or "embed" in k or "lm_head" in k]
    shard2_keys = [k for k in tensors if k not in shard1_keys]

    save_file(
        {k: tensors[k] for k in shard1_keys},
        str(model_dir / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {k: tensors[k] for k in shard2_keys},
        str(model_dir / "model-00002-of-00002.safetensors"),
    )

    weight_map = {}
    for k in shard1_keys:
        weight_map[k] = "model-00001-of-00002.safetensors"
    for k in shard2_keys:
        weight_map[k] = "model-00002-of-00002.safetensors"

    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}, indent=2)
    )
    (model_dir / "config.json").write_text(json.dumps(_make_kimi_k2_config(), indent=2))
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"})
    )
    (model_dir / "tokenizer.json").write_text(
        json.dumps({
            "version": "1.0",
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
            "added_tokens": [],
        })
    )


def _save_kimi_k2_adapter(
    adapter_dir: Path,
    *,
    include_attention: bool = True,
    include_experts: bool = True,
) -> None:
    """Save a synthetic LoRA adapter targeting K2 keys (standard prefix)."""
    weights: dict[str, torch.Tensor] = {}

    if include_attention:
        weights["base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight"] = (
            torch.ones(RANK, HIDDEN) * FILL_A
        )
        weights["base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.weight"] = torch.ones(
            HIDDEN, RANK
        )

    if include_experts:
        prefix = "base_model.model.model.layers.1.mlp.experts"
        weights[f"{prefix}.w1.lora_A.weight"] = torch.ones(1, RANK, HIDDEN) * FILL_A
        weights[f"{prefix}.w1.lora_B.weight"] = torch.ones(NUM_EXPERTS, EXPERT_DIM, RANK)
        weights[f"{prefix}.w3.lora_A.weight"] = torch.ones(1, RANK, HIDDEN) * FILL_B
        weights[f"{prefix}.w3.lora_B.weight"] = torch.ones(NUM_EXPERTS, EXPERT_DIM, RANK)

    adapter_dir.mkdir(parents=True)
    save_file(weights, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"lora_alpha": 1, "r": RANK})
    )


def _load_merged_tensors(output_dir: Path) -> dict[str, torch.Tensor]:
    all_tensors: dict[str, torch.Tensor] = {}
    for sf in sorted(output_dir.glob("*.safetensors")):
        all_tensors.update(load_file(str(sf)))
    return all_tensors


class TestKimiK2ShardMerge:
    """E2E shard-by-shard merge for Kimi K2."""

    def test_uses_default_profile_not_deepseek(self):
        """K2 (model_type=kimi_k2) must use default profile, not deepseek."""
        from tinker_cookbook.weights._merge import detect_merge_profile

        config = _make_kimi_k2_config()
        # Use standard key names (no packed) for profile detection test
        keys = {"model.layers.0.self_attn.q_a_proj.weight"}
        profile = detect_merge_profile(config, keys)
        # Should be default, NOT deepseek (which would block adapter conversion)
        assert profile.model_family == "default"
        assert profile.expert_layout == "separate"

    def test_attention_merge_bf16(self):
        """Merge attention LoRA into bf16 dense layer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir, adapter_dir, output_dir = root / "model", root / "adapter", root / "output"
            model_dir.mkdir()

            _build_synthetic_model(model_dir)
            _save_kimi_k2_adapter(adapter_dir, include_attention=True, include_experts=False)

            orig = load_file(str(model_dir / "model-00001-of-00002.safetensors"))
            orig_attn = orig["model.layers.0.self_attn.q_a_proj.weight"].clone()

            build_hf_model(
                base_model=str(model_dir),
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
            )

            merged = _load_merged_tensors(output_dir)
            delta = merged["model.layers.0.self_attn.q_a_proj.weight"].float() - orig_attn.float()
            expected = torch.full((HIDDEN, HIDDEN), FILL_A)
            assert torch.allclose(delta, expected, atol=0.01)

    def test_expert_merge_int4_roundtrip(self):
        """Merge expert LoRA through INT4 dequant → merge → requant."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir, adapter_dir, output_dir = root / "model", root / "adapter", root / "output"
            model_dir.mkdir()

            _build_synthetic_model(model_dir)
            _save_kimi_k2_adapter(adapter_dir, include_attention=False, include_experts=True)

            build_hf_model(
                base_model=str(model_dir),
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
            )

            merged = _load_merged_tensors(output_dir)
            for i in range(NUM_EXPERTS):
                prefix = f"model.layers.1.mlp.experts.{i}"
                for proj, fill in [("gate_proj", FILL_A), ("up_proj", FILL_B)]:
                    packed = merged[f"{prefix}.{proj}.weight_packed"]
                    scale = merged[f"{prefix}.{proj}.weight_scale"]
                    shape_t = merged[f"{prefix}.{proj}.weight_shape"]
                    shape = tuple(shape_t.tolist())

                    merged_bf16 = dequantize_int4_group(packed, scale, shape, GROUP_SIZE)

                    orig_packed = load_file(
                        str(model_dir / "model-00002-of-00002.safetensors")
                    )[f"{prefix}.{proj}.weight_packed"]
                    orig_scale = load_file(
                        str(model_dir / "model-00002-of-00002.safetensors")
                    )[f"{prefix}.{proj}.weight_scale"]
                    orig_dequant = dequantize_int4_group(orig_packed, orig_scale, shape, GROUP_SIZE)

                    delta = merged_bf16.float() - orig_dequant.float()
                    expected = torch.full(shape, fill)
                    assert torch.allclose(delta, expected, atol=1.0), (
                        f"Expert {i} {proj}: max diff = {(delta - expected).abs().max():.4f}"
                    )

    def test_output_preserves_packed_format(self):
        """Merged output keeps INT4 packed format for expert weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir, adapter_dir, output_dir = root / "model", root / "adapter", root / "output"
            model_dir.mkdir()

            _build_synthetic_model(model_dir)
            _save_kimi_k2_adapter(adapter_dir)

            build_hf_model(
                base_model=str(model_dir),
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
            )

            merged = _load_merged_tensors(output_dir)
            for i in range(NUM_EXPERTS):
                prefix = f"model.layers.1.mlp.experts.{i}.gate_proj"
                assert f"{prefix}.weight_packed" in merged
                assert f"{prefix}.weight_scale" in merged
                assert f"{prefix}.weight_shape" in merged
                assert f"{prefix}.weight" not in merged

    def test_shared_experts_unchanged(self):
        """Shared experts (bf16, no LoRA) pass through unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir, adapter_dir, output_dir = root / "model", root / "adapter", root / "output"
            model_dir.mkdir()

            _build_synthetic_model(model_dir)
            _save_kimi_k2_adapter(adapter_dir, include_attention=False, include_experts=True)

            orig = load_file(str(model_dir / "model-00002-of-00002.safetensors"))
            shared_orig = orig["model.layers.1.mlp.shared_experts.gate_proj.weight"].clone()

            build_hf_model(
                base_model=str(model_dir),
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
            )

            merged = _load_merged_tensors(output_dir)
            assert torch.equal(
                shared_orig, merged["model.layers.1.mlp.shared_experts.gate_proj.weight"]
            )
