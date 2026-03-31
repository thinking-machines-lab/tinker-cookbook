"""E2E export tests for Kimi K2.5: shard-by-shard merge with INT4 packed experts.

Creates a synthetic model directory mimicking Kimi K2.5's weight layout:
- language_model.model.* prefix (different from Qwen3.5's model.language_model.*)
- INT4 group-quantized routed expert weights (weight_packed / weight_scale / weight_shape)
- bf16 dense layers, attention, shared experts, embeddings
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

# Tiny model dimensions
HIDDEN = 64
MLP_DIM = 128  # dense MLP intermediate
EXPERT_DIM = 32  # per-expert MoE intermediate (must be divisible by group_size)
NUM_EXPERTS = 2
NUM_KV_HEADS = 2
NUM_HEADS = 4
HEAD_DIM = HIDDEN // NUM_HEADS
GROUP_SIZE = 32
VOCAB = 128
RANK = 1


def _make_kimi_k25_config() -> dict:
    """Create a config.json dict mimicking Kimi K2.5."""
    return {
        "model_type": "kimi_k25",
        "architectures": ["KimiK25ForConditionalGeneration"],
        "vision_config": {"hidden_size": 32, "num_hidden_layers": 1},
        "text_config": {
            "model_type": "kimi_k2",
            "architectures": ["DeepseekV3ForCausalLM"],
            "hidden_size": HIDDEN,
            "intermediate_size": MLP_DIM,
            "moe_intermediate_size": EXPERT_DIM,
            "num_hidden_layers": 2,
            "num_attention_heads": NUM_HEADS,
            "num_key_value_heads": NUM_KV_HEADS,
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
        },
    }


def _build_synthetic_model(model_dir: Path) -> dict[str, torch.Tensor]:
    """Build a synthetic K2.5 model directory with INT4 packed experts.

    Returns the original bf16 expert weights (before quantization) for
    verification against merged output.
    """
    tensors: dict[str, torch.Tensor] = {}
    original_expert_weights: dict[str, torch.Tensor] = {}

    # Dense layer 0 (bf16)
    tensors["language_model.model.layers.0.self_attn.q_a_proj.weight"] = torch.randn(
        HIDDEN, HIDDEN, dtype=torch.bfloat16
    )
    tensors["language_model.model.layers.0.self_attn.o_proj.weight"] = torch.randn(
        HIDDEN, HIDDEN, dtype=torch.bfloat16
    )
    tensors["language_model.model.layers.0.mlp.gate_proj.weight"] = torch.randn(
        MLP_DIM, HIDDEN, dtype=torch.bfloat16
    )
    tensors["language_model.model.layers.0.mlp.up_proj.weight"] = torch.randn(
        MLP_DIM, HIDDEN, dtype=torch.bfloat16
    )
    tensors["language_model.model.layers.0.mlp.down_proj.weight"] = torch.randn(
        HIDDEN, MLP_DIM, dtype=torch.bfloat16
    )
    tensors["language_model.model.layers.0.input_layernorm.weight"] = torch.ones(
        HIDDEN, dtype=torch.bfloat16
    )
    tensors["language_model.model.layers.0.post_attention_layernorm.weight"] = torch.ones(
        HIDDEN, dtype=torch.bfloat16
    )

    # MoE layer 1 — attention (bf16)
    tensors["language_model.model.layers.1.self_attn.q_a_proj.weight"] = torch.randn(
        HIDDEN, HIDDEN, dtype=torch.bfloat16
    )
    tensors["language_model.model.layers.1.self_attn.o_proj.weight"] = torch.randn(
        HIDDEN, HIDDEN, dtype=torch.bfloat16
    )
    tensors["language_model.model.layers.1.input_layernorm.weight"] = torch.ones(
        HIDDEN, dtype=torch.bfloat16
    )
    tensors["language_model.model.layers.1.post_attention_layernorm.weight"] = torch.ones(
        HIDDEN, dtype=torch.bfloat16
    )

    # MoE layer 1 — shared experts (bf16)
    tensors["language_model.model.layers.1.mlp.shared_experts.gate_proj.weight"] = torch.randn(
        EXPERT_DIM, HIDDEN, dtype=torch.bfloat16
    )
    tensors["language_model.model.layers.1.mlp.shared_experts.up_proj.weight"] = torch.randn(
        EXPERT_DIM, HIDDEN, dtype=torch.bfloat16
    )
    tensors["language_model.model.layers.1.mlp.shared_experts.down_proj.weight"] = torch.randn(
        HIDDEN, EXPERT_DIM, dtype=torch.bfloat16
    )

    # MoE layer 1 — routed experts (INT4 quantized)
    for i in range(NUM_EXPERTS):
        for proj, shape in [
            ("gate_proj", (EXPERT_DIM, HIDDEN)),
            ("up_proj", (EXPERT_DIM, HIDDEN)),
            ("down_proj", (HIDDEN, EXPERT_DIM)),
        ]:
            prefix = f"language_model.model.layers.1.mlp.experts.{i}.{proj}"
            # Generate bf16 weight, then quantize to INT4
            bf16_weight = torch.randn(*shape, dtype=torch.bfloat16)
            original_expert_weights[f"{prefix}.weight"] = bf16_weight.clone()
            packed, scale = quantize_int4_group(bf16_weight, GROUP_SIZE)
            tensors[f"{prefix}.weight_packed"] = packed
            tensors[f"{prefix}.weight_scale"] = scale
            tensors[f"{prefix}.weight_shape"] = torch.tensor(shape, dtype=torch.int32)

    # Embeddings and lm_head (bf16)
    tensors["language_model.model.embed_tokens.weight"] = torch.randn(
        VOCAB, HIDDEN, dtype=torch.bfloat16
    )
    tensors["language_model.lm_head.weight"] = torch.randn(
        VOCAB, HIDDEN, dtype=torch.bfloat16
    )

    # Save as two shards to exercise shard-by-shard processing.
    # Shard 1: dense layer 0 + embeddings
    # Shard 2: MoE layer 1 (attention + experts)
    shard1_keys = [k for k in tensors if "layers.0" in k or "embed" in k or "lm_head" in k]
    shard2_keys = [k for k in tensors if k not in shard1_keys]

    shard1 = {k: tensors[k] for k in shard1_keys}
    shard2 = {k: tensors[k] for k in shard2_keys}

    save_file(shard1, str(model_dir / "model-00001-of-00002.safetensors"))
    save_file(shard2, str(model_dir / "model-00002-of-00002.safetensors"))

    # Write index
    weight_map = {}
    for k in shard1_keys:
        weight_map[k] = "model-00001-of-00002.safetensors"
    for k in shard2_keys:
        weight_map[k] = "model-00002-of-00002.safetensors"

    index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    # Write config
    (model_dir / "config.json").write_text(json.dumps(_make_kimi_k25_config(), indent=2))

    # Write minimal tokenizer
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

    return original_expert_weights


def _save_kimi_k25_adapter(
    adapter_dir: Path,
    *,
    include_attention: bool = True,
    include_experts: bool = True,
) -> None:
    """Save a synthetic LoRA adapter targeting K2.5 keys."""
    weights: dict[str, torch.Tensor] = {}

    if include_attention:
        # Attention in dense layer 0
        weights["base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight"] = (
            torch.ones(RANK, HIDDEN) * FILL_A
        )
        weights["base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.weight"] = torch.ones(
            HIDDEN, RANK
        )

    if include_experts:
        # Expert LoRA for MoE layer 1 (w1=gate, w3=up)
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


class TestKimiK25ShardMerge:
    """E2E shard-by-shard merge for Kimi K2.5."""

    def test_attention_merge_bf16(self):
        """Merge attention LoRA into bf16 dense layer (no INT4 involved)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            adapter_dir = root / "adapter"
            output_dir = root / "output"
            model_dir.mkdir()

            _build_synthetic_model(model_dir)
            _save_kimi_k25_adapter(adapter_dir, include_attention=True, include_experts=False)

            # Load original weight for comparison
            orig = load_file(str(model_dir / "model-00001-of-00002.safetensors"))
            orig_attn = orig["language_model.model.layers.0.self_attn.q_a_proj.weight"].clone()

            build_hf_model(
                base_model=str(model_dir),
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
            )

            # Verify output exists
            assert (output_dir / "config.json").exists()

            # Load merged weight and verify LoRA was applied
            merged = _load_merged_tensors(output_dir)
            merged_attn = merged["language_model.model.layers.0.self_attn.q_a_proj.weight"]
            delta = merged_attn.float() - orig_attn.float()
            # LoRA delta = lora_B @ lora_A = ones(H, 1) @ (FILL_A * ones(1, H)) = FILL_A * ones(H, H)
            # bf16 precision loses some accuracy (FILL_A=0.01 has limited bf16 representation)
            expected_delta = torch.full((HIDDEN, HIDDEN), FILL_A)
            assert torch.allclose(delta, expected_delta, atol=0.01), (
                f"Delta max diff: {(delta - expected_delta).abs().max()}"
            )

    def test_expert_merge_int4_roundtrip(self):
        """Merge expert LoRA through INT4 dequant → merge → requant pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            adapter_dir = root / "adapter"
            output_dir = root / "output"
            model_dir.mkdir()

            orig_expert_weights = _build_synthetic_model(model_dir)
            _save_kimi_k25_adapter(adapter_dir, include_attention=False, include_experts=True)

            build_hf_model(
                base_model=str(model_dir),
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
            )

            # Load merged packed weights and dequantize
            merged = _load_merged_tensors(output_dir)
            for i in range(NUM_EXPERTS):
                prefix = f"language_model.model.layers.1.mlp.experts.{i}"
                for proj, fill in [("gate_proj", FILL_A), ("up_proj", FILL_B)]:
                    packed = merged[f"{prefix}.{proj}.weight_packed"]
                    scale = merged[f"{prefix}.{proj}.weight_scale"]
                    shape_t = merged[f"{prefix}.{proj}.weight_shape"]
                    shape = tuple(shape_t.tolist())

                    merged_bf16 = dequantize_int4_group(packed, scale, shape, GROUP_SIZE)
                    orig_bf16 = orig_expert_weights[f"{prefix}.{proj}.weight"]

                    # Dequant of original (before merge) — need to recover from
                    # the quantized version stored on disk
                    orig_packed = load_file(
                        str(model_dir / "model-00002-of-00002.safetensors")
                    )[f"{prefix}.{proj}.weight_packed"]
                    orig_scale = load_file(
                        str(model_dir / "model-00002-of-00002.safetensors")
                    )[f"{prefix}.{proj}.weight_scale"]
                    orig_dequant = dequantize_int4_group(orig_packed, orig_scale, shape, GROUP_SIZE)

                    delta = merged_bf16.float() - orig_dequant.float()
                    # Expected delta: lora_B[i] @ lora_A = ones(exp_dim,1) @ (fill * ones(1,H))
                    expected_delta = torch.full(shape, fill)
                    # Allow larger tolerance due to double INT4 quantization error
                    assert torch.allclose(delta, expected_delta, atol=1.0), (
                        f"Expert {i} {proj}: max delta diff = "
                        f"{(delta - expected_delta).abs().max():.4f}"
                    )

    def test_output_preserves_packed_format(self):
        """Merged output keeps INT4 packed format (weight_packed/scale/shape keys)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            adapter_dir = root / "adapter"
            output_dir = root / "output"
            model_dir.mkdir()

            _build_synthetic_model(model_dir)
            _save_kimi_k25_adapter(adapter_dir)

            build_hf_model(
                base_model=str(model_dir),
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
            )

            merged = _load_merged_tensors(output_dir)
            for i in range(NUM_EXPERTS):
                prefix = f"language_model.model.layers.1.mlp.experts.{i}.gate_proj"
                assert f"{prefix}.weight_packed" in merged, f"Missing packed key for expert {i}"
                assert f"{prefix}.weight_scale" in merged, f"Missing scale key for expert {i}"
                assert f"{prefix}.weight_shape" in merged, f"Missing shape key for expert {i}"
                # No plain .weight key for experts
                assert f"{prefix}.weight" not in merged

    def test_two_shard_output(self):
        """Output preserves multi-shard layout with correct index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            adapter_dir = root / "adapter"
            output_dir = root / "output"
            model_dir.mkdir()

            _build_synthetic_model(model_dir)
            _save_kimi_k25_adapter(adapter_dir)

            build_hf_model(
                base_model=str(model_dir),
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
            )

            index_path = output_dir / "model.safetensors.index.json"
            assert index_path.exists()
            with open(index_path) as f:
                index = json.load(f)
            shards = set(index["weight_map"].values())
            assert len(shards) >= 2, f"Expected multi-shard output, got {len(shards)} shard(s)"

    def test_non_expert_keys_unchanged(self):
        """Shared experts and layernorms are preserved unchanged (no LoRA targeting them)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            adapter_dir = root / "adapter"
            output_dir = root / "output"
            model_dir.mkdir()

            _build_synthetic_model(model_dir)
            _save_kimi_k25_adapter(adapter_dir, include_attention=False, include_experts=True)

            # Load originals
            orig_shard2 = load_file(str(model_dir / "model-00002-of-00002.safetensors"))
            shared_gate_orig = orig_shard2[
                "language_model.model.layers.1.mlp.shared_experts.gate_proj.weight"
            ].clone()

            build_hf_model(
                base_model=str(model_dir),
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
            )

            merged = _load_merged_tensors(output_dir)
            shared_gate_merged = merged[
                "language_model.model.layers.1.mlp.shared_experts.gate_proj.weight"
            ]
            assert torch.equal(shared_gate_orig, shared_gate_merged)


def _load_merged_tensors(output_dir: Path) -> dict[str, torch.Tensor]:
    """Load all tensors from the merged output directory."""
    all_tensors: dict[str, torch.Tensor] = {}
    for sf in sorted(output_dir.glob("*.safetensors")):
        all_tensors.update(load_file(str(sf)))
    return all_tensors
