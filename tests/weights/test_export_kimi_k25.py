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
from safetensors.torch import load_file

from tests.weights.conftest import FILL_A, FILL_B, load_merged_tensors
from tests.weights.test_export_kimi_common import (
    EXPERT_DIM,
    GROUP_SIZE,
    HIDDEN,
    NUM_EXPERTS,
    build_synthetic_kimi_model,
    save_kimi_adapter,
)
from tinker_cookbook.weights import build_hf_model
from tinker_cookbook.weights._packed_int4 import dequantize_int4_group


def _make_kimi_k25_config() -> dict:
    return {
        "model_type": "kimi_k25",
        "architectures": ["KimiK25ForConditionalGeneration"],
        "vision_config": {"hidden_size": 32, "num_hidden_layers": 1},
        "text_config": {
            "model_type": "kimi_k2",
            "architectures": ["DeepseekV3ForCausalLM"],
            "hidden_size": HIDDEN,
            "intermediate_size": 128,
            "moe_intermediate_size": EXPERT_DIM,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "n_routed_experts": NUM_EXPERTS,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "first_k_dense_replace": 1,
            "vocab_size": 128,
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
                "ignore": ["lm_head", "re:.*self_attn.*", "re:.*shared_experts.*"],
            },
        },
    }


# K2.5 nests language model under language_model.
_PREFIX = "language_model."


class TestKimiK25ShardMerge:
    """E2E shard-by-shard merge for Kimi K2.5."""

    def test_attention_merge_bf16(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir, adapter_dir, output_dir = root / "model", root / "adapter", root / "output"
            model_dir.mkdir()

            build_synthetic_kimi_model(model_dir, _make_kimi_k25_config(), _PREFIX)
            save_kimi_adapter(adapter_dir, _PREFIX, include_attention=True, include_experts=False)

            orig = load_file(str(model_dir / "model-00001-of-00002.safetensors"))
            attn_key = f"{_PREFIX}model.layers.0.self_attn.q_a_proj.weight"
            orig_attn = orig[attn_key].clone()

            build_hf_model(
                base_model=str(model_dir),
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
            )

            merged = load_merged_tensors(output_dir)
            delta = merged[attn_key].float() - orig_attn.float()
            expected = torch.full((HIDDEN, HIDDEN), FILL_A)
            assert torch.allclose(delta, expected, atol=0.01)

    def test_expert_merge_int4_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir, adapter_dir, output_dir = root / "model", root / "adapter", root / "output"
            model_dir.mkdir()

            build_synthetic_kimi_model(model_dir, _make_kimi_k25_config(), _PREFIX)
            save_kimi_adapter(adapter_dir, _PREFIX, include_attention=False, include_experts=True)

            build_hf_model(
                base_model=str(model_dir),
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
            )

            merged = load_merged_tensors(output_dir)
            orig_shard = load_file(str(model_dir / "model-00002-of-00002.safetensors"))

            for i in range(NUM_EXPERTS):
                prefix = f"{_PREFIX}model.layers.1.mlp.experts.{i}"
                for proj, fill in [("gate_proj", FILL_A), ("up_proj", FILL_B)]:
                    key = f"{prefix}.{proj}"
                    shape = tuple(merged[f"{key}.weight_shape"].tolist())
                    merged_bf16 = dequantize_int4_group(
                        merged[f"{key}.weight_packed"],
                        merged[f"{key}.weight_scale"],
                        shape,
                        GROUP_SIZE,
                    )
                    orig_dequant = dequantize_int4_group(
                        orig_shard[f"{key}.weight_packed"],
                        orig_shard[f"{key}.weight_scale"],
                        shape,
                        GROUP_SIZE,
                    )
                    delta = merged_bf16.float() - orig_dequant.float()
                    expected = torch.full(shape, fill)
                    assert torch.allclose(delta, expected, atol=1.0), (
                        f"Expert {i} {proj}: max diff = {(delta - expected).abs().max():.4f}"
                    )

    def test_output_preserves_packed_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir, adapter_dir, output_dir = root / "model", root / "adapter", root / "output"
            model_dir.mkdir()

            build_synthetic_kimi_model(model_dir, _make_kimi_k25_config(), _PREFIX)
            save_kimi_adapter(adapter_dir, _PREFIX)

            build_hf_model(
                base_model=str(model_dir),
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
            )

            merged = load_merged_tensors(output_dir)
            for i in range(NUM_EXPERTS):
                prefix = f"{_PREFIX}model.layers.1.mlp.experts.{i}.gate_proj"
                assert f"{prefix}.weight_packed" in merged
                assert f"{prefix}.weight_scale" in merged
                assert f"{prefix}.weight_shape" in merged
                assert f"{prefix}.weight" not in merged

    def test_two_shard_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir, adapter_dir, output_dir = root / "model", root / "adapter", root / "output"
            model_dir.mkdir()

            build_synthetic_kimi_model(model_dir, _make_kimi_k25_config(), _PREFIX)
            save_kimi_adapter(adapter_dir, _PREFIX)

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
            assert len(shards) >= 2

    def test_non_expert_keys_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir, adapter_dir, output_dir = root / "model", root / "adapter", root / "output"
            model_dir.mkdir()

            build_synthetic_kimi_model(model_dir, _make_kimi_k25_config(), _PREFIX)
            save_kimi_adapter(adapter_dir, _PREFIX, include_attention=False, include_experts=True)

            orig = load_file(str(model_dir / "model-00002-of-00002.safetensors"))
            key = f"{_PREFIX}model.layers.1.mlp.shared_experts.gate_proj.weight"
            shared_orig = orig[key].clone()

            build_hf_model(
                base_model=str(model_dir),
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
            )

            merged = load_merged_tensors(output_dir)
            assert torch.equal(shared_orig, merged[key])
