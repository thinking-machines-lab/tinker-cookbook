"""Unit tests for quantized export strategy.

Covers FP8 math, DeepSeek detection, weight classification, vLLM config
generation, resume state, and output shard assembly. Uses synthetic data —
no network or GPU required.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tinker_cookbook.weights._export._quantized import (
    _build_vllm_quantization_config,
    _is_routed_expert_weight,
    _load_resume_state,
    _save_merge_state,
    _save_shard_atomic,
    _serialize_for_vllm,
    _should_skip_checkpoint_key,
    dequantize_blockwise,
    is_deepseek_config,
    quantize_blockwise,
)

# ---------------------------------------------------------------------------
# FP8 quantize / dequantize round-trip
# ---------------------------------------------------------------------------


class TestFP8RoundTrip:
    def test_exact_round_trip_block_size_1(self):
        """Block size (1,1) should give exact round-trip for representable values."""
        tensor = torch.tensor([[1.0, -0.5], [0.25, 0.0]])
        fp8, scale = quantize_blockwise(tensor, block_size=(1, 1))
        assert fp8.dtype == torch.float8_e4m3fn
        assert scale.dtype == torch.float32
        recovered = dequantize_blockwise(fp8, scale, block_size=(1, 1))
        assert torch.allclose(recovered.float(), tensor, atol=1e-2)

    def test_padded_dimensions(self):
        """Tensor dimensions not divisible by block size should still round-trip."""
        tensor = torch.randn(5, 7)
        fp8, scale = quantize_blockwise(tensor, block_size=(2, 3))
        assert fp8.shape == (5, 7)
        # Scale shape should be ceil(5/2) x ceil(7/3) = 3 x 3
        assert scale.shape == (3, 3)
        recovered = dequantize_blockwise(fp8, scale, block_size=(2, 3), dtype=torch.float32)
        assert recovered.shape == (5, 7)
        # Round-trip error should be small
        assert torch.allclose(recovered, tensor, atol=0.2)

    def test_large_block_preserves_shape(self):
        """Standard 128x128 block size with a realistic shape."""
        tensor = torch.randn(256, 384)
        fp8, scale = quantize_blockwise(tensor)
        assert fp8.shape == (256, 384)
        assert scale.shape == (2, 3)  # ceil(256/128) x ceil(384/128)

    def test_zeros_round_trip(self):
        """All-zero tensor should round-trip cleanly."""
        tensor = torch.zeros(4, 4)
        fp8, scale = quantize_blockwise(tensor, block_size=(2, 2))
        recovered = dequantize_blockwise(fp8, scale, block_size=(2, 2), dtype=torch.float32)
        assert torch.allclose(recovered, tensor)


# ---------------------------------------------------------------------------
# DeepSeek detection
# ---------------------------------------------------------------------------


class TestIsDeepseekConfig:
    def test_deepseek_v3_detected(self):
        assert is_deepseek_config({"model_type": "deepseek_v3"})

    def test_non_deepseek_rejected(self):
        assert not is_deepseek_config({"model_type": "qwen2_moe"})
        assert not is_deepseek_config({"model_type": "llama"})
        assert not is_deepseek_config({})

    def test_similar_strings_rejected(self):
        assert not is_deepseek_config({"model_type": "deepseek"})
        assert not is_deepseek_config({"model_type": "deepseek_v2"})


# ---------------------------------------------------------------------------
# Weight classification
# ---------------------------------------------------------------------------


class TestIsRoutedExpertWeight:
    def test_routed_expert_matched(self):
        assert _is_routed_expert_weight("model.layers.3.mlp.experts.42.gate_proj.weight")
        assert _is_routed_expert_weight("model.layers.0.mlp.experts.0.down_proj.weight")

    def test_shared_expert_rejected(self):
        assert not _is_routed_expert_weight("model.layers.0.mlp.shared_experts.gate_proj.weight")

    def test_attention_rejected(self):
        assert not _is_routed_expert_weight("model.layers.0.self_attn.q_proj.weight")

    def test_norm_rejected(self):
        assert not _is_routed_expert_weight("model.layers.0.input_layernorm.weight")

    def test_embed_rejected(self):
        assert not _is_routed_expert_weight("model.embed_tokens.weight")


# ---------------------------------------------------------------------------
# Skip key logic
# ---------------------------------------------------------------------------


class TestShouldSkipCheckpointKey:
    def test_rotary_emb_skipped(self):
        assert _should_skip_checkpoint_key("model.layers.0.self_attn.rotary_emb.inv_freq")

    def test_layer_61_skipped(self):
        assert _should_skip_checkpoint_key("model.layers.61.self_attn.q_proj.weight")
        assert _should_skip_checkpoint_key("model.layers.61.mlp.experts.0.gate_proj.weight")

    def test_normal_layer_not_skipped(self):
        assert not _should_skip_checkpoint_key("model.layers.0.self_attn.q_proj.weight")
        assert not _should_skip_checkpoint_key("model.layers.60.mlp.gate_proj.weight")

    def test_embed_not_skipped(self):
        assert not _should_skip_checkpoint_key("model.embed_tokens.weight")


# ---------------------------------------------------------------------------
# vLLM quantization config
# ---------------------------------------------------------------------------


class TestBuildVllmQuantizationConfig:
    def test_correct_schema(self):
        weight_map = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": "shard-1.safetensors",
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale": "shard-1.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "shard-1.safetensors",
            "model.embed_tokens.weight": "shard-1.safetensors",
        }
        config = _build_vllm_quantization_config(weight_map)
        assert config["quant_method"] == "compressed-tensors"
        assert config["format"] == "float-quantized"
        assert config["quantization_status"] == "compressed"
        assert "config_groups" in config
        assert "ignore" in config
        # Verify block quantization config
        weights = config["config_groups"]["group_0"]["weights"]
        assert weights["strategy"] == "block"
        assert weights["block_structure"] == [128, 128]
        # Verify input activations
        ia = config["config_groups"]["group_0"]["input_activations"]
        assert ia["dynamic"] is True

    def test_ignore_list_correct(self):
        """Dense projections should be in ignore, routed experts should NOT."""
        weight_map = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": "s.safetensors",
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale": "s.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "s.safetensors",
            "model.layers.0.mlp.shared_experts.gate_proj.weight": "s.safetensors",
        }
        config = _build_vllm_quantization_config(weight_map)
        ignore = config["ignore"]
        # Dense/shared should be in ignore
        assert "model.layers.0.self_attn.q_proj" in ignore
        assert "model.layers.0.mlp.shared_experts.gate_proj" in ignore
        # Routed expert should NOT be in ignore
        assert "model.layers.0.mlp.experts.0.gate_proj" not in ignore


class TestSerializeForVllm:
    def test_strips_unknown_fields(self):
        config = {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "unknown_field": "should be stripped",
            "another_unknown": 42,
            "ignore": [],
            "config_groups": {},
        }
        result = _serialize_for_vllm(config)
        assert "unknown_field" not in result
        assert "another_unknown" not in result
        assert result["quant_method"] == "compressed-tensors"
        assert result["ignore"] == []

    def test_preserves_known_fields(self):
        config = {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "quantization_status": "compressed",
            "global_compression_ratio": None,
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 8, "strategy": "block"},
                }
            },
            "ignore": ["a.b"],
        }
        result = _serialize_for_vllm(config)
        assert result["quant_method"] == "compressed-tensors"
        assert result["ignore"] == ["a.b"]
        assert result["config_groups"]["group_0"]["targets"] == ["Linear"]


# ---------------------------------------------------------------------------
# Resume state
# ---------------------------------------------------------------------------


class TestResumeState:
    def test_no_state_file_returns_empty(self, tmp_path: Path):
        assert _load_resume_state(tmp_path) == {}

    def test_load_valid_state(self, tmp_path: Path):
        save_file({"x": torch.zeros(1)}, str(tmp_path / "shard-1.safetensors"))
        _save_merge_state(
            tmp_path,
            status="in_progress",
            completed_shards=["shard-1.safetensors"],
            total_shards=2,
        )
        state = _load_resume_state(tmp_path)
        assert state["status"] == "in_progress"
        assert state["completed_shards"] == ["shard-1.safetensors"]
        assert state["total_shards"] == 2

    def test_missing_shard_file_raises(self, tmp_path: Path):
        """Resume state references a shard that doesn't exist on disk."""
        _save_merge_state(
            tmp_path,
            status="in_progress",
            completed_shards=["missing.safetensors"],
            total_shards=1,
        )
        with pytest.raises(RuntimeError, match="not found"):
            _load_resume_state(tmp_path)

    def test_atomic_save(self, tmp_path: Path):
        """Merge state should be saved atomically (no partial writes)."""
        _save_merge_state(tmp_path, status="in_progress", completed_shards=[], total_shards=3)
        state_file = tmp_path / "merge_state.json"
        assert state_file.exists()
        # Temp file should not exist
        assert not (tmp_path / "merge_state.json.tmp").exists()


class TestSaveShardAtomic:
    def test_atomic_write(self, tmp_path: Path):
        tensors = {"a": torch.ones(2, 3)}
        _save_shard_atomic(tmp_path, "shard-1.safetensors", tensors)
        assert (tmp_path / "shard-1.safetensors").exists()
        assert not (tmp_path / "shard-1.safetensors.tmp").exists()
        loaded = load_file(str(tmp_path / "shard-1.safetensors"))
        assert torch.equal(loaded["a"], torch.ones(2, 3))


# ---------------------------------------------------------------------------
# Output shard assembly (quantize behavior)
# ---------------------------------------------------------------------------


class TestOutputShardAssembly:
    """Test that routed experts get FP8+scale and dense stays BF16."""

    def _make_model_and_adapter(self, tmp_path: Path):
        """Create a minimal DeepSeek-like model with one shard."""
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        num_experts = 2

        # Model state: routed experts + one dense weight + shared expert
        state_dict = {}
        for i in range(num_experts):
            state_dict[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.randn(
                8, 4, dtype=torch.bfloat16
            )
        state_dict["model.layers.0.self_attn.q_proj.weight"] = torch.randn(
            8, 4, dtype=torch.bfloat16
        )
        state_dict["model.layers.0.mlp.shared_experts.gate_proj.weight"] = torch.randn(
            8, 4, dtype=torch.bfloat16
        )
        model_dir.mkdir(parents=True)
        save_file(state_dict, str(model_dir / "model.safetensors"))
        config = {"model_type": "deepseek_v3", "architectures": ["DeepseekV3ForCausalLM"]}
        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"})
        )
        (model_dir / "tokenizer.json").write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
                    "added_tokens": [],
                }
            )
        )

        # Adapter targeting the experts and dense weight
        adapter_weights = {
            "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight": torch.ones(
                num_experts, 1, 4
            )
            * 0.01,
            "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight": torch.ones(
                num_experts, 8, 1
            ),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1, 4)
            * 0.01,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(8, 1),
        }
        adapter_dir.mkdir(parents=True)
        save_file(adapter_weights, str(adapter_dir / "adapter_model.safetensors"))
        (adapter_dir / "adapter_config.json").write_text(json.dumps({"lora_alpha": 1, "r": 1}))

        return model_dir, adapter_dir

    def test_routed_expert_quantized_to_fp8(self, tmp_path: Path):
        from tinker_cookbook.weights._export._quantized import build_quantized

        model_dir, adapter_dir = self._make_model_and_adapter(tmp_path)
        output_dir = tmp_path / "output"

        build_quantized(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            trust_remote_code=False,
            model_dir=model_dir,
            config_dict=json.loads((model_dir / "config.json").read_text()),
            serving_format="vllm",
        )

        out_tensors = load_file(str(output_dir / "model.safetensors"))

        # Routed expert should be FP8
        expert_w = out_tensors["model.layers.0.mlp.experts.0.gate_proj.weight"]
        assert expert_w.dtype == torch.float8_e4m3fn

        # Should have a scale tensor
        assert "model.layers.0.mlp.experts.0.gate_proj.weight_scale" in out_tensors
        scale = out_tensors["model.layers.0.mlp.experts.0.gate_proj.weight_scale"]
        assert scale.dtype == torch.float32

    def test_dense_stays_bf16(self, tmp_path: Path):
        from tinker_cookbook.weights._export._quantized import build_quantized

        model_dir, adapter_dir = self._make_model_and_adapter(tmp_path)
        output_dir = tmp_path / "output"

        build_quantized(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            trust_remote_code=False,
            model_dir=model_dir,
            config_dict=json.loads((model_dir / "config.json").read_text()),
            serving_format="vllm",
        )

        out_tensors = load_file(str(output_dir / "model.safetensors"))

        # Dense weight should stay BF16
        q_proj = out_tensors["model.layers.0.self_attn.q_proj.weight"]
        assert q_proj.dtype == torch.bfloat16

        # No scale tensor for dense weights
        assert "model.layers.0.self_attn.q_proj.weight_scale" not in out_tensors

    def test_shared_expert_stays_bf16(self, tmp_path: Path):
        from tinker_cookbook.weights._export._quantized import build_quantized

        model_dir, adapter_dir = self._make_model_and_adapter(tmp_path)
        output_dir = tmp_path / "output"

        build_quantized(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            trust_remote_code=False,
            model_dir=model_dir,
            config_dict=json.loads((model_dir / "config.json").read_text()),
            serving_format="vllm",
        )

        out_tensors = load_file(str(output_dir / "model.safetensors"))

        shared = out_tensors["model.layers.0.mlp.shared_experts.gate_proj.weight"]
        assert shared.dtype == torch.bfloat16
        assert "model.layers.0.mlp.shared_experts.gate_proj.weight_scale" not in out_tensors

    def test_config_has_compression_config(self, tmp_path: Path):
        from tinker_cookbook.weights._export._quantized import build_quantized

        model_dir, adapter_dir = self._make_model_and_adapter(tmp_path)
        output_dir = tmp_path / "output"

        build_quantized(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            trust_remote_code=False,
            model_dir=model_dir,
            config_dict=json.loads((model_dir / "config.json").read_text()),
            serving_format="vllm",
        )

        config = json.loads((output_dir / "config.json").read_text())
        assert "compression_config" in config
        cc = config["compression_config"]
        assert cc["quant_method"] == "compressed-tensors"
        assert "quantization_config" not in config


# ---------------------------------------------------------------------------
# Cross-shard native FP8 scale handling
# ---------------------------------------------------------------------------


class TestCrossShardFP8Scale:
    """Test that native FP8 weights are dequantized correctly even when
    the weight and its scale_inv are in different shards."""

    def test_cross_shard_scale_dequantizes_correctly(self, tmp_path: Path):
        from tinker_cookbook.weights._export._quantized import build_quantized, quantize_blockwise

        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"
        model_dir.mkdir(parents=True)

        # Create a native FP8 expert weight and its scale, in DIFFERENT shards
        original_weight = torch.randn(8, 4, dtype=torch.bfloat16)
        fp8_weight, scale_inv = quantize_blockwise(original_weight, block_size=(4, 4))

        # Shard 1: has the FP8 weight but NOT its scale
        shard1 = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": fp8_weight,
            "model.layers.0.mlp.experts.1.gate_proj.weight": fp8_weight.clone(),
        }
        # Shard 2: has the scale_inv but NOT the weight, plus another expert
        shard2 = {
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": scale_inv,
            "model.layers.0.mlp.experts.1.gate_proj.weight_scale_inv": scale_inv.clone(),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
        }

        save_file(shard1, str(model_dir / "model-00001-of-00002.safetensors"))
        save_file(shard2, str(model_dir / "model-00002-of-00002.safetensors"))

        weight_map = {}
        for k in shard1:
            weight_map[k] = "model-00001-of-00002.safetensors"
        for k in shard2:
            weight_map[k] = "model-00002-of-00002.safetensors"

        total_size = sum(t.nelement() * t.element_size() for t in {**shard1, **shard2}.values())
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))

        # Config with native FP8 quantization
        config = {
            "model_type": "deepseek_v3",
            "architectures": ["DeepseekV3ForCausalLM"],
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [4, 4],
            },
        }
        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"})
        )
        (model_dir / "tokenizer.json").write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
                    "added_tokens": [],
                }
            )
        )

        # Empty adapter (no merge, just quantize)
        adapter_dir.mkdir(parents=True)
        save_file({"dummy": torch.zeros(1)}, str(adapter_dir / "adapter_model.safetensors"))
        (adapter_dir / "adapter_config.json").write_text(json.dumps({"lora_alpha": 1, "r": 1}))

        build_quantized(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            trust_remote_code=False,
            model_dir=model_dir,
            config_dict=config,
            serving_format="vllm",
        )

        # Verify: expert weights should be FP8 (re-quantized after dequant)
        out = {}
        for p in sorted(output_dir.glob("*.safetensors")):
            out.update(load_file(str(p)))

        expert_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
        assert expert_key in out
        assert out[expert_key].dtype == torch.float8_e4m3fn
        # Scale should use compressed-tensors naming
        assert "model.layers.0.mlp.experts.0.gate_proj.weight_scale" in out
        # No native scale_inv in output
        assert "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv" not in out

    def test_merge_applied_before_requantize_on_native_fp8(self, tmp_path: Path):
        """Regression: LoRA merge must happen AFTER dequant, BEFORE requant.

        If merge is applied to the raw FP8 tensor (before dequant), the delta
        gets corrupted because FP8 can't represent the fine-grained LoRA values.
        """
        from tinker_cookbook.weights._export._quantized import (
            build_quantized,
            dequantize_blockwise,
            quantize_blockwise,
        )

        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"
        model_dir.mkdir(parents=True)

        # Create a native FP8 expert weight
        original_bf16 = torch.randn(8, 4, dtype=torch.bfloat16)
        fp8_weight, scale_inv = quantize_blockwise(original_bf16, block_size=(4, 4))

        num_experts = 2
        shard1: dict[str, torch.Tensor] = {}
        for i in range(num_experts):
            shard1[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = fp8_weight.clone()
            shard1[f"model.layers.0.mlp.experts.{i}.gate_proj.weight_scale_inv"] = scale_inv.clone()

        save_file(shard1, str(model_dir / "model.safetensors"))
        config = {
            "model_type": "deepseek_v3",
            "architectures": ["DeepseekV3ForCausalLM"],
            "quantization_config": {"quant_method": "fp8", "weight_block_size": [4, 4]},
        }
        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"})
        )
        (model_dir / "tokenizer.json").write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
                    "added_tokens": [],
                }
            )
        )

        # LoRA adapter targeting gate_proj (w1) with a known delta
        lora_fill = 0.1
        adapter_weights = {
            "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight": (
                torch.ones(num_experts, 1, 4) * lora_fill
            ),
            "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight": torch.ones(
                num_experts, 8, 1
            ),
        }
        adapter_dir.mkdir(parents=True)
        save_file(adapter_weights, str(adapter_dir / "adapter_model.safetensors"))
        (adapter_dir / "adapter_config.json").write_text(json.dumps({"lora_alpha": 1, "r": 1}))

        build_quantized(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            trust_remote_code=False,
            model_dir=model_dir,
            config_dict=config,
            serving_format="vllm",
        )

        # Load output and dequantize to check the merge was applied
        out = {}
        for p in sorted(output_dir.glob("*.safetensors")):
            out.update(load_file(str(p)))

        expert_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
        scale_key = "model.layers.0.mlp.experts.0.gate_proj.weight_scale"
        merged_dequantized = dequantize_blockwise(
            out[expert_key], out[scale_key], block_size=(128, 128)
        )

        # The original dequantized value
        original_dequantized = dequantize_blockwise(fp8_weight, scale_inv, block_size=(4, 4))

        # The merged result should differ from original by approximately the LoRA delta
        delta = (merged_dequantized.float() - original_dequantized.float()).abs()
        assert delta.sum() > 0, "LoRA merge had no effect — merge may have been applied to FP8"
        # The delta should be approximately lora_fill (0.1) everywhere
        # Allow tolerance for FP8 round-trip quantization error
        assert delta.mean().item() == pytest.approx(lora_fill, abs=0.05), (
            f"Expected delta ~{lora_fill}, got {delta.mean().item():.4f}. "
            "Merge may have been applied before dequantization."
        )
