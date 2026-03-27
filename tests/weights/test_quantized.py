"""End-to-end tests for quantized export (DeepSeek FP8).

Uses a tiny 1-layer DeepSeek V3 model created from config with synthetic
random weights. Tests exercise the full pipeline including merge, quantize,
shard layout preservation, config patching, and resume.
"""

import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file

from tinker_cookbook.weights import build_hf_model

# ---------------------------------------------------------------------------
# Tiny DeepSeek model fixture
# ---------------------------------------------------------------------------

_HIDDEN = 64
_INTER = 128
_NUM_EXPERTS = 2
_VOCAB = 256


def _deepseek_config() -> dict:
    """Minimal DeepSeek V3 config."""
    return {
        "model_type": "deepseek_v3",
        "architectures": ["DeepseekV3ForCausalLM"],
        "hidden_size": _HIDDEN,
        "intermediate_size": _INTER,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "n_routed_experts": _NUM_EXPERTS,
        "vocab_size": _VOCAB,
    }


def _deepseek_state_dict() -> dict[str, torch.Tensor]:
    """Create synthetic weights for a tiny 1-layer DeepSeek V3 model."""
    sd: dict[str, torch.Tensor] = {}

    # Embedding
    sd["model.embed_tokens.weight"] = torch.randn(_VOCAB, _HIDDEN, dtype=torch.bfloat16)

    # Attention
    for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"):
        # Simplified dims — don't need to match real DeepSeek exactly
        sd[f"model.layers.0.self_attn.{proj}.weight"] = torch.randn(
            _HIDDEN, _HIDDEN, dtype=torch.bfloat16
        )

    # Layer norms
    sd["model.layers.0.input_layernorm.weight"] = torch.ones(_HIDDEN, dtype=torch.bfloat16)
    sd["model.layers.0.post_attention_layernorm.weight"] = torch.ones(_HIDDEN, dtype=torch.bfloat16)

    # Router
    sd["model.layers.0.mlp.gate.weight"] = torch.randn(_NUM_EXPERTS, _HIDDEN, dtype=torch.bfloat16)

    # Routed experts
    for i in range(_NUM_EXPERTS):
        for proj, shape in [
            ("gate_proj", (_INTER, _HIDDEN)),
            ("up_proj", (_INTER, _HIDDEN)),
            ("down_proj", (_HIDDEN, _INTER)),
        ]:
            sd[f"model.layers.0.mlp.experts.{i}.{proj}.weight"] = torch.randn(
                *shape, dtype=torch.bfloat16
            )

    # Shared experts
    for proj, shape in [
        ("gate_proj", (_INTER, _HIDDEN)),
        ("up_proj", (_INTER, _HIDDEN)),
        ("down_proj", (_HIDDEN, _INTER)),
    ]:
        sd[f"model.layers.0.mlp.shared_experts.{proj}.weight"] = torch.randn(
            *shape, dtype=torch.bfloat16
        )

    # LM head
    sd["lm_head.weight"] = torch.randn(_VOCAB, _HIDDEN, dtype=torch.bfloat16)

    # Final norm
    sd["model.norm.weight"] = torch.ones(_HIDDEN, dtype=torch.bfloat16)

    return sd


def _split_into_shards(
    sd: dict[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    """Split state dict into 2 shards: attention+embed in shard 1, MLP+rest in shard 2."""
    shard1: dict[str, torch.Tensor] = {}
    shard2: dict[str, torch.Tensor] = {}

    for key, tensor in sd.items():
        if "self_attn" in key or "embed_tokens" in key or "input_layernorm" in key:
            shard1[key] = tensor
        else:
            shard2[key] = tensor

    return {
        "model-00001-of-00002.safetensors": shard1,
        "model-00002-of-00002.safetensors": shard2,
    }


def _create_deepseek_model(model_dir: Path, shards: dict[str, dict[str, torch.Tensor]]) -> None:
    """Write a sharded DeepSeek model to disk."""
    model_dir.mkdir(parents=True, exist_ok=True)
    config = _deepseek_config()
    (model_dir / "config.json").write_text(json.dumps(config))

    weight_map: dict[str, str] = {}
    for shard_name, tensors in shards.items():
        save_file(tensors, str(model_dir / shard_name))
        for key in tensors:
            weight_map[key] = shard_name

    total_size = sum(t.nelement() * t.element_size() for s in shards.values() for t in s.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    # Minimal tokenizer
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

    # Custom model code files (for copy_model_code_files test)
    (model_dir / "configuration_deepseek.py").write_text("# DeepSeek config\n")
    (model_dir / "modeling_deepseek.py").write_text("# DeepSeek model\n")


def _create_deepseek_adapter(adapter_dir: Path) -> None:
    """Create a LoRA adapter targeting attention and expert weights."""
    adapter_dir.mkdir(parents=True, exist_ok=True)

    rank = 1
    weights: dict[str, torch.Tensor] = {}

    # Target attention q_a_proj
    weights["base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight"] = (
        torch.ones(rank, _HIDDEN) * 0.01
    )
    weights["base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.weight"] = torch.ones(
        _HIDDEN, rank
    )

    # Target routed experts gate_proj (w1): A shared, B per-expert
    weights["base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight"] = (
        torch.ones(1, rank, _HIDDEN) * 0.01
    )
    weights["base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight"] = torch.ones(
        _NUM_EXPERTS, _INTER, rank
    )

    save_file(weights, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"lora_alpha": 1, "r": rank}))


@pytest.fixture
def deepseek_model(tmp_path: Path):
    """Set up a tiny DeepSeek model + adapter."""
    sd = _deepseek_state_dict()
    shards = _split_into_shards(sd)
    model_dir = tmp_path / "model"
    adapter_dir = tmp_path / "adapter"

    _create_deepseek_model(model_dir, shards)
    _create_deepseek_adapter(adapter_dir)

    return model_dir, adapter_dir, sd


def _load_output(output_dir: Path) -> dict[str, torch.Tensor]:
    """Load all output tensors from a sharded output directory."""
    index_path = output_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        tensors: dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(weight_map.values())):
            tensors.update(load_file(str(output_dir / shard_name)))
        return tensors
    single = output_dir / "model.safetensors"
    assert single.exists()
    return load_file(str(single))


# ---------------------------------------------------------------------------
# Branch 1: Dense weights (attention/embedding)
# ---------------------------------------------------------------------------


class TestDenseWeights:
    def test_dense_weights_change_after_merge(self, tmp_path: Path, deepseek_model):
        model_dir, adapter_dir, orig_sd = deepseek_model
        output_dir = tmp_path / "output"

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            quantize="experts-fp8",
            serving_format="vllm",
        )

        out = _load_output(output_dir)
        key = "model.layers.0.self_attn.q_a_proj.weight"
        delta = (out[key].float() - orig_sd[key].float()).abs().sum()
        assert delta > 0, "q_a_proj should have changed after merge"

    def test_dense_weights_stay_bf16(self, tmp_path: Path, deepseek_model):
        model_dir, adapter_dir, _ = deepseek_model
        output_dir = tmp_path / "output"

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            quantize="experts-fp8",
            serving_format="vllm",
        )

        out = _load_output(output_dir)
        assert out["model.layers.0.self_attn.q_a_proj.weight"].dtype == torch.bfloat16

    def test_untargeted_dense_unchanged(self, tmp_path: Path, deepseek_model):
        model_dir, adapter_dir, orig_sd = deepseek_model
        output_dir = tmp_path / "output"

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            quantize="experts-fp8",
            serving_format="vllm",
        )

        out = _load_output(output_dir)
        key = "model.embed_tokens.weight"
        assert torch.equal(out[key], orig_sd[key]), "embed_tokens should be bit-identical"


# ---------------------------------------------------------------------------
# Branch 2: Routed expert weights
# ---------------------------------------------------------------------------


class TestRoutedExperts:
    def test_routed_experts_change_after_merge(self, tmp_path: Path, deepseek_model):
        model_dir, adapter_dir, orig_sd = deepseek_model
        output_dir = tmp_path / "output"

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            quantize="experts-fp8",
            serving_format="vllm",
        )

        out = _load_output(output_dir)
        from tinker_cookbook.weights._export._quantized import dequantize_blockwise

        key = "model.layers.0.mlp.experts.0.gate_proj.weight"
        scale_key = key.replace(".weight", ".weight_scale")
        merged = dequantize_blockwise(out[key], out[scale_key], dtype=torch.bfloat16)
        delta = (merged.float() - orig_sd[key].float()).abs().sum()
        assert delta > 0, "Expert gate_proj should have changed after merge"

    def test_routed_experts_quantized_to_fp8(self, tmp_path: Path, deepseek_model):
        model_dir, adapter_dir, _ = deepseek_model
        output_dir = tmp_path / "output"

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            quantize="experts-fp8",
            serving_format="vllm",
        )

        out = _load_output(output_dir)
        key = "model.layers.0.mlp.experts.0.gate_proj.weight"
        assert out[key].dtype == torch.float8_e4m3fn

    def test_expert_has_float32_scale(self, tmp_path: Path, deepseek_model):
        model_dir, adapter_dir, _ = deepseek_model
        output_dir = tmp_path / "output"

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            quantize="experts-fp8",
            serving_format="vllm",
        )

        out = _load_output(output_dir)
        scale_key = "model.layers.0.mlp.experts.0.gate_proj.weight_scale"
        assert scale_key in out
        assert out[scale_key].dtype == torch.float32

    def test_scale_shape_matches_block_structure(self, tmp_path: Path, deepseek_model):
        model_dir, adapter_dir, _ = deepseek_model
        output_dir = tmp_path / "output"

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            quantize="experts-fp8",
            serving_format="vllm",
        )

        out = _load_output(output_dir)
        scale = out["model.layers.0.mlp.experts.0.gate_proj.weight_scale"]
        # gate_proj shape is (_INTER, _HIDDEN) = (128, 64)
        # block_size = 128, so scale should be ceil(128/128) x ceil(64/128) = (1, 1)
        expected = (math.ceil(_INTER / 128), math.ceil(_HIDDEN / 128))
        assert scale.shape == expected


# ---------------------------------------------------------------------------
# Branch 3: Shared expert weights
# ---------------------------------------------------------------------------


class TestSharedExperts:
    def test_shared_experts_stay_bf16(self, tmp_path: Path, deepseek_model):
        model_dir, adapter_dir, _ = deepseek_model
        output_dir = tmp_path / "output"

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            quantize="experts-fp8",
            serving_format="vllm",
        )

        out = _load_output(output_dir)
        key = "model.layers.0.mlp.shared_experts.gate_proj.weight"
        assert out[key].dtype == torch.bfloat16

    def test_shared_experts_no_scale(self, tmp_path: Path, deepseek_model):
        model_dir, adapter_dir, _ = deepseek_model
        output_dir = tmp_path / "output"

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            quantize="experts-fp8",
            serving_format="vllm",
        )

        out = _load_output(output_dir)
        assert "model.layers.0.mlp.shared_experts.gate_proj.weight_scale" not in out


# ---------------------------------------------------------------------------
# Branch 4: Shard layout preservation
# ---------------------------------------------------------------------------


class TestShardLayout:
    def test_two_shard_input_produces_two_shard_output(self, tmp_path: Path, deepseek_model):
        model_dir, adapter_dir, _ = deepseek_model
        output_dir = tmp_path / "output"

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            quantize="experts-fp8",
            serving_format="vllm",
        )

        index_path = output_dir / "model.safetensors.index.json"
        assert index_path.exists()
        with open(index_path) as f:
            index = json.load(f)
        shard_files = set(index["weight_map"].values())
        assert len(shard_files) == 2

    def test_index_consistent(self, tmp_path: Path, deepseek_model):
        model_dir, adapter_dir, _ = deepseek_model
        output_dir = tmp_path / "output"

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            quantize="experts-fp8",
            serving_format="vllm",
        )

        index_path = output_dir / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)

        # All listed files should exist
        for shard_file in set(index["weight_map"].values()):
            assert (output_dir / shard_file).exists(), f"Missing shard: {shard_file}"

        # All keys in weight_map should exist in corresponding shard
        for key, shard_file in index["weight_map"].items():
            shard_tensors = load_file(str(output_dir / shard_file))
            assert key in shard_tensors, f"Key {key} not in {shard_file}"


# ---------------------------------------------------------------------------
# Branch 5: Config and metadata
# ---------------------------------------------------------------------------


class TestConfigMetadata:
    def test_compression_config_present(self, tmp_path: Path, deepseek_model):
        model_dir, adapter_dir, _ = deepseek_model
        output_dir = tmp_path / "output"

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            quantize="experts-fp8",
            serving_format="vllm",
        )

        config = json.loads((output_dir / "config.json").read_text())
        assert "compression_config" in config
        cc = config["compression_config"]
        assert cc["quant_method"] == "compressed-tensors"
        assert cc["format"] == "float-quantized"

    def test_quantization_config_absent(self, tmp_path: Path, deepseek_model):
        model_dir, adapter_dir, _ = deepseek_model
        # Add a quantization_config to the input config to verify it gets removed
        input_config = json.loads((model_dir / "config.json").read_text())
        input_config["quantization_config"] = {"quant_method": "fp8"}
        (model_dir / "config.json").write_text(json.dumps(input_config))

        output_dir = tmp_path / "output"
        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            quantize="experts-fp8",
            serving_format="vllm",
        )

        config = json.loads((output_dir / "config.json").read_text())
        assert "quantization_config" not in config

    def test_ignore_list_correct(self, tmp_path: Path, deepseek_model):
        model_dir, adapter_dir, _ = deepseek_model
        output_dir = tmp_path / "output"

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            quantize="experts-fp8",
            serving_format="vllm",
        )

        config = json.loads((output_dir / "config.json").read_text())
        ignore = config["compression_config"]["ignore"]
        # Dense projections should be in ignore
        assert "model.layers.0.self_attn.q_a_proj" in ignore
        # Routed experts should NOT be in ignore
        routed_in_ignore = [
            x for x in ignore if ".mlp.experts." in x and ".shared_experts." not in x
        ]
        assert len(routed_in_ignore) == 0, f"Routed experts in ignore: {routed_in_ignore}"

    def test_model_code_files_copied(self, tmp_path: Path, deepseek_model):
        model_dir, adapter_dir, _ = deepseek_model
        output_dir = tmp_path / "output"

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            quantize="experts-fp8",
            serving_format="vllm",
        )

        assert (output_dir / "configuration_deepseek.py").exists()
        assert (output_dir / "modeling_deepseek.py").exists()


# ---------------------------------------------------------------------------
# Branch 6: Resume (crash + restart)
# ---------------------------------------------------------------------------


class TestResume:
    def test_crash_after_shard_1_shows_in_progress(self, tmp_path: Path, deepseek_model):
        model_dir, adapter_dir, _ = deepseek_model
        output_dir = tmp_path / "output"

        # Monkeypatch to crash after first shard is saved
        call_count = 0
        original_load = load_file

        def crash_on_second_shard(path, *args, **kwargs):
            nonlocal call_count
            result = original_load(path, *args, **kwargs)
            # Count loads from model_dir (not adapter or output)
            if str(model_dir) in str(path) and "model-" in str(path):
                call_count += 1
                if call_count >= 2:
                    raise RuntimeError("Simulated crash")
            return result

        with patch(
            "tinker_cookbook.weights._export._quantized.load_file",
            side_effect=crash_on_second_shard,
        ):
            with pytest.raises(RuntimeError, match="Simulated crash"):
                build_hf_model(
                    base_model=str(model_dir),
                    adapter_path=str(adapter_dir),
                    output_path=str(output_dir),
                    quantize="experts-fp8",
                    serving_format="vllm",
                )

        # Check merge state
        state = json.loads((output_dir / "merge_state.json").read_text())
        assert state["status"] == "in_progress"
        assert len(state["completed_shards"]) == 1

    def test_resume_completes(self, tmp_path: Path, deepseek_model):
        model_dir, adapter_dir, _ = deepseek_model
        output_dir = tmp_path / "output"

        # First run: crash after shard 1
        call_count = 0
        original_load = load_file

        def crash_on_second_shard(path, *args, **kwargs):
            nonlocal call_count
            result = original_load(path, *args, **kwargs)
            if str(model_dir) in str(path) and "model-" in str(path):
                call_count += 1
                if call_count >= 2:
                    raise RuntimeError("Simulated crash")
            return result

        with patch(
            "tinker_cookbook.weights._export._quantized.load_file",
            side_effect=crash_on_second_shard,
        ):
            with pytest.raises(RuntimeError):
                build_hf_model(
                    base_model=str(model_dir),
                    adapter_path=str(adapter_dir),
                    output_path=str(output_dir),
                    quantize="experts-fp8",
                    serving_format="vllm",
                )

        # Second run: resume should complete
        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            quantize="experts-fp8",
            serving_format="vllm",
        )

        state = json.loads((output_dir / "merge_state.json").read_text())
        assert state["status"] == "completed"
        assert len(state["completed_shards"]) == 2


# ---------------------------------------------------------------------------
# Branch 7: New API interactions
# ---------------------------------------------------------------------------


class TestApiValidation:
    def test_quantize_none_does_standard_merge(self, tmp_path: Path, deepseek_model):
        """quantize=None with DeepSeek model should do standard BF16 merge."""
        model_dir, adapter_dir, _ = deepseek_model
        output_dir = tmp_path / "output"

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        out = _load_output(output_dir)
        # No FP8 tensors
        for key, tensor in out.items():
            assert tensor.dtype != torch.float8_e4m3fn, f"{key} should not be FP8"
        # No compression_config
        config = json.loads((output_dir / "config.json").read_text())
        assert "compression_config" not in config

    def test_quantize_without_serving_format_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="serving_format"):
            build_hf_model(
                base_model=str(tmp_path),
                adapter_path=str(tmp_path),
                output_path=str(tmp_path / "out"),
                quantize="experts-fp8",
            )

    def test_serving_format_without_quantize_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="quantize"):
            build_hf_model(
                base_model=str(tmp_path),
                adapter_path=str(tmp_path),
                output_path=str(tmp_path / "out"),
                serving_format="vllm",
            )

    def test_quantize_with_wrong_dtype_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="bfloat16"):
            build_hf_model(
                base_model=str(tmp_path),
                adapter_path=str(tmp_path),
                output_path=str(tmp_path / "out"),
                quantize="experts-fp8",
                serving_format="vllm",
                dtype="float32",
            )

    def test_unknown_quantize_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="quantize"):
            build_hf_model(
                base_model=str(tmp_path),
                adapter_path=str(tmp_path),
                output_path=str(tmp_path / "out"),
                quantize="unknown",
                serving_format="vllm",
            )

    def test_unknown_serving_format_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="serving_format"):
            build_hf_model(
                base_model=str(tmp_path),
                adapter_path=str(tmp_path),
                output_path=str(tmp_path / "out"),
                quantize="experts-fp8",
                serving_format="unknown",
            )
