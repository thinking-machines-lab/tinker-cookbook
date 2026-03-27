"""Equivalence tests: verify our quantized export matches PR #470 behavior.

PR #470 (tinker_cookbook/weights/_deepseek.py) established the reference behavior
for DeepSeek FP8 export. This test suite verifies that our reimplementation
in _export/_quantized.py produces equivalent output.

Uses a tiny 1-layer DeepSeek V3 model with synthetic weights — no network needed.
"""

import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tinker_cookbook.weights import build_hf_model

# ---------------------------------------------------------------------------
# Constants matching PR #470
# ---------------------------------------------------------------------------

_HIDDEN = 64
_INTER = 128  # moe_intermediate_size in real DeepSeek, but simplified here
_NUM_EXPERTS = 2
_VOCAB = 256
_BLOCK_SIZE = 128  # DeepSeek native FP8 block size

# PR #470 used these suffixes to determine what goes in the ignore list
_LINEAR_PROJ_SUFFIXES = (
    ".q_proj.weight",
    ".q_a_proj.weight",
    ".q_b_proj.weight",
    ".kv_a_proj_with_mqa.weight",
    ".kv_b_proj.weight",
    ".o_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
)


# ---------------------------------------------------------------------------
# Test model setup
# ---------------------------------------------------------------------------


def _create_test_model(model_dir: Path) -> dict[str, torch.Tensor]:
    """Create a tiny sharded DeepSeek V3 model matching PR #470 test structure."""
    sd: dict[str, torch.Tensor] = {}

    # Embedding
    sd["model.embed_tokens.weight"] = torch.randn(_VOCAB, _HIDDEN, dtype=torch.bfloat16)

    # Attention (using DeepSeek-specific projection names)
    for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"):
        sd[f"model.layers.0.self_attn.{proj}.weight"] = torch.randn(
            _HIDDEN, _HIDDEN, dtype=torch.bfloat16
        )

    # Layer norms
    sd["model.layers.0.input_layernorm.weight"] = torch.ones(_HIDDEN, dtype=torch.bfloat16)
    sd["model.layers.0.post_attention_layernorm.weight"] = torch.ones(_HIDDEN, dtype=torch.bfloat16)

    # Router
    sd["model.layers.0.mlp.gate.weight"] = torch.randn(_NUM_EXPERTS, _HIDDEN, dtype=torch.bfloat16)

    # Routed experts (gate, up, down for each expert)
    for i in range(_NUM_EXPERTS):
        sd[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.randn(
            _INTER, _HIDDEN, dtype=torch.bfloat16
        )
        sd[f"model.layers.0.mlp.experts.{i}.up_proj.weight"] = torch.randn(
            _INTER, _HIDDEN, dtype=torch.bfloat16
        )
        sd[f"model.layers.0.mlp.experts.{i}.down_proj.weight"] = torch.randn(
            _HIDDEN, _INTER, dtype=torch.bfloat16
        )

    # Shared experts
    sd["model.layers.0.mlp.shared_experts.gate_proj.weight"] = torch.randn(
        _INTER, _HIDDEN, dtype=torch.bfloat16
    )
    sd["model.layers.0.mlp.shared_experts.up_proj.weight"] = torch.randn(
        _INTER, _HIDDEN, dtype=torch.bfloat16
    )
    sd["model.layers.0.mlp.shared_experts.down_proj.weight"] = torch.randn(
        _HIDDEN, _INTER, dtype=torch.bfloat16
    )

    # LM head
    sd["lm_head.weight"] = torch.randn(_VOCAB, _HIDDEN, dtype=torch.bfloat16)

    # Final norm
    sd["model.norm.weight"] = torch.ones(_HIDDEN, dtype=torch.bfloat16)

    # Reshard into 2 shards (matching PR #470's test pattern)
    shard1_keys = {
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
    }

    shard1 = {k: v for k, v in sd.items() if k in shard1_keys}
    shard2 = {k: v for k, v in sd.items() if k not in shard1_keys}

    shards = {
        "model-00001-of-00002.safetensors": shard1,
        "model-00002-of-00002.safetensors": shard2,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    config = {
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
    (model_dir / "config.json").write_text(json.dumps(config))

    weight_map: dict[str, str] = {}
    total_size = 0
    for shard_name, tensors in shards.items():
        save_file(tensors, str(model_dir / shard_name))
        for key, tensor in tensors.items():
            weight_map[key] = shard_name
            total_size += tensor.nelement() * tensor.element_size()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    # Tokenizer
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

    # Custom model code
    (model_dir / "configuration_deepseek.py").write_text("# config\n")
    (model_dir / "modeling_deepseek.py").write_text("# model\n")

    return sd


def _create_test_adapter(adapter_dir: Path) -> None:
    """Create adapter targeting both dense and expert weights (matching PR #470)."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    rank = 1
    weights: dict[str, torch.Tensor] = {}

    # Dense: q_a_proj
    weights["base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight"] = (
        torch.ones(rank, _HIDDEN, dtype=torch.bfloat16) * 0.01
    )
    weights["base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.weight"] = torch.ones(
        _HIDDEN, rank, dtype=torch.bfloat16
    )

    # Expert gate (w1): A shared, B per-expert
    weights["base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight"] = (
        torch.ones(1, rank, _HIDDEN, dtype=torch.bfloat16) * 0.01
    )
    weights["base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight"] = torch.ones(
        _NUM_EXPERTS, _INTER, rank, dtype=torch.bfloat16
    )

    # Expert up (w3): A shared, B per-expert
    weights["base_model.model.model.layers.0.mlp.experts.w3.lora_A.weight"] = (
        torch.ones(1, rank, _HIDDEN, dtype=torch.bfloat16) * 0.05
    )
    weights["base_model.model.model.layers.0.mlp.experts.w3.lora_B.weight"] = torch.ones(
        _NUM_EXPERTS, _INTER, rank, dtype=torch.bfloat16
    )

    save_file(weights, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"lora_alpha": 1, "r": rank}))


def _load_all_output_tensors(output_dir: Path) -> dict[str, torch.Tensor]:
    """Load all output tensors from safetensors files."""
    tensors: dict[str, torch.Tensor] = {}
    for path in sorted(output_dir.glob("*.safetensors")):
        tensors.update(load_file(str(path)))
    return tensors


@pytest.fixture
def equivalence_model(tmp_path: Path):
    """Set up test model + adapter + run build."""
    model_dir = tmp_path / "model"
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "output"

    orig_sd = _create_test_model(model_dir)
    _create_test_adapter(adapter_dir)

    build_hf_model(
        base_model=str(model_dir),
        adapter_path=str(adapter_dir),
        output_path=str(output_dir),
        quantize="experts-fp8",
        serving_format="vllm",
    )

    saved_sd = _load_all_output_tensors(output_dir)
    saved_config = json.loads((output_dir / "config.json").read_text())
    saved_index = json.loads((output_dir / "model.safetensors.index.json").read_text())

    return {
        "orig_sd": orig_sd,
        "saved_sd": saved_sd,
        "saved_config": saved_config,
        "saved_index": saved_index,
        "output_dir": output_dir,
        "model_dir": model_dir,
    }


# ---------------------------------------------------------------------------
# 1. Scale tensor naming: must use .weight_scale (not .weight_scale_inv)
#    PR #470 uses compressed-tensors convention: .weight_scale
# ---------------------------------------------------------------------------


class TestScaleTensorNaming:
    def test_routed_expert_scales_use_weight_scale_name(self, equivalence_model):
        """PR #470 emits .weight_scale, not .weight_scale_inv."""
        sd = equivalence_model["saved_sd"]

        for i in range(_NUM_EXPERTS):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.0.mlp.experts.{i}.{proj}.weight_scale"
                inv_key = f"model.layers.0.mlp.experts.{i}.{proj}.weight_scale_inv"
                assert key in sd, f"Expected {key} in output (PR #470 convention)"
                assert inv_key not in sd, f"Should not have {inv_key} (DeepSeek native convention)"

    def test_no_weight_scale_inv_in_output(self, equivalence_model):
        """PR #470 explicitly asserts: no .weight_scale_inv in output."""
        sd = equivalence_model["saved_sd"]
        inv_keys = [k for k in sd if k.endswith(".weight_scale_inv")]
        assert not inv_keys, f"Found .weight_scale_inv keys (should be .weight_scale): {inv_keys}"


# ---------------------------------------------------------------------------
# 2. Compressed-tensors config schema
#    PR #470 uses strategy="block" with block_structure=[128, 128]
#    and includes input_activations with dynamic=True
# ---------------------------------------------------------------------------


class TestCompressedTensorsConfig:
    def test_weights_strategy_is_block(self, equivalence_model):
        """PR #470: config_groups.group_0.weights.strategy == 'block'."""
        cc = equivalence_model["saved_config"]["compression_config"]
        weights = cc["config_groups"]["group_0"]["weights"]
        assert weights["strategy"] == "block", (
            f"Expected strategy='block' (PR #470), got {weights.get('strategy')!r}"
        )

    def test_block_structure_present(self, equivalence_model):
        """PR #470: config_groups.group_0.weights.block_structure == [128, 128]."""
        cc = equivalence_model["saved_config"]["compression_config"]
        weights = cc["config_groups"]["group_0"]["weights"]
        assert weights.get("block_structure") == [128, 128], (
            f"Expected block_structure=[128, 128], got {weights.get('block_structure')}"
        )

    def test_input_activations_dynamic(self, equivalence_model):
        """PR #470: input_activations with dynamic=True."""
        cc = equivalence_model["saved_config"]["compression_config"]
        group = cc["config_groups"]["group_0"]
        assert "input_activations" in group, "Missing input_activations section"
        assert group["input_activations"]["dynamic"] is True, (
            "input_activations.dynamic should be True"
        )

    def test_quant_method_compressed_tensors(self, equivalence_model):
        cc = equivalence_model["saved_config"]["compression_config"]
        assert cc["quant_method"] == "compressed-tensors"

    def test_format_float_quantized(self, equivalence_model):
        cc = equivalence_model["saved_config"]["compression_config"]
        assert cc["format"] == "float-quantized"

    def test_quantization_status_compressed(self, equivalence_model):
        cc = equivalence_model["saved_config"]["compression_config"]
        assert cc["quantization_status"] == "compressed"

    def test_quantization_config_absent(self, equivalence_model):
        assert "quantization_config" not in equivalence_model["saved_config"]


# ---------------------------------------------------------------------------
# 3. Ignore list: PR #470 uses _LINEAR_PROJ_SUFFIXES, not all .weight keys
# ---------------------------------------------------------------------------


class TestIgnoreList:
    def test_dense_projections_in_ignore(self, equivalence_model):
        """Dense linear projections must be in ignore list."""
        ignore = set(equivalence_model["saved_config"]["compression_config"]["ignore"])
        assert "model.layers.0.self_attn.q_a_proj" in ignore
        assert "model.layers.0.self_attn.q_b_proj" in ignore
        assert "model.layers.0.self_attn.kv_a_proj_with_mqa" in ignore
        assert "model.layers.0.self_attn.kv_b_proj" in ignore

    def test_shared_experts_in_ignore(self, equivalence_model):
        ignore = set(equivalence_model["saved_config"]["compression_config"]["ignore"])
        assert "model.layers.0.mlp.shared_experts.gate_proj" in ignore
        assert "model.layers.0.mlp.shared_experts.up_proj" in ignore
        assert "model.layers.0.mlp.shared_experts.down_proj" in ignore

    def test_routed_experts_not_in_ignore(self, equivalence_model):
        ignore = set(equivalence_model["saved_config"]["compression_config"]["ignore"])
        for i in range(_NUM_EXPERTS):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                assert f"model.layers.0.mlp.experts.{i}.{proj}" not in ignore

    def test_lm_head_in_ignore(self, equivalence_model):
        """PR #470 explicitly adds lm_head to ignore if not quantized."""
        ignore = set(equivalence_model["saved_config"]["compression_config"]["ignore"])
        assert "lm_head" in ignore


# ---------------------------------------------------------------------------
# 4. Weight dtype and merge correctness
# ---------------------------------------------------------------------------


class TestWeightDtypes:
    def test_dense_weights_bf16(self, equivalence_model):
        sd = equivalence_model["saved_sd"]
        assert sd["model.layers.0.self_attn.q_a_proj.weight"].dtype == torch.bfloat16

    def test_routed_experts_fp8(self, equivalence_model):
        sd = equivalence_model["saved_sd"]
        for i in range(_NUM_EXPERTS):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.0.mlp.experts.{i}.{proj}.weight"
                assert sd[key].dtype == torch.float8_e4m3fn, f"{key} should be FP8"

    def test_routed_expert_scales_float32(self, equivalence_model):
        sd = equivalence_model["saved_sd"]
        for i in range(_NUM_EXPERTS):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.0.mlp.experts.{i}.{proj}.weight_scale"
                assert sd[key].dtype == torch.float32, f"{key} should be float32"

    def test_shared_experts_bf16(self, equivalence_model):
        sd = equivalence_model["saved_sd"]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.0.mlp.shared_experts.{proj}.weight"
            assert sd[key].dtype == torch.bfloat16

    def test_dense_weight_changed_after_merge(self, equivalence_model):
        orig = equivalence_model["orig_sd"]
        saved = equivalence_model["saved_sd"]
        key = "model.layers.0.self_attn.q_a_proj.weight"
        delta = (saved[key].float() - orig[key].float()).abs().sum()
        assert delta > 0

    def test_untargeted_embedding_unchanged(self, equivalence_model):
        orig = equivalence_model["orig_sd"]
        saved = equivalence_model["saved_sd"]
        assert torch.equal(saved["model.embed_tokens.weight"], orig["model.embed_tokens.weight"])


# ---------------------------------------------------------------------------
# 5. FP8 scale shape matches block structure
# ---------------------------------------------------------------------------


class TestScaleShapes:
    def test_scale_shape_matches_blockwise_quantization(self, equivalence_model):
        """Scale shape must be ceil(rows/128) x ceil(cols/128)."""
        sd = equivalence_model["saved_sd"]
        # gate_proj: (_INTER, _HIDDEN) = (128, 64)
        scale = sd["model.layers.0.mlp.experts.0.gate_proj.weight_scale"]
        expected = (math.ceil(_INTER / _BLOCK_SIZE), math.ceil(_HIDDEN / _BLOCK_SIZE))
        assert scale.shape == expected, f"Expected {expected}, got {tuple(scale.shape)}"

    def test_down_proj_scale_shape(self, equivalence_model):
        sd = equivalence_model["saved_sd"]
        # down_proj: (_HIDDEN, _INTER) = (64, 128)
        scale = sd["model.layers.0.mlp.experts.0.down_proj.weight_scale"]
        expected = (math.ceil(_HIDDEN / _BLOCK_SIZE), math.ceil(_INTER / _BLOCK_SIZE))
        assert scale.shape == expected


# ---------------------------------------------------------------------------
# 6. Shard layout and index consistency
# ---------------------------------------------------------------------------


class TestShardConsistency:
    def test_two_shard_output(self, equivalence_model):
        index = equivalence_model["saved_index"]
        assert len(set(index["weight_map"].values())) == 2

    def test_index_covers_all_tensors(self, equivalence_model):
        """PR #470: weight_map should cover every emitted tensor exactly once."""
        index = equivalence_model["saved_index"]
        sd = equivalence_model["saved_sd"]
        assert set(index["weight_map"]) == set(sd)

    def test_all_shards_referenced_and_exist(self, equivalence_model):
        index = equivalence_model["saved_index"]
        output_dir = equivalence_model["output_dir"]
        for shard_file in set(index["weight_map"].values()):
            assert (output_dir / shard_file).exists()

    def test_scale_tensors_in_same_shard_as_weights(self, equivalence_model):
        """PR #470: scale tensors should be alongside their weight tensors."""
        wm = equivalence_model["saved_index"]["weight_map"]
        for i in range(_NUM_EXPERTS):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                weight_key = f"model.layers.0.mlp.experts.{i}.{proj}.weight"
                scale_key = f"model.layers.0.mlp.experts.{i}.{proj}.weight_scale"
                if weight_key in wm and scale_key in wm:
                    assert wm[weight_key] == wm[scale_key], (
                        f"Scale {scale_key} not in same shard as {weight_key}"
                    )


# ---------------------------------------------------------------------------
# 7. Custom files copied
# ---------------------------------------------------------------------------


class TestCustomFiles:
    def test_configuration_deepseek_copied(self, equivalence_model):
        assert (equivalence_model["output_dir"] / "configuration_deepseek.py").exists()

    def test_modeling_deepseek_copied(self, equivalence_model):
        assert (equivalence_model["output_dir"] / "modeling_deepseek.py").exists()
