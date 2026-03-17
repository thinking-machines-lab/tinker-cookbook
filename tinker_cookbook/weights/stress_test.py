"""Stress tests for the weights module.

Exercises edge cases, numerical correctness, and cross-path consistency
to catch subtle bugs. Uses synthetic models — no network or GPU required.
"""

import json
import logging
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tinker_cookbook.weights._artifacts import (
    ShardWriter,
    get_model_state_shapes,
)
from tinker_cookbook.weights._export import build_hf_model, load_config_dict
from tinker_cookbook.weights._merge import (
    MergeOp,
    MergeProfile,
    apply_merge_op,
    detect_merge_profile,
    merge_adapter_weights,
    merge_lora_matrices,
    plan_merge_ops,
    validate_merge_op_shapes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_model(path: Path, config: dict, state_dict: dict) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config))
    save_file(state_dict, str(path / "model.safetensors"))
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


def _create_adapter(path: Path, weights: dict, config: dict) -> None:
    path.mkdir(parents=True)
    save_file(weights, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps(config))


# ---------------------------------------------------------------------------
# Numerical correctness: verify exact LoRA delta values
# ---------------------------------------------------------------------------


class TestNumericalCorrectness:
    """Verify that the LoRA math produces exactly the right values."""

    def test_merge_lora_matrices_matches_manual(self):
        """B @ A should match manual computation."""
        lora_A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        lora_B = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # (3, 2)
        result = merge_lora_matrices(lora_A, lora_B)
        # B @ A = [[1,2,3], [4,5,6], [5,7,9]]
        expected = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]])
        assert torch.allclose(result, expected)

    def test_scaling_applied_correctly(self):
        """lora_alpha/r scaling should multiply lora_B before merge."""
        keys = {"w.weight"}
        profile = MergeProfile()
        adapter = {
            "base_model.model.w.lora_A.weight": torch.ones(1, 4),
            "base_model.model.w.lora_B.weight": torch.ones(8, 1),
        }
        # alpha=4, r=2 → scaling=2, so B becomes 2*ones
        ops = plan_merge_ops(adapter, {"lora_alpha": 4, "r": 2}, keys, profile)
        op = ops["w.weight"][0]
        # delta = scaled_B @ A = 2*ones(8,1) @ ones(1,4) = 2*ones(8,4)
        tensors = {"w.weight": torch.zeros(8, 4)}
        apply_merge_op(tensors, op)
        assert torch.allclose(tensors["w.weight"], torch.full((8, 4), 2.0))

    def test_higher_rank_lora(self):
        """Rank > 1 LoRA should produce correct delta."""
        rank = 4
        lora_A = torch.eye(rank, 8)  # (4, 8)
        lora_B = torch.ones(8, rank)  # (8, 4)
        delta = merge_lora_matrices(lora_A, lora_B)
        # ones(8,4) @ eye(4,8) = first 4 cols are ones, rest are zeros
        assert delta.shape == (8, 8)
        assert torch.allclose(delta[:, :4], torch.ones(8, 4))
        assert torch.allclose(delta[:, 4:], torch.zeros(8, 4))

    def test_bfloat16_precision(self):
        """Merge should upcast to float32 then cast back, preserving precision."""
        tensors = {"w": torch.zeros(4, 4, dtype=torch.bfloat16)}
        op = MergeOp(
            target_key="w",
            lora_A=torch.ones(1, 4, dtype=torch.float32) * 0.001,
            lora_B=torch.ones(4, 1, dtype=torch.float32),
        )
        apply_merge_op(tensors, op)
        # Result should be close to 0.001 (bfloat16 can represent this approximately)
        assert tensors["w"].dtype == torch.bfloat16
        assert tensors["w"].float().mean().item() == pytest.approx(0.001, abs=1e-4)


# ---------------------------------------------------------------------------
# Cross-path consistency: full vs shard should produce identical output
# (using synthetic models that work with both paths)
# ---------------------------------------------------------------------------


class TestCrossPathConsistency:
    """Verify plan_merge_ops + apply_merge_op matches merge_adapter_weights."""

    def _make_model_and_adapter(self):
        """Create synthetic model state dict and adapter weights."""
        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 4),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(8, 4),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 4),
        }
        adapter = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(2, 4),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(8, 2),
            "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight": torch.randn(2, 4),
            "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight": torch.randn(16, 2),
        }
        config = {"lora_alpha": 8, "r": 2}
        return state_dict, adapter, config

    def test_plan_apply_matches_merge_adapter_weights(self):
        """Plan+apply path should produce bit-identical results to merge_adapter_weights."""
        state_dict, adapter, config = self._make_model_and_adapter()

        # Path 1: merge_adapter_weights (backward-compat wrapper)
        sd1 = {k: v.clone() for k, v in state_dict.items()}
        model1 = type("Model", (), {"state_dict": lambda self: sd1})()
        merge_adapter_weights(model1, adapter, config)

        # Path 2: plan + apply (new API)
        sd2 = {k: v.clone() for k, v in state_dict.items()}
        profile = detect_merge_profile({"architectures": []}, set(sd2.keys()))
        ops = plan_merge_ops(adapter, config, set(sd2.keys()), profile)
        for op_list in ops.values():
            for op in op_list:
                apply_merge_op(sd2, op)

        # Must be bit-identical
        for key in state_dict:
            assert torch.equal(sd1[key], sd2[key]), f"Mismatch on {key}"

    def test_multiple_adapters_targeting_same_key(self):
        """Multiple LoRA ops on the same key should all be applied."""
        tensors = {"w.weight": torch.zeros(4, 4)}
        # Two ops targeting the same key
        ops = {
            "w.weight": [
                MergeOp(target_key="w.weight", lora_A=torch.ones(1, 4), lora_B=torch.ones(4, 1)),
                MergeOp(
                    target_key="w.weight",
                    lora_A=torch.ones(1, 4) * 2,
                    lora_B=torch.ones(4, 1),
                ),
            ]
        }
        for op in ops["w.weight"]:
            apply_merge_op(tensors, op)
        # First adds 1.0, second adds 2.0 → total 3.0
        assert torch.allclose(tensors["w.weight"], torch.full((4, 4), 3.0))


# ---------------------------------------------------------------------------
# Expert merge edge cases
# ---------------------------------------------------------------------------


class TestExpertEdgeCases:
    def test_fused_gate_and_up_both_applied(self):
        """Both gate and up ops should be applied to the same fused tensor."""
        n_exp, in_dim, fused_dim = 2, 4, 8
        tensors = {"fused": torch.zeros(n_exp, in_dim, fused_dim)}

        # Gate op (idx=0, concatenated) → first half
        gate_op = MergeOp(
            target_key="fused",
            lora_A=torch.ones(n_exp, 1, in_dim) * 0.1,
            lora_B=torch.ones(n_exp, fused_dim // 2, 1),
            is_expert_3d=True,
            fused_proj_idx=0,
        )
        # Up op (idx=1, concatenated) → second half
        up_op = MergeOp(
            target_key="fused",
            lora_A=torch.ones(n_exp, 1, in_dim) * 0.2,
            lora_B=torch.ones(n_exp, fused_dim // 2, 1),
            is_expert_3d=True,
            fused_proj_idx=1,
        )
        apply_merge_op(tensors, gate_op)
        apply_merge_op(tensors, up_op)

        gate_half = tensors["fused"][:, :, : fused_dim // 2]
        up_half = tensors["fused"][:, :, fused_dim // 2 :]
        assert torch.allclose(gate_half, torch.full_like(gate_half, 0.1), atol=1e-6)
        assert torch.allclose(up_half, torch.full_like(up_half, 0.2), atol=1e-6)

    def test_fused_down_proj_no_slicing(self):
        """Down proj (fused_proj_idx=None) should apply to full tensor."""
        n_exp, in_dim, out_dim = 2, 4, 8
        tensors = {"down": torch.zeros(n_exp, in_dim, out_dim)}
        op = MergeOp(
            target_key="down",
            lora_A=torch.ones(n_exp, 1, in_dim) * 0.3,
            lora_B=torch.ones(n_exp, out_dim, 1),
            is_expert_3d=True,
            fused_proj_idx=None,
        )
        apply_merge_op(tensors, op)
        assert torch.allclose(tensors["down"], torch.full_like(tensors["down"], 0.3), atol=1e-6)


# ---------------------------------------------------------------------------
# Shard path edge cases
# ---------------------------------------------------------------------------


class TestShardPathEdgeCases:
    def test_adapter_targets_subset_of_model_keys(self, tmp_path: Path):
        """Adapter only touches some model keys; untouched keys should be preserved."""
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        original_value = torch.randn(8, 4)
        _create_model(
            model_dir,
            {"architectures": ["Test"]},
            {
                "model.layers.0.q_proj.weight": torch.zeros(8, 4),
                "model.layers.0.k_proj.weight": original_value.clone(),
            },
        )
        _create_adapter(
            adapter_dir,
            {
                "base_model.model.model.layers.0.q_proj.lora_A.weight": torch.ones(1, 4),
                "base_model.model.model.layers.0.q_proj.lora_B.weight": torch.ones(8, 1),
            },
            {"lora_alpha": 1, "r": 1},
        )

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            merge_strategy="shard",
        )

        out = load_file(str(output_dir / "model.safetensors"))
        # q_proj was merged
        assert out["model.layers.0.q_proj.weight"].abs().sum() > 0
        # k_proj should be exactly preserved (bit-for-bit)
        assert torch.equal(out["model.layers.0.k_proj.weight"], original_value)

    def test_empty_adapter_produces_identical_output(self, tmp_path: Path, caplog):
        """Adapter with no LoRA weights should produce model identical to input."""
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        original = torch.randn(4, 4)
        _create_model(
            model_dir,
            {"architectures": ["Test"]},
            {"model.weight": original.clone()},
        )
        # Adapter with no LoRA keys (just some random tensor)
        _create_adapter(adapter_dir, {"random_key": torch.zeros(1)}, {"lora_alpha": 1, "r": 1})

        with caplog.at_level(logging.WARNING):
            build_hf_model(
                base_model=str(model_dir),
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
                merge_strategy="shard",
            )

        assert "No LoRA weights found" in caplog.text
        out = load_file(str(output_dir / "model.safetensors"))
        assert torch.equal(out["model.weight"], original)

    def test_output_exists_raises_before_any_work(self, tmp_path: Path):
        """Should raise FileExistsError early, before expensive merge work."""
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_model(
            model_dir,
            {"architectures": ["Test"]},
            {"model.w.weight": torch.zeros(4, 4)},
        )
        _create_adapter(
            adapter_dir,
            {
                "base_model.model.model.w.lora_A.weight": torch.ones(1, 4),
                "base_model.model.model.w.lora_B.weight": torch.ones(4, 1),
            },
            {"lora_alpha": 1, "r": 1},
        )
        output_dir.mkdir()  # pre-create to trigger conflict

        with pytest.raises(FileExistsError, match="already exists"):
            build_hf_model(
                base_model=str(model_dir),
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
                merge_strategy="shard",
            )


# ---------------------------------------------------------------------------
# Shape validation edge cases
# ---------------------------------------------------------------------------


class TestShapeValidationEdgeCases:
    def test_shapes_read_without_loading_tensors(self, tmp_path: Path):
        """get_model_state_shapes should be fast and not load actual tensor data."""
        # Create a model with known shapes
        tensors = {
            "a": torch.zeros(100, 200),
            "b": torch.zeros(3, 4, 5),
            "c": torch.zeros(7),
        }
        save_file(tensors, str(tmp_path / "model.safetensors"))

        shapes = get_model_state_shapes(tmp_path)
        assert shapes == {"a": (100, 200), "b": (3, 4, 5), "c": (7,)}

    def test_validate_catches_wrong_rank(self):
        """Shape validation should catch rank-1 LoRA targeting rank-2 weight."""
        ops = {"w": [MergeOp(target_key="w", lora_A=torch.ones(1, 4), lora_B=torch.ones(8, 1))]}
        shapes = {"w": (4, 8)}  # transposed from what the LoRA produces
        with pytest.raises(ValueError, match="Shape mismatch"):
            validate_merge_op_shapes(ops, shapes)


# ---------------------------------------------------------------------------
# Profile detection edge cases
# ---------------------------------------------------------------------------


class TestProfileDetectionEdgeCases:
    def test_gpt_oss_without_fused_experts(self):
        """GPT-OSS model without experts should get separate layout."""
        config: dict = {"architectures": ["GptOssForCausalLM"]}
        keys = {"model.layers.0.self_attn.q_proj.weight"}
        profile = detect_merge_profile(config, keys)
        assert profile.expert_layout == "separate"
        assert (".attn", ".self_attn") in profile.extra_key_remaps

    def test_vision_gpt_oss_combination(self):
        """GPT-OSS vision model should get both attn remap and language_model prefix."""
        config: dict = {"architectures": ["GptOssVisionModel"]}
        keys = {
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
        }
        profile = detect_merge_profile(config, keys)
        assert profile.expert_layout == "fused_interleaved"
        assert profile.has_language_model_prefix is True
        assert (".attn", ".self_attn") in profile.extra_key_remaps


# ---------------------------------------------------------------------------
# Config loading edge cases
# ---------------------------------------------------------------------------


class TestConfigLoadingEdgeCases:
    def test_local_dir_without_config_raises(self, tmp_path: Path):
        """Local directory without config.json should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match=r"config\.json"):
            load_config_dict(tmp_path)

    def test_local_dir_with_config_loads(self, tmp_path: Path):
        (tmp_path / "config.json").write_text('{"architectures": ["TestModel"]}')
        config = load_config_dict(tmp_path)
        assert config["architectures"] == ["TestModel"]


# ---------------------------------------------------------------------------
# ShardWriter edge cases
# ---------------------------------------------------------------------------


class TestShardWriterEdgeCases:
    def test_single_tensor_larger_than_max_shard(self, tmp_path: Path):
        """A tensor larger than max_shard_size should get its own shard."""
        writer = ShardWriter(tmp_path, max_shard_size=100)
        # float32 tensor of 1000 elements = 4000 bytes >> 100 byte limit
        writer.add_tensor("big", torch.zeros(1000))
        writer.add_tensor("small", torch.zeros(1))
        weight_map = writer.finalize()

        assert len(set(weight_map.values())) == 2
        assert weight_map["big"] != weight_map["small"]

    def test_shard_count_correct_after_multiple_flushes(self, tmp_path: Path):
        writer = ShardWriter(tmp_path, max_shard_size=100)
        for i in range(5):
            writer.add_tensor(f"t{i}", torch.zeros(100))  # each triggers flush
        weight_map = writer.finalize()
        assert len(set(weight_map.values())) == 5
