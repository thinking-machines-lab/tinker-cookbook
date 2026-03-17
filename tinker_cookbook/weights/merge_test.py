"""Unit tests for LoRA merge logic.

Uses synthetic tensors to cover all code paths without needing real models
or network access.
"""

from typing import Any

import pytest
import torch

from tinker_cookbook.weights._merge import (
    MergeOp,
    MergeProfile,
    apply_merge_op,
    apply_merged_weight,
    detect_merge_profile,
    expand_expert_lora_tensors,
    merge_adapter_weights,
    merge_lora_matrices,
    plan_merge_ops,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_base_model(state_dict: dict[str, torch.Tensor], class_name: str = "SomeModel") -> Any:
    """Create a minimal mock model with a real state_dict and controllable class name.

    Uses a dynamically-created class so ``str(type(model))`` contains the
    desired class name (important for GPT-OSS detection).
    """
    cls = type(class_name, (), {"state_dict": lambda self: state_dict})
    return cls()


def _make_expert_lora_pair(
    num_experts: int, out_dim: int, in_dim: int, rank: int = 1, fill: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create LoRA A/B pair for experts with predictable merged output.

    lora_A = fill * ones, lora_B = ones → merged = fill * ones(in_dim, out_dim) * rank.
    """
    lora_A = torch.ones(num_experts, rank, in_dim) * fill
    lora_B = torch.ones(num_experts, out_dim, rank)
    return lora_A, lora_B


# ---------------------------------------------------------------------------
# apply_merged_weight
# ---------------------------------------------------------------------------


class TestApplyMergedWeight:
    def test_adds_delta_in_place(self):
        target = torch.zeros(3, 4)
        delta = torch.ones(3, 4) * 0.5
        apply_merged_weight(target, delta)
        assert torch.allclose(target, torch.full((3, 4), 0.5))

    def test_raises_on_shape_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            apply_merged_weight(torch.zeros(3, 4), torch.zeros(3, 5))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_missing_lora_alpha(self):
        model = _make_base_model({})
        with pytest.raises(KeyError, match="lora_alpha"):
            merge_adapter_weights(model, {}, {"r": 1})

    def test_missing_r(self):
        model = _make_base_model({})
        with pytest.raises(KeyError, match="'r'"):
            merge_adapter_weights(model, {}, {"lora_alpha": 1})


# ---------------------------------------------------------------------------
# Non-expert linear layers
# ---------------------------------------------------------------------------


class TestNonExpertMerge:
    def test_standard_linear_merge(self):
        state_dict = {"model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 4)}
        model = _make_base_model(state_dict)

        # rank=1 LoRA: A=(1,4), B=(8,1) → merged=(8,4) all equal to fill*scaling
        adapter_weights = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1, 4),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(8, 1),
        }

        merge_adapter_weights(model, adapter_weights, {"lora_alpha": 2, "r": 1})

        result = state_dict["model.layers.0.self_attn.q_proj.weight"]
        # merged = linear(A.T, B*scaling).T = linear((4,1), (8,1)*2).T
        # = (4,8)*2 transposed... actually let's just check it's nonzero and uniform
        assert result.abs().sum() > 0
        assert torch.allclose(result, result[0, 0].expand_as(result))

    def test_missing_target_key_raises(self):
        model = _make_base_model({"some.other.weight": torch.zeros(4, 4)})
        adapter_weights = {
            "base_model.model.model.layers.0.q_proj.lora_A.weight": torch.ones(1, 4),
            "base_model.model.model.layers.0.q_proj.lora_B.weight": torch.ones(4, 1),
        }
        with pytest.raises(KeyError, match="does not exist in the model state dict"):
            merge_adapter_weights(model, adapter_weights, {"lora_alpha": 1, "r": 1})

    def test_gpt_oss_attn_remapping(self):
        state_dict = {"model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 4)}
        model = _make_base_model(state_dict, class_name="GptOssForCausalLM")

        # Tinker adapter uses .attn instead of .self_attn for GPT-OSS
        adapter_weights = {
            "base_model.model.model.layers.0.attn.q_proj.lora_A.weight": torch.ones(1, 4),
            "base_model.model.model.layers.0.attn.q_proj.lora_B.weight": torch.ones(8, 1),
        }

        merge_adapter_weights(model, adapter_weights, {"lora_alpha": 1, "r": 1})
        assert state_dict["model.layers.0.self_attn.q_proj.weight"].abs().sum() > 0

    def test_vision_model_prefix_remapping(self):
        state_dict = {"model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 4)}
        model = _make_base_model(state_dict)

        adapter_weights = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1, 4),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(8, 1),
        }

        merge_adapter_weights(model, adapter_weights, {"lora_alpha": 1, "r": 1})
        assert state_dict["model.language_model.layers.0.self_attn.q_proj.weight"].abs().sum() > 0


# ---------------------------------------------------------------------------
# Separate per-expert weights (Qwen3 MoE, DeepSeek, Kimi)
# ---------------------------------------------------------------------------


class TestSeparateExpertMerge:
    def test_per_expert_merge(self):
        num_experts = 2
        state_dict = {
            f"model.layers.0.mlp.experts.{i}.gate_proj.weight": torch.zeros(8, 4)
            for i in range(num_experts)
        }
        model = _make_base_model(state_dict)

        gate_A, gate_B = _make_expert_lora_pair(num_experts, 8, 4, fill=0.1)
        adapter_weights = {
            "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight": gate_A,
            "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight": gate_B,
        }

        merge_adapter_weights(model, adapter_weights, {"lora_alpha": 1, "r": 1})

        for i in range(num_experts):
            w = state_dict[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"]
            assert w.abs().sum() > 0, f"Expert {i} was not updated"

    def test_shared_lora_a_broadcast(self):
        """lora_A has 1 expert, lora_B has N — A should be broadcast."""
        num_experts = 3
        state_dict = {
            f"model.layers.0.mlp.experts.{i}.gate_proj.weight": torch.zeros(8, 4)
            for i in range(num_experts)
        }
        model = _make_base_model(state_dict)

        lora_A = torch.ones(1, 1, 4) * 0.5  # shared across experts
        lora_B = torch.ones(num_experts, 8, 1)
        adapter_weights = {
            "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight": lora_A,
            "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight": lora_B,
        }

        merge_adapter_weights(model, adapter_weights, {"lora_alpha": 1, "r": 1})

        for i in range(num_experts):
            assert state_dict[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"].abs().sum() > 0


# ---------------------------------------------------------------------------
# Fused expert weights — interleaved (GPT-OSS)
# ---------------------------------------------------------------------------


class TestFusedInterleavedMerge:
    """GPT-OSS: gate_up_proj uses [g0, u0, g1, u1, ...] layout."""

    NUM_EXPERTS = 2
    IN_DIM = 4
    OUT_DIM = 4
    FUSED_DIM = OUT_DIM * 2

    def _make_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "model.layers.0.mlp.experts.gate_up_proj": torch.zeros(
                self.NUM_EXPERTS, self.IN_DIM, self.FUSED_DIM
            ),
            "model.layers.0.mlp.experts.down_proj": torch.zeros(
                self.NUM_EXPERTS, self.OUT_DIM, self.IN_DIM
            ),
        }

    def _make_adapter(self, gate_fill: float, up_fill: float) -> dict[str, torch.Tensor]:
        prefix = "base_model.model.model.layers.0.mlp.experts"
        gate_A, gate_B = _make_expert_lora_pair(
            self.NUM_EXPERTS, self.OUT_DIM, self.IN_DIM, fill=gate_fill
        )
        up_A, up_B = _make_expert_lora_pair(
            self.NUM_EXPERTS, self.OUT_DIM, self.IN_DIM, fill=up_fill
        )
        return {
            f"{prefix}.w1.lora_A.weight": gate_A,
            f"{prefix}.w1.lora_B.weight": gate_B,
            f"{prefix}.w3.lora_A.weight": up_A,
            f"{prefix}.w3.lora_B.weight": up_B,
        }

    def test_gate_and_up_in_correct_slots(self):
        state_dict = self._make_state_dict()
        model = _make_base_model(state_dict, class_name="GptOssModel")
        adapter = self._make_adapter(gate_fill=0.01, up_fill=0.05)

        merge_adapter_weights(model, adapter, {"lora_alpha": 1, "r": 1})

        fused = state_dict["model.layers.0.mlp.experts.gate_up_proj"]
        gate_slots = fused[:, :, 0::2]
        up_slots = fused[:, :, 1::2]

        assert torch.allclose(gate_slots, torch.full_like(gate_slots, 0.01), atol=1e-6)
        assert torch.allclose(up_slots, torch.full_like(up_slots, 0.05), atol=1e-6)

    def test_up_does_not_leak_into_gate(self):
        state_dict = self._make_state_dict()
        model = _make_base_model(state_dict, class_name="GptOssModel")

        prefix = "base_model.model.model.layers.0.mlp.experts"
        up_A, up_B = _make_expert_lora_pair(self.NUM_EXPERTS, self.OUT_DIM, self.IN_DIM, fill=0.1)
        adapter = {
            f"{prefix}.w3.lora_A.weight": up_A,
            f"{prefix}.w3.lora_B.weight": up_B,
        }

        merge_adapter_weights(model, adapter, {"lora_alpha": 1, "r": 1})

        fused = state_dict["model.layers.0.mlp.experts.gate_up_proj"]
        assert fused[:, :, 0::2].abs().max() == 0.0, "up delta leaked into gate slots"
        assert fused[:, :, 1::2].abs().sum() > 0


# ---------------------------------------------------------------------------
# Fused expert weights — concatenated (Qwen3.5, Qwen3-VL)
# ---------------------------------------------------------------------------


class TestFusedConcatenatedMerge:
    """Non-GPT-OSS fused: gate_up_proj uses [gate | up] layout."""

    NUM_EXPERTS = 2
    IN_DIM = 4
    OUT_DIM = 4
    FUSED_DIM = OUT_DIM * 2

    def _make_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "model.layers.0.mlp.experts.gate_up_proj": torch.zeros(
                self.NUM_EXPERTS, self.IN_DIM, self.FUSED_DIM
            ),
            "model.layers.0.mlp.experts.down_proj": torch.zeros(
                self.NUM_EXPERTS, self.OUT_DIM, self.IN_DIM
            ),
        }

    def _make_adapter(self, gate_fill: float, up_fill: float) -> dict[str, torch.Tensor]:
        prefix = "base_model.model.model.layers.0.mlp.experts"
        gate_A, gate_B = _make_expert_lora_pair(
            self.NUM_EXPERTS, self.OUT_DIM, self.IN_DIM, fill=gate_fill
        )
        up_A, up_B = _make_expert_lora_pair(
            self.NUM_EXPERTS, self.OUT_DIM, self.IN_DIM, fill=up_fill
        )
        return {
            f"{prefix}.w1.lora_A.weight": gate_A,
            f"{prefix}.w1.lora_B.weight": gate_B,
            f"{prefix}.w3.lora_A.weight": up_A,
            f"{prefix}.w3.lora_B.weight": up_B,
        }

    def test_gate_and_up_in_correct_halves(self):
        state_dict = self._make_state_dict()
        model = _make_base_model(state_dict, class_name="QwenModel")
        adapter = self._make_adapter(gate_fill=0.02, up_fill=0.07)

        merge_adapter_weights(model, adapter, {"lora_alpha": 1, "r": 1})

        fused = state_dict["model.layers.0.mlp.experts.gate_up_proj"]
        sz = self.FUSED_DIM // 2
        gate_half = fused[:, :, :sz]
        up_half = fused[:, :, sz:]

        assert torch.allclose(gate_half, torch.full_like(gate_half, 0.02), atol=1e-6)
        assert torch.allclose(up_half, torch.full_like(up_half, 0.07), atol=1e-6)

    def test_up_does_not_leak_into_gate(self):
        state_dict = self._make_state_dict()
        model = _make_base_model(state_dict, class_name="QwenModel")

        prefix = "base_model.model.model.layers.0.mlp.experts"
        up_A, up_B = _make_expert_lora_pair(self.NUM_EXPERTS, self.OUT_DIM, self.IN_DIM, fill=0.1)
        adapter = {
            f"{prefix}.w3.lora_A.weight": up_A,
            f"{prefix}.w3.lora_B.weight": up_B,
        }

        merge_adapter_weights(model, adapter, {"lora_alpha": 1, "r": 1})

        fused = state_dict["model.layers.0.mlp.experts.gate_up_proj"]
        sz = self.FUSED_DIM // 2
        assert fused[:, :, :sz].abs().max() == 0.0, "up delta leaked into gate half"
        assert fused[:, :, sz:].abs().sum() > 0


# ---------------------------------------------------------------------------
# Error cases for expert LoRA
# ---------------------------------------------------------------------------


class TestExpertErrorCases:
    def test_non_3d_expert_lora_raises(self):
        state_dict = {"model.layers.0.mlp.experts.0.gate_proj.weight": torch.zeros(8, 4)}
        model = _make_base_model(state_dict)

        adapter_weights = {
            "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight": torch.ones(1, 4),  # 2D!
            "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight": torch.ones(8, 1),  # 2D!
        }
        with pytest.raises(ValueError, match="must be 3D"):
            merge_adapter_weights(model, adapter_weights, {"lora_alpha": 1, "r": 1})


# ===========================================================================
# Tests for new APIs: MergeProfile, detect_merge_profile, plan/apply,
# merge_lora_matrices, expand_expert_lora_tensors
# ===========================================================================

# ---------------------------------------------------------------------------
# merge_lora_matrices
# ---------------------------------------------------------------------------


class TestMergeLoraMatrices:
    def test_basic_multiplication(self):
        lora_A = torch.ones(1, 4)  # (rank=1, in_dim=4)
        lora_B = torch.ones(8, 1)  # (out_dim=8, rank=1)
        result = merge_lora_matrices(lora_A, lora_B)
        assert result.shape == (8, 4)
        assert torch.allclose(result, torch.ones(8, 4))

    def test_with_scaling(self):
        lora_A = torch.ones(2, 3) * 0.5
        lora_B = torch.eye(3, 2)  # (3, 2)
        result = merge_lora_matrices(lora_A, lora_B)
        # eye(3,2) @ (ones(2,3) * 0.5) = first 2 rows are 0.5, last row is 0
        assert result.shape == (3, 3)
        assert torch.allclose(result[:2], torch.full((2, 3), 0.5))
        assert torch.allclose(result[2], torch.zeros(3))


# ---------------------------------------------------------------------------
# expand_expert_lora_tensors
# ---------------------------------------------------------------------------


class TestExpandExpertLoraTensors:
    def test_expand_A_to_match_B(self):
        lora_A = torch.ones(1, 2, 4)
        lora_B = torch.ones(3, 8, 2)
        out_A, out_B = expand_expert_lora_tensors(lora_A, lora_B)
        assert out_A.shape[0] == 3
        assert out_B is lora_B

    def test_expand_B_to_match_A(self):
        lora_A = torch.ones(3, 2, 4)
        lora_B = torch.ones(1, 8, 2)
        out_A, out_B = expand_expert_lora_tensors(lora_A, lora_B)
        assert out_A is lora_A
        assert out_B.shape[0] == 3

    def test_both_single_raises(self):
        with pytest.raises(ValueError, match="both A and B have 1 expert"):
            expand_expert_lora_tensors(torch.ones(1, 2, 4), torch.ones(1, 8, 2))

    def test_already_matched_is_noop(self):
        lora_A = torch.ones(4, 2, 4)
        lora_B = torch.ones(4, 8, 2)
        out_A, out_B = expand_expert_lora_tensors(lora_A, lora_B)
        assert out_A is lora_A
        assert out_B is lora_B

    def test_mismatched_expert_counts_raises(self):
        with pytest.raises(ValueError, match="Expert count mismatch"):
            expand_expert_lora_tensors(torch.ones(3, 2, 4), torch.ones(5, 8, 2))


# ---------------------------------------------------------------------------
# detect_merge_profile
# ---------------------------------------------------------------------------


class TestDetectMergeProfile:
    def test_standard_model(self):
        config: dict = {"architectures": ["QwenForCausalLM"]}
        keys = {"model.layers.0.self_attn.q_proj.weight"}
        profile = detect_merge_profile(config, keys)
        assert profile.expert_layout == "separate"
        assert profile.extra_key_remaps == ()
        assert profile.has_language_model_prefix is False

    def test_gpt_oss_detection(self):
        config: dict = {"architectures": ["GptOssForCausalLM"]}
        keys = {
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.experts.gate_up_proj",
        }
        profile = detect_merge_profile(config, keys)
        assert profile.expert_layout == "fused_interleaved"
        assert (".attn", ".self_attn") in profile.extra_key_remaps

    def test_fused_concatenated_non_gpt_oss(self):
        config: dict = {"architectures": ["Qwen3ForCausalLM"]}
        keys = {"model.layers.0.mlp.experts.gate_up_proj"}
        profile = detect_merge_profile(config, keys)
        assert profile.expert_layout == "fused_concatenated"
        assert profile.extra_key_remaps == ()

    def test_vision_model_prefix(self):
        config: dict = {"architectures": ["Qwen3VLForConditionalGeneration"]}
        keys = {"model.language_model.layers.0.self_attn.q_proj.weight"}
        profile = detect_merge_profile(config, keys)
        assert profile.has_language_model_prefix is True

    def test_separate_experts_without_fused(self):
        config: dict = {"architectures": ["QwenMoEForCausalLM"]}
        keys = {"model.layers.0.mlp.experts.0.gate_proj.weight"}
        profile = detect_merge_profile(config, keys)
        assert profile.expert_layout == "separate"

    def test_empty_architectures(self):
        profile = detect_merge_profile({}, {"model.layers.0.weight"})
        assert profile.expert_layout == "separate"
        assert profile.extra_key_remaps == ()


# ---------------------------------------------------------------------------
# plan_merge_ops
# ---------------------------------------------------------------------------


class TestPlanMergeOps:
    def test_non_expert_produces_2d_op(self):
        keys = {"model.layers.0.self_attn.q_proj.weight"}
        profile = MergeProfile()
        adapter = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1, 4),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(8, 1),
        }
        ops = plan_merge_ops(adapter, {"lora_alpha": 2, "r": 1}, keys, profile)
        assert "model.layers.0.self_attn.q_proj.weight" in ops
        op = ops["model.layers.0.self_attn.q_proj.weight"][0]
        assert op.lora_A.ndim == 2
        assert op.is_expert_3d is False
        # B should be pre-scaled by alpha/r = 2
        assert torch.allclose(op.lora_B, torch.ones(8, 1) * 2)

    def test_separate_experts_produce_per_expert_2d_ops(self):
        keys = {
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.1.gate_proj.weight",
        }
        profile = MergeProfile(expert_layout="separate")
        adapter = {
            "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight": torch.ones(2, 1, 4),
            "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight": torch.ones(2, 8, 1),
        }
        ops = plan_merge_ops(adapter, {"lora_alpha": 1, "r": 1}, keys, profile)
        # Should have ops for both experts
        assert "model.layers.0.mlp.experts.0.gate_proj.weight" in ops
        assert "model.layers.0.mlp.experts.1.gate_proj.weight" in ops
        # Each op should have 2D tensors
        for key in ops:
            assert ops[key][0].lora_A.ndim == 2

    def test_fused_experts_produce_3d_ops(self):
        keys = {"model.layers.0.mlp.experts.gate_up_proj"}
        profile = MergeProfile(expert_layout="fused_concatenated")
        adapter = {
            "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight": torch.ones(2, 1, 4),
            "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight": torch.ones(2, 8, 1),
        }
        ops = plan_merge_ops(adapter, {"lora_alpha": 1, "r": 1}, keys, profile)
        assert "model.layers.0.mlp.experts.gate_up_proj" in ops
        op = ops["model.layers.0.mlp.experts.gate_up_proj"][0]
        assert op.is_expert_3d is True
        assert op.fused_proj_idx == 0  # gate = w1
        assert op.fused_proj_interleaved is False

    def test_fused_interleaved_sets_flag(self):
        keys = {"model.layers.0.mlp.experts.gate_up_proj"}
        profile = MergeProfile(expert_layout="fused_interleaved")
        adapter = {
            "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight": torch.ones(2, 1, 4),
            "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight": torch.ones(2, 8, 1),
        }
        ops = plan_merge_ops(adapter, {"lora_alpha": 1, "r": 1}, keys, profile)
        op = ops["model.layers.0.mlp.experts.gate_up_proj"][0]
        assert op.fused_proj_interleaved is True

    def test_extra_key_remaps_applied(self):
        keys = {"model.layers.0.self_attn.q_proj.weight"}
        profile = MergeProfile(extra_key_remaps=((".attn", ".self_attn"),))
        adapter = {
            "base_model.model.model.layers.0.attn.q_proj.lora_A.weight": torch.ones(1, 4),
            "base_model.model.model.layers.0.attn.q_proj.lora_B.weight": torch.ones(8, 1),
        }
        ops = plan_merge_ops(adapter, {"lora_alpha": 1, "r": 1}, keys, profile)
        assert "model.layers.0.self_attn.q_proj.weight" in ops

    def test_vision_prefix_remapping(self):
        keys = {"model.language_model.layers.0.self_attn.q_proj.weight"}
        profile = MergeProfile(has_language_model_prefix=True)
        adapter = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1, 4),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(8, 1),
        }
        ops = plan_merge_ops(adapter, {"lora_alpha": 1, "r": 1}, keys, profile)
        assert "model.language_model.layers.0.self_attn.q_proj.weight" in ops

    def test_unembed_tokens_remapped_to_lm_head(self):
        keys = {"lm_head.weight"}
        profile = MergeProfile()
        adapter = {
            "base_model.model.model.unembed_tokens.lora_A.weight": torch.ones(1, 4),
            "base_model.model.model.unembed_tokens.lora_B.weight": torch.ones(8, 1),
        }
        ops = plan_merge_ops(adapter, {"lora_alpha": 1, "r": 1}, keys, profile)
        assert "lm_head.weight" in ops

    def test_missing_key_raises(self):
        keys = {"some.other.weight"}
        profile = MergeProfile()
        adapter = {
            "base_model.model.model.layers.0.q_proj.lora_A.weight": torch.ones(1, 4),
            "base_model.model.model.layers.0.q_proj.lora_B.weight": torch.ones(4, 1),
        }
        with pytest.raises(KeyError, match="does not exist"):
            plan_merge_ops(adapter, {"lora_alpha": 1, "r": 1}, keys, profile)

    def test_missing_config_key_raises(self):
        with pytest.raises(KeyError, match="lora_alpha"):
            plan_merge_ops({}, {"r": 1}, set(), MergeProfile())

    def test_invalid_expert_layout_raises(self):
        with pytest.raises(ValueError, match="Invalid expert_layout"):
            plan_merge_ops({}, {"lora_alpha": 1, "r": 1}, set(), MergeProfile(expert_layout="bad"))


# ---------------------------------------------------------------------------
# apply_merge_op
# ---------------------------------------------------------------------------


class TestApplyMergeOp:
    def test_2d_standard_merge(self):
        tensors = {"q_proj.weight": torch.zeros(8, 4)}
        op = MergeOp(
            target_key="q_proj.weight",
            lora_A=torch.ones(1, 4),
            lora_B=torch.ones(8, 1),
        )
        apply_merge_op(tensors, op)
        assert tensors["q_proj.weight"].abs().sum() > 0
        # delta = B @ A = ones(8,1) @ ones(1,4) = ones(8,4)
        assert torch.allclose(tensors["q_proj.weight"], torch.ones(8, 4))

    def test_3d_fused_concatenated_gate(self):
        n_exp, in_dim, fused_dim = 2, 4, 8
        tensors = {"gate_up_proj": torch.zeros(n_exp, in_dim, fused_dim)}
        # Gate op (fused_proj_idx=0): delta goes into first half
        lora_A = torch.ones(n_exp, 1, in_dim) * 0.1
        lora_B = torch.ones(n_exp, fused_dim // 2, 1)
        op = MergeOp(
            target_key="gate_up_proj",
            lora_A=lora_A,
            lora_B=lora_B,
            is_expert_3d=True,
            fused_proj_idx=0,
            fused_proj_interleaved=False,
        )
        apply_merge_op(tensors, op)
        gate_half = tensors["gate_up_proj"][:, :, : fused_dim // 2]
        up_half = tensors["gate_up_proj"][:, :, fused_dim // 2 :]
        assert gate_half.abs().sum() > 0
        assert up_half.abs().sum() == 0

    def test_3d_fused_interleaved_up(self):
        n_exp, in_dim, fused_dim = 2, 4, 8
        tensors = {"gate_up_proj": torch.zeros(n_exp, in_dim, fused_dim)}
        # Up op (fused_proj_idx=1, interleaved): delta goes into odd columns
        lora_A = torch.ones(n_exp, 1, in_dim) * 0.2
        lora_B = torch.ones(n_exp, fused_dim // 2, 1)
        op = MergeOp(
            target_key="gate_up_proj",
            lora_A=lora_A,
            lora_B=lora_B,
            is_expert_3d=True,
            fused_proj_idx=1,
            fused_proj_interleaved=True,
        )
        apply_merge_op(tensors, op)
        gate_slots = tensors["gate_up_proj"][:, :, 0::2]
        up_slots = tensors["gate_up_proj"][:, :, 1::2]
        assert gate_slots.abs().sum() == 0
        assert up_slots.abs().sum() > 0

    def test_shape_mismatch_raises(self):
        tensors = {"weight": torch.zeros(4, 4)}
        op = MergeOp(
            target_key="weight",
            lora_A=torch.ones(1, 8),  # wrong in_dim
            lora_B=torch.ones(4, 1),
        )
        with pytest.raises(ValueError, match="Shape mismatch"):
            apply_merge_op(tensors, op)
