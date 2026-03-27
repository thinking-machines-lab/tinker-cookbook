"""Unit tests for LoRA merge logic.

Uses synthetic tensors to cover all code paths without needing real models
or network access.
"""

from typing import Any

import pytest
import torch

from tinker_cookbook.exceptions import WeightsMergeError
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
    validate_merge_op_shapes,
)
from tinker_cookbook.weights._merge_deepseek import detect_profile as detect_deepseek_profile
from tinker_cookbook.weights._merge_default import detect_profile as detect_default_profile
from tinker_cookbook.weights._merge_gpt_oss import detect_profile as detect_gpt_oss_profile
from tinker_cookbook.weights._merge_nemotron import detect_profile as detect_nemotron_profile
from tinker_cookbook.weights._merge_qwen3_5 import detect_profile as detect_qwen3_5_profile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_base_model(
    state_dict: dict[str, torch.Tensor],
    class_name: str = "SomeModel",
    config_dict: dict | None = None,
) -> Any:
    """Create a minimal mock model with a real state_dict and controllable class name.

    Uses a dynamically-created class so ``str(type(model))`` contains the
    desired class name (important for GPT-OSS detection).

    Args:
        config_dict: If provided, attaches a mock ``config`` with a ``to_dict()``
            method. This is used by ``merge_adapter_weights`` for profile detection.
    """

    class _Config:
        def __init__(self, d: dict) -> None:
            self._d = d

        def to_dict(self) -> dict:
            return self._d

    attrs: dict[str, Any] = {"state_dict": lambda self: state_dict}
    if config_dict is not None:
        attrs["config"] = _Config(config_dict)
    cls = type(class_name, (), attrs)
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
        with pytest.raises(WeightsMergeError, match="Shape mismatch"):
            apply_merged_weight(torch.zeros(3, 4), torch.zeros(3, 5))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_missing_lora_alpha(self):
        model = _make_base_model({})
        with pytest.raises(WeightsMergeError, match="lora_alpha"):
            merge_adapter_weights(model, {}, {"r": 1})

    def test_missing_r(self):
        model = _make_base_model({})
        with pytest.raises(WeightsMergeError, match="'r'"):
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
        with pytest.raises(WeightsMergeError, match="does not exist in the model state dict"):
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

    def test_broadcast_lora_a_interleaved(self):
        """Real Tinker pattern: lora_A shared (1, rank, dim), lora_B per-expert."""
        state_dict = self._make_state_dict()
        model = _make_base_model(state_dict, class_name="GptOssModel")

        prefix = "base_model.model.model.layers.0.mlp.experts"
        rank = 1
        adapter = {
            f"{prefix}.w1.lora_A.weight": torch.ones(1, rank, self.IN_DIM) * 0.01,
            f"{prefix}.w1.lora_B.weight": torch.ones(self.NUM_EXPERTS, self.OUT_DIM, rank),
            f"{prefix}.w3.lora_A.weight": torch.ones(1, rank, self.IN_DIM) * 0.05,
            f"{prefix}.w3.lora_B.weight": torch.ones(self.NUM_EXPERTS, self.OUT_DIM, rank),
        }

        merge_adapter_weights(model, adapter, {"lora_alpha": 1, "r": rank})

        fused = state_dict["model.layers.0.mlp.experts.gate_up_proj"]
        assert torch.allclose(
            fused[:, :, 0::2], torch.full_like(fused[:, :, 0::2], 0.01), atol=1e-6
        )
        assert torch.allclose(
            fused[:, :, 1::2], torch.full_like(fused[:, :, 1::2], 0.05), atol=1e-6
        )


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

    def test_broadcast_lora_a_concatenated(self):
        """Real Tinker pattern: lora_A shared (1, rank, dim), lora_B per-expert."""
        state_dict = self._make_state_dict()
        model = _make_base_model(state_dict, class_name="QwenModel")

        prefix = "base_model.model.model.layers.0.mlp.experts"
        rank = 1
        adapter = {
            f"{prefix}.w1.lora_A.weight": torch.ones(1, rank, self.IN_DIM) * 0.02,
            f"{prefix}.w1.lora_B.weight": torch.ones(self.NUM_EXPERTS, self.OUT_DIM, rank),
            f"{prefix}.w3.lora_A.weight": torch.ones(1, rank, self.IN_DIM) * 0.07,
            f"{prefix}.w3.lora_B.weight": torch.ones(self.NUM_EXPERTS, self.OUT_DIM, rank),
        }

        merge_adapter_weights(model, adapter, {"lora_alpha": 1, "r": rank})

        fused = state_dict["model.layers.0.mlp.experts.gate_up_proj"]
        sz = self.FUSED_DIM // 2
        assert torch.allclose(fused[:, :, :sz], torch.full_like(fused[:, :, :sz], 0.02), atol=1e-6)
        assert torch.allclose(fused[:, :, sz:], torch.full_like(fused[:, :, sz:], 0.07), atol=1e-6)


# ---------------------------------------------------------------------------
# Fused expert weights — concatenated, transposed (Qwen3.5 MoE)
# ---------------------------------------------------------------------------


class TestFusedConcatenatedTransposedMerge:
    """Qwen3.5 MoE: gate_up_proj = (n_exp, fused, hidden) with fused on dim 1.

    Uses asymmetric dims (HIDDEN != FUSED_DIM) so _detect_fused_axis can
    distinguish which axis is fused.
    """

    NUM_EXPERTS = 2
    HIDDEN = 8  # hidden_size
    INTERMEDIATE = 3  # moe_intermediate_size (asymmetric vs HIDDEN)
    FUSED_DIM = INTERMEDIATE * 2  # gate + up = 6

    def _make_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            # Qwen3.5 layout: (num_experts, fused_dim, hidden_size)
            "model.layers.0.mlp.experts.gate_up_proj": torch.zeros(
                self.NUM_EXPERTS, self.FUSED_DIM, self.HIDDEN
            ),
            "model.layers.0.mlp.experts.down_proj": torch.zeros(
                self.NUM_EXPERTS, self.HIDDEN, self.INTERMEDIATE
            ),
        }

    def _make_adapter(self, gate_fill: float, up_fill: float) -> dict[str, torch.Tensor]:
        prefix = "base_model.model.model.layers.0.mlp.experts"
        # LoRA for gate_proj: maps hidden→intermediate
        gate_A, gate_B = _make_expert_lora_pair(
            self.NUM_EXPERTS, self.INTERMEDIATE, self.HIDDEN, fill=gate_fill
        )
        up_A, up_B = _make_expert_lora_pair(
            self.NUM_EXPERTS, self.INTERMEDIATE, self.HIDDEN, fill=up_fill
        )
        return {
            f"{prefix}.w1.lora_A.weight": gate_A,
            f"{prefix}.w1.lora_B.weight": gate_B,
            f"{prefix}.w3.lora_A.weight": up_A,
            f"{prefix}.w3.lora_B.weight": up_B,
        }

    def test_gate_and_up_in_correct_halves(self):
        """Gate delta goes to dim1[:intermediate], up to dim1[intermediate:]."""
        state_dict = self._make_state_dict()
        model = _make_base_model(state_dict, class_name="QwenModel")
        adapter = self._make_adapter(gate_fill=0.02, up_fill=0.07)

        merge_adapter_weights(model, adapter, {"lora_alpha": 1, "r": 1})

        fused = state_dict["model.layers.0.mlp.experts.gate_up_proj"]
        gate_half = fused[:, : self.INTERMEDIATE, :]
        up_half = fused[:, self.INTERMEDIATE :, :]

        assert torch.allclose(gate_half, torch.full_like(gate_half, 0.02), atol=1e-6)
        assert torch.allclose(up_half, torch.full_like(up_half, 0.07), atol=1e-6)

    def test_up_does_not_leak_into_gate(self):
        """Up-only adapter should not touch the gate half (dim 1)."""
        state_dict = self._make_state_dict()
        model = _make_base_model(state_dict, class_name="QwenModel")

        prefix = "base_model.model.model.layers.0.mlp.experts"
        up_A, up_B = _make_expert_lora_pair(
            self.NUM_EXPERTS, self.INTERMEDIATE, self.HIDDEN, fill=0.1
        )
        adapter = {
            f"{prefix}.w3.lora_A.weight": up_A,
            f"{prefix}.w3.lora_B.weight": up_B,
        }

        merge_adapter_weights(model, adapter, {"lora_alpha": 1, "r": 1})

        fused = state_dict["model.layers.0.mlp.experts.gate_up_proj"]
        assert fused[:, : self.INTERMEDIATE, :].abs().max() == 0.0, "up delta leaked into gate half"
        assert fused[:, self.INTERMEDIATE :, :].abs().sum() > 0

    def test_gate_only_does_not_leak_into_up(self):
        """Gate-only adapter should not touch the up half (dim 1)."""
        state_dict = self._make_state_dict()
        model = _make_base_model(state_dict, class_name="QwenModel")

        prefix = "base_model.model.model.layers.0.mlp.experts"
        gate_A, gate_B = _make_expert_lora_pair(
            self.NUM_EXPERTS, self.INTERMEDIATE, self.HIDDEN, fill=0.05
        )
        adapter = {
            f"{prefix}.w1.lora_A.weight": gate_A,
            f"{prefix}.w1.lora_B.weight": gate_B,
        }

        merge_adapter_weights(model, adapter, {"lora_alpha": 1, "r": 1})

        fused = state_dict["model.layers.0.mlp.experts.gate_up_proj"]
        assert fused[:, : self.INTERMEDIATE, :].abs().sum() > 0
        assert fused[:, self.INTERMEDIATE :, :].abs().max() == 0.0, "gate delta leaked into up half"

    def test_broadcast_lora_a_transposed(self):
        """Real Tinker pattern: lora_A shared (1, rank, dim), lora_B per-expert."""
        state_dict = self._make_state_dict()
        model = _make_base_model(state_dict, class_name="QwenModel")

        prefix = "base_model.model.model.layers.0.mlp.experts"
        rank = 1
        adapter = {
            f"{prefix}.w1.lora_A.weight": torch.ones(1, rank, self.HIDDEN) * 0.02,
            f"{prefix}.w1.lora_B.weight": torch.ones(self.NUM_EXPERTS, self.INTERMEDIATE, rank),
            f"{prefix}.w3.lora_A.weight": torch.ones(1, rank, self.HIDDEN) * 0.07,
            f"{prefix}.w3.lora_B.weight": torch.ones(self.NUM_EXPERTS, self.INTERMEDIATE, rank),
        }

        merge_adapter_weights(model, adapter, {"lora_alpha": 1, "r": rank})

        fused = state_dict["model.layers.0.mlp.experts.gate_up_proj"]
        gate_half = fused[:, : self.INTERMEDIATE, :]
        up_half = fused[:, self.INTERMEDIATE :, :]
        assert torch.allclose(gate_half, torch.full_like(gate_half, 0.02), atol=1e-6)
        assert torch.allclose(up_half, torch.full_like(up_half, 0.07), atol=1e-6)


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
        with pytest.raises(WeightsMergeError, match="must be 3D"):
            merge_adapter_weights(model, adapter_weights, {"lora_alpha": 1, "r": 1})


# ---------------------------------------------------------------------------
# Tests: Nemotron detection and empty expert handling
# ---------------------------------------------------------------------------


class TestNemotronProfile:
    def test_detect_nemotron_profile(self):
        config = {"architectures": ["NemotronHForCausalLM"]}
        keys: set[str] = {"backbone.layers.0.mixer.q_proj.weight"}
        profile = detect_nemotron_profile(config, keys)
        assert profile is not None
        assert profile.model_family == "nemotron"
        assert profile.expert_layout == "separate"
        # Nemotron maps w1→up_proj, w2→down_proj (no gate_proj)
        assert (".w1.weight", ".up_proj.weight") in profile.expert_key_remaps
        assert (".w2.weight", ".down_proj.weight") in profile.expert_key_remaps
        assert not any("gate_proj" in new for _, new in profile.expert_key_remaps)

    def test_non_nemotron_not_detected(self):
        config = {"architectures": ["LlamaForCausalLM"]}
        keys: set[str] = {"model.layers.0.self_attn.q_proj.weight"}
        assert detect_nemotron_profile(config, keys) is None

    def test_detect_merge_profile_dispatches_to_nemotron(self):
        """detect_merge_profile should pick up Nemotron via the registered detector."""
        config = {"architectures": ["NemotronHForCausalLM"]}
        keys: set[str] = {"backbone.layers.0.mixer.q_proj.weight"}
        profile = detect_merge_profile(config, keys)
        assert profile.model_family == "nemotron"


class TestEmptyExpertMerge:
    def test_empty_expert_tensors_skipped_in_merge(self):
        """Empty expert LoRA tensors (Nemotron w3) should be silently skipped."""
        num_experts = 2
        state_dict: dict[str, torch.Tensor] = {}
        for e in range(num_experts):
            state_dict[f"backbone.layers.0.mixer.experts.{e}.up_proj.weight"] = torch.zeros(8, 4)
            state_dict[f"backbone.layers.0.mixer.experts.{e}.down_proj.weight"] = torch.zeros(4, 8)

        model = _make_base_model(
            state_dict,
            config_dict={"architectures": ["NemotronHForCausalLM"]},
        )

        adapter_weights = {
            # w1 (up_proj) — real 3D expert tensors
            "base_model.model.backbone.layers.0.mixer.experts.w1.lora_A.weight": torch.ones(
                num_experts, 1, 4
            ),
            "base_model.model.backbone.layers.0.mixer.experts.w1.lora_B.weight": torch.ones(
                num_experts, 8, 1
            ),
            # w2 (down_proj) — real 3D expert tensors
            "base_model.model.backbone.layers.0.mixer.experts.w2.lora_A.weight": torch.ones(
                num_experts, 1, 8
            ),
            "base_model.model.backbone.layers.0.mixer.experts.w2.lora_B.weight": torch.ones(
                num_experts, 4, 1
            ),
            # w3 — empty placeholders (no gate_proj in Nemotron)
            "base_model.model.backbone.layers.0.mixer.experts.w3.lora_A.weight": torch.empty(0),
            "base_model.model.backbone.layers.0.mixer.experts.w3.lora_B.weight": torch.empty(0),
        }

        # Should not raise — empty tensors should be skipped.
        merge_adapter_weights(model, adapter_weights, {"lora_alpha": 1, "r": 1})

        # Verify the non-empty expert weights were actually merged.
        for e in range(num_experts):
            up_weight = model.state_dict()[f"backbone.layers.0.mixer.experts.{e}.up_proj.weight"]
            assert up_weight.abs().sum() > 0, f"Expert {e} up_proj should have been merged"


class TestNemotronFusedProjectionMerge:
    """Test Mamba gate_proj/x_proj → fused in_proj merge via slice_start."""

    HIDDEN = 8
    INTERMEDIATE = 12
    # in_proj = [gate (INTERMEDIATE) | x (INTERMEDIATE) | B+C+dt (6)]
    IN_PROJ_DIM = INTERMEDIATE * 2 + 6

    def _make_model_and_adapter(self):
        state_dict = {
            "backbone.layers.0.mixer.in_proj.weight": torch.zeros(self.IN_PROJ_DIM, self.HIDDEN),
            "backbone.layers.0.mixer.out_proj.weight": torch.zeros(self.HIDDEN, self.INTERMEDIATE),
        }
        model = _make_base_model(
            state_dict,
            config_dict={"architectures": ["NemotronHForCausalLM"]},
        )
        adapter_weights = {
            "base_model.model.backbone.layers.0.mixer.gate_proj.lora_A.weight": (
                torch.ones(1, self.HIDDEN) * 0.5
            ),
            "base_model.model.backbone.layers.0.mixer.gate_proj.lora_B.weight": (
                torch.ones(self.INTERMEDIATE, 1)
            ),
            "base_model.model.backbone.layers.0.mixer.x_proj.lora_A.weight": (
                torch.ones(1, self.HIDDEN) * 0.3
            ),
            "base_model.model.backbone.layers.0.mixer.x_proj.lora_B.weight": (
                torch.ones(self.INTERMEDIATE, 1)
            ),
        }
        return model, adapter_weights

    def test_gate_and_x_merged_into_in_proj(self):
        model, adapter_weights = self._make_model_and_adapter()
        merge_adapter_weights(model, adapter_weights, {"lora_alpha": 1, "r": 1})

        in_proj = model.state_dict()["backbone.layers.0.mixer.in_proj.weight"]
        gate_slice = in_proj[: self.INTERMEDIATE]
        x_slice = in_proj[self.INTERMEDIATE : 2 * self.INTERMEDIATE]
        rest = in_proj[2 * self.INTERMEDIATE :]

        # gate delta = lora_B @ lora_A = ones(12,1) @ (ones(1,8)*0.5) = 0.5
        assert torch.allclose(gate_slice, torch.full_like(gate_slice, 0.5), atol=1e-6)
        # x delta = ones(12,1) @ (ones(1,8)*0.3) = 0.3
        assert torch.allclose(x_slice, torch.full_like(x_slice, 0.3), atol=1e-6)
        # B/C/dt rows should be unchanged (zeros)
        assert rest.abs().sum() == 0

    def test_out_proj_unchanged(self):
        model, adapter_weights = self._make_model_and_adapter()
        merge_adapter_weights(model, adapter_weights, {"lora_alpha": 1, "r": 1})

        out_proj = model.state_dict()["backbone.layers.0.mixer.out_proj.weight"]
        assert out_proj.abs().sum() == 0, "out_proj should not be affected"


class TestNemotronPartialLoraMerge:
    """Test merge with partial LoRA coverage (train_attn/train_mlp flags)."""

    HIDDEN = 8
    INTERMEDIATE = 12
    IN_PROJ_DIM = INTERMEDIATE * 2 + 6
    NUM_EXPERTS = 2

    def test_attn_only_merges_into_in_proj(self):
        """Adapter with only Mamba gate_proj/x_proj (train_mlp=False)."""
        state_dict: dict[str, torch.Tensor] = {
            "backbone.layers.0.mixer.in_proj.weight": torch.zeros(self.IN_PROJ_DIM, self.HIDDEN),
            "backbone.layers.0.mixer.out_proj.weight": torch.zeros(self.HIDDEN, self.INTERMEDIATE),
        }
        for e in range(self.NUM_EXPERTS):
            state_dict[f"backbone.layers.1.mixer.experts.{e}.up_proj.weight"] = torch.zeros(
                self.INTERMEDIATE, self.HIDDEN
            )

        model = _make_base_model(
            state_dict, config_dict={"architectures": ["NemotronHForCausalLM"]}
        )

        # Only ATTN group: gate_proj + x_proj, no experts
        adapter_weights = {
            "base_model.model.backbone.layers.0.mixer.gate_proj.lora_A.weight": (
                torch.ones(1, self.HIDDEN) * 0.5
            ),
            "base_model.model.backbone.layers.0.mixer.gate_proj.lora_B.weight": (
                torch.ones(self.INTERMEDIATE, 1)
            ),
            "base_model.model.backbone.layers.0.mixer.x_proj.lora_A.weight": (
                torch.ones(1, self.HIDDEN) * 0.3
            ),
            "base_model.model.backbone.layers.0.mixer.x_proj.lora_B.weight": (
                torch.ones(self.INTERMEDIATE, 1)
            ),
        }

        merge_adapter_weights(model, adapter_weights, {"lora_alpha": 1, "r": 1})

        in_proj = model.state_dict()["backbone.layers.0.mixer.in_proj.weight"]
        assert torch.allclose(
            in_proj[: self.INTERMEDIATE],
            torch.full_like(in_proj[: self.INTERMEDIATE], 0.5),
            atol=1e-6,
        )
        assert torch.allclose(
            in_proj[self.INTERMEDIATE : 2 * self.INTERMEDIATE],
            torch.full_like(in_proj[self.INTERMEDIATE : 2 * self.INTERMEDIATE], 0.3),
            atol=1e-6,
        )

        # Experts should be unchanged (no LoRA)
        for e in range(self.NUM_EXPERTS):
            w = model.state_dict()[f"backbone.layers.1.mixer.experts.{e}.up_proj.weight"]
            assert w.abs().sum() == 0

    def test_mlp_only_no_mamba_merge(self):
        """Adapter with only expert weights (train_attn=False)."""
        state_dict: dict[str, torch.Tensor] = {
            "backbone.layers.0.mixer.in_proj.weight": torch.zeros(self.IN_PROJ_DIM, self.HIDDEN),
        }
        for e in range(self.NUM_EXPERTS):
            state_dict[f"backbone.layers.1.mixer.experts.{e}.up_proj.weight"] = torch.zeros(
                self.INTERMEDIATE, self.HIDDEN
            )

        model = _make_base_model(
            state_dict, config_dict={"architectures": ["NemotronHForCausalLM"]}
        )

        # Only MLP group: experts, no Mamba
        adapter_weights = {
            "base_model.model.backbone.layers.1.mixer.experts.w1.lora_A.weight": (
                torch.ones(self.NUM_EXPERTS, 1, self.HIDDEN) * 0.1
            ),
            "base_model.model.backbone.layers.1.mixer.experts.w1.lora_B.weight": torch.ones(
                self.NUM_EXPERTS, self.INTERMEDIATE, 1
            ),
            "base_model.model.backbone.layers.1.mixer.experts.w2.lora_A.weight": torch.empty(0),
            "base_model.model.backbone.layers.1.mixer.experts.w2.lora_B.weight": torch.empty(0),
            "base_model.model.backbone.layers.1.mixer.experts.w3.lora_A.weight": torch.empty(0),
            "base_model.model.backbone.layers.1.mixer.experts.w3.lora_B.weight": torch.empty(0),
        }

        merge_adapter_weights(model, adapter_weights, {"lora_alpha": 1, "r": 1})

        # Experts should have delta
        for e in range(self.NUM_EXPERTS):
            w = model.state_dict()[f"backbone.layers.1.mixer.experts.{e}.up_proj.weight"]
            assert w.abs().sum() > 0, f"Expert {e} should have been merged"

        # in_proj should be unchanged (no Mamba LoRA)
        in_proj = model.state_dict()["backbone.layers.0.mixer.in_proj.weight"]
        assert in_proj.abs().sum() == 0, "in_proj should be unchanged without Mamba LoRA"


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
        with pytest.raises(WeightsMergeError, match="both A and B have 1 expert"):
            expand_expert_lora_tensors(torch.ones(1, 2, 4), torch.ones(1, 8, 2))

    def test_already_matched_is_noop(self):
        lora_A = torch.ones(4, 2, 4)
        lora_B = torch.ones(4, 8, 2)
        out_A, out_B = expand_expert_lora_tensors(lora_A, lora_B)
        assert out_A is lora_A
        assert out_B is lora_B

    def test_mismatched_expert_counts_raises(self):
        with pytest.raises(WeightsMergeError, match="Expert count mismatch"):
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
        with pytest.raises(WeightsMergeError, match="does not exist"):
            plan_merge_ops(adapter, {"lora_alpha": 1, "r": 1}, keys, profile)

    def test_missing_config_key_raises(self):
        with pytest.raises(WeightsMergeError, match="lora_alpha"):
            plan_merge_ops({}, {"r": 1}, set(), MergeProfile())

    def test_invalid_expert_layout_raises(self):
        with pytest.raises(WeightsMergeError, match="Invalid expert_layout"):
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
        with pytest.raises(WeightsMergeError, match="Shape mismatch"):
            apply_merge_op(tensors, op)


# ---------------------------------------------------------------------------
# validate_merge_op_shapes
# ---------------------------------------------------------------------------


class TestValidateMergeOpShapes:
    def test_valid_2d_op_passes(self):
        ops = {
            "q_proj.weight": [
                MergeOp(
                    target_key="q_proj.weight", lora_A=torch.ones(1, 4), lora_B=torch.ones(8, 1)
                )
            ]
        }
        shapes = {"q_proj.weight": (8, 4)}
        validate_merge_op_shapes(ops, shapes)  # should not raise

    def test_invalid_2d_shape_raises(self):
        ops = {
            "q_proj.weight": [
                MergeOp(
                    target_key="q_proj.weight", lora_A=torch.ones(1, 8), lora_B=torch.ones(4, 1)
                )
            ]
        }
        shapes = {"q_proj.weight": (8, 4)}  # target is (8,4) but delta is (4,8)
        with pytest.raises(WeightsMergeError, match=r"Shape mismatch.*q_proj"):
            validate_merge_op_shapes(ops, shapes)

    def test_valid_3d_fused_concatenated_passes(self):
        ops = {
            "gate_up_proj": [
                MergeOp(
                    target_key="gate_up_proj",
                    lora_A=torch.ones(2, 1, 4),
                    lora_B=torch.ones(2, 4, 1),
                    is_expert_3d=True,
                    fused_proj_idx=0,
                    fused_proj_interleaved=False,
                )
            ]
        }
        # Target is (2, 4, 8) — fused gate+up, each half is (2, 4, 4)
        # Delta via bmm is (2, 4, 4) which matches the half
        shapes = {"gate_up_proj": (2, 4, 8)}
        validate_merge_op_shapes(ops, shapes)  # should not raise

    def test_invalid_3d_fused_shape_raises(self):
        ops = {
            "gate_up_proj": [
                MergeOp(
                    target_key="gate_up_proj",
                    lora_A=torch.ones(2, 1, 4),
                    lora_B=torch.ones(2, 6, 1),  # wrong out_dim
                    is_expert_3d=True,
                    fused_proj_idx=0,
                    fused_proj_interleaved=False,
                )
            ]
        }
        shapes = {"gate_up_proj": (2, 4, 8)}  # half is (2, 4, 4), delta is (2, 4, 6)
        with pytest.raises(WeightsMergeError, match=r"gate_up_proj"):
            validate_merge_op_shapes(ops, shapes)

    def test_valid_3d_fused_transposed_passes(self):
        """Qwen3.5 MoE: gate_up_proj = (n, fused, hidden) — fused on dim 1."""
        ops = {
            "gate_up_proj": [
                MergeOp(
                    target_key="gate_up_proj",
                    lora_A=torch.ones(2, 1, 8),  # (n_exp, rank, hidden=8)
                    lora_B=torch.ones(2, 3, 1),  # (n_exp, intermediate=3, rank)
                    is_expert_3d=True,
                    fused_proj_idx=0,
                    fused_proj_interleaved=False,
                )
            ]
        }
        # Target (2, 6, 8): fused on dim 1 (6=2*3), hidden=8
        # Delta via bmm is (2, 8, 3) → transposed to (2, 3, 8) to match slice
        shapes = {"gate_up_proj": (2, 6, 8)}
        validate_merge_op_shapes(ops, shapes)  # should not raise

    def test_valid_3d_non_fused_passes(self):
        ops = {
            "down_proj": [
                MergeOp(
                    target_key="down_proj",
                    lora_A=torch.ones(2, 1, 4),
                    lora_B=torch.ones(2, 8, 1),
                    is_expert_3d=True,
                    fused_proj_idx=None,
                )
            ]
        }
        shapes = {"down_proj": (2, 4, 8)}
        validate_merge_op_shapes(ops, shapes)  # should not raise


# ---------------------------------------------------------------------------
# Per-model profile detection and dispatch
# ---------------------------------------------------------------------------


class TestPerModelProfileDetection:
    """Tests for per-model detect_profile functions and model_family dispatch."""

    def test_default_profile_sets_model_family(self):
        config: dict = {"architectures": ["QwenForCausalLM"]}
        keys = {"model.layers.0.self_attn.q_proj.weight"}
        profile = detect_default_profile(config, keys)
        assert profile.model_family == "default"

    def test_gpt_oss_profile_sets_model_family(self):
        config: dict = {"architectures": ["GptOssForCausalLM"]}
        keys = {"model.layers.0.self_attn.q_proj.weight"}
        profile = detect_gpt_oss_profile(config, keys)
        assert profile is not None
        assert profile.model_family == "gpt_oss"

    def test_gpt_oss_returns_none_for_non_gpt_oss(self):
        config: dict = {"architectures": ["QwenForCausalLM"]}
        keys = {"model.layers.0.self_attn.q_proj.weight"}
        assert detect_gpt_oss_profile(config, keys) is None

    def test_deepseek_profile_sets_model_family(self):
        config: dict = {"model_type": "deepseek_v3"}
        keys = {"model.layers.0.self_attn.q_proj.weight"}
        profile = detect_deepseek_profile(config, keys)
        assert profile is not None
        assert profile.model_family == "deepseek"
        assert profile.expert_layout == "separate"

    def test_deepseek_returns_none_for_non_deepseek(self):
        config: dict = {"model_type": "qwen3"}
        keys = {"model.layers.0.self_attn.q_proj.weight"}
        assert detect_deepseek_profile(config, keys) is None

    def test_detect_merge_profile_sets_model_family_for_gpt_oss(self):
        config: dict = {"architectures": ["GptOssForCausalLM"]}
        keys = {"model.layers.0.self_attn.q_proj.weight"}
        profile = detect_merge_profile(config, keys)
        assert profile.model_family == "gpt_oss"

    def test_detect_merge_profile_sets_model_family_for_deepseek(self):
        config: dict = {"model_type": "deepseek_v3"}
        keys = {"model.layers.0.self_attn.q_proj.weight"}
        profile = detect_merge_profile(config, keys)
        assert profile.model_family == "deepseek"

    def test_detect_merge_profile_defaults_to_default(self):
        config: dict = {"architectures": ["QwenForCausalLM"]}
        keys = {"model.layers.0.self_attn.q_proj.weight"}
        profile = detect_merge_profile(config, keys)
        assert profile.model_family == "default"

    def test_unknown_model_family_raises(self):
        """plan_merge_ops rejects unknown model_family values."""
        profile = MergeProfile(model_family="nonexistent")
        with pytest.raises(WeightsMergeError, match="Unknown model_family"):
            plan_merge_ops({}, {"lora_alpha": 1, "r": 1}, set(), profile)

    def test_qwen3_5_profile_sets_model_family(self):
        config: dict = {"model_type": "qwen3_5"}
        keys: set[str] = {"model.language_model.layers.0.linear_attn.in_proj_qkv.weight"}
        profile = detect_qwen3_5_profile(config, keys)
        assert profile is not None
        assert profile.model_family == "qwen3_5"
        assert profile.split_qkv_projections is True
        assert profile.has_language_model_prefix is True
        assert profile.expert_layout == "separate"

    def test_qwen3_5_moe_profile(self):
        config: dict = {"model_type": "qwen3_5_moe"}
        keys: set[str] = {
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
        }
        profile = detect_qwen3_5_profile(config, keys)
        assert profile is not None
        assert profile.expert_layout == "fused_concatenated"

    def test_qwen3_5_returns_none_for_qwen3(self):
        config: dict = {"model_type": "qwen3"}
        keys: set[str] = {"model.layers.0.self_attn.q_proj.weight"}
        assert detect_qwen3_5_profile(config, keys) is None

    def test_detect_merge_profile_dispatches_qwen3_5(self):
        config: dict = {"model_type": "qwen3_5"}
        keys: set[str] = {"model.language_model.layers.0.linear_attn.in_proj_qkv.weight"}
        profile = detect_merge_profile(config, keys)
        assert profile.model_family == "qwen3_5"
        assert profile.split_qkv_projections is True


# ---------------------------------------------------------------------------
# Qwen3.5 split in_proj_q/k/v → fused in_proj_qkv
# ---------------------------------------------------------------------------


def _make_qkv_adapter(
    q_out: int, k_out: int, v_out: int, in_dim: int, rank: int = 1
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Build a minimal adapter dict with in_proj_q/k/v LoRA weights.

    Uses fill values 1/2/3 for Q/K/V so we can verify each slice is updated
    with the correct delta independently.
    """
    prefix = "base_model.model.model.layers.0.linear_attn"
    adapter_weights = {
        f"{prefix}.in_proj_q.lora_A.weight": torch.ones(rank, in_dim),
        f"{prefix}.in_proj_q.lora_B.weight": torch.ones(q_out, rank) * 1.0,
        f"{prefix}.in_proj_k.lora_A.weight": torch.ones(rank, in_dim),
        f"{prefix}.in_proj_k.lora_B.weight": torch.ones(k_out, rank) * 2.0,
        f"{prefix}.in_proj_v.lora_A.weight": torch.ones(rank, in_dim),
        f"{prefix}.in_proj_v.lora_B.weight": torch.ones(v_out, rank) * 3.0,
    }
    adapter_config = {"lora_alpha": 1, "r": rank}
    return adapter_weights, adapter_config


class TestQwen35QkvFusion:
    """Tests for the in_proj_q/k/v → in_proj_qkv fusion fix."""

    Q_OUT = 4
    K_OUT = 4
    V_OUT = 8
    IN_DIM = 6
    FUSED_ROWS = Q_OUT + K_OUT + V_OUT  # 16

    def _make_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "model.layers.0.linear_attn.in_proj_qkv.weight": torch.zeros(
                self.FUSED_ROWS, self.IN_DIM
            )
        }

    def _make_profile(self, **kwargs: Any) -> MergeProfile:
        return MergeProfile(model_family="qwen3_5", split_qkv_projections=True, **kwargs)

    def test_planning_maps_to_fused_key(self):
        adapter_weights, adapter_config = _make_qkv_adapter(
            self.Q_OUT, self.K_OUT, self.V_OUT, self.IN_DIM
        )
        model_state_keys = set(self._make_state_dict())
        profile = self._make_profile()
        ops = plan_merge_ops(adapter_weights, adapter_config, model_state_keys, profile)

        fused_key = "model.layers.0.linear_attn.in_proj_qkv.weight"
        assert fused_key in ops
        assert not any(
            k.endswith((".in_proj_q.weight", ".in_proj_k.weight", ".in_proj_v.weight")) for k in ops
        )
        assert len(ops[fused_key]) == 3

    def test_slice_starts_are_correct(self):
        adapter_weights, adapter_config = _make_qkv_adapter(
            self.Q_OUT, self.K_OUT, self.V_OUT, self.IN_DIM
        )
        model_state_keys = set(self._make_state_dict())
        profile = self._make_profile()
        ops = plan_merge_ops(adapter_weights, adapter_config, model_state_keys, profile)
        merge_ops = ops["model.layers.0.linear_attn.in_proj_qkv.weight"]
        starts = sorted(op.slice_start for op in merge_ops if op.slice_start is not None)
        assert starts == [0, self.Q_OUT, self.Q_OUT + self.K_OUT]

    def test_correct_delta_applied_to_each_slice(self):
        adapter_weights, adapter_config = _make_qkv_adapter(
            self.Q_OUT, self.K_OUT, self.V_OUT, self.IN_DIM
        )
        state_dict = self._make_state_dict()
        profile = self._make_profile()
        ops = plan_merge_ops(adapter_weights, adapter_config, set(state_dict.keys()), profile)
        for op_list in ops.values():
            for op in op_list:
                apply_merge_op(state_dict, op)

        fused = state_dict["model.layers.0.linear_attn.in_proj_qkv.weight"]
        assert torch.allclose(fused[: self.Q_OUT], torch.full_like(fused[: self.Q_OUT], 1.0))
        assert torch.allclose(
            fused[self.Q_OUT : self.Q_OUT + self.K_OUT],
            torch.full_like(fused[self.Q_OUT : self.Q_OUT + self.K_OUT], 2.0),
        )
        assert torch.allclose(
            fused[self.Q_OUT + self.K_OUT :],
            torch.full_like(fused[self.Q_OUT + self.K_OUT :], 3.0),
        )

    def test_slices_do_not_overlap(self):
        adapter_weights, adapter_config = _make_qkv_adapter(
            self.Q_OUT, self.K_OUT, self.V_OUT, self.IN_DIM
        )
        fused = torch.full((self.FUSED_ROWS, self.IN_DIM), 99.0)
        state_dict = {"model.layers.0.linear_attn.in_proj_qkv.weight": fused}
        profile = self._make_profile()
        ops = plan_merge_ops(adapter_weights, adapter_config, set(state_dict.keys()), profile)
        for op_list in ops.values():
            for op in op_list:
                apply_merge_op(state_dict, op)

        assert torch.allclose(fused[: self.Q_OUT], torch.full_like(fused[: self.Q_OUT], 100.0))
        assert torch.allclose(
            fused[self.Q_OUT : self.Q_OUT + self.K_OUT],
            torch.full_like(fused[self.Q_OUT : self.Q_OUT + self.K_OUT], 101.0),
        )
        assert torch.allclose(
            fused[self.Q_OUT + self.K_OUT :],
            torch.full_like(fused[self.Q_OUT + self.K_OUT :], 102.0),
        )

    def test_unequal_qkv_dims(self):
        q_out, k_out, v_out, in_dim = 3, 5, 7, 4
        adapter_weights, adapter_config = _make_qkv_adapter(q_out, k_out, v_out, in_dim)
        fused_rows = q_out + k_out + v_out
        state_dict = {
            "model.layers.0.linear_attn.in_proj_qkv.weight": torch.zeros(fused_rows, in_dim)
        }
        profile = self._make_profile()
        ops = plan_merge_ops(adapter_weights, adapter_config, set(state_dict.keys()), profile)
        for op_list in ops.values():
            for op in op_list:
                apply_merge_op(state_dict, op)

        fused = state_dict["model.layers.0.linear_attn.in_proj_qkv.weight"]
        assert torch.allclose(fused[:q_out], torch.full((q_out, in_dim), 1.0))
        assert torch.allclose(fused[q_out : q_out + k_out], torch.full((k_out, in_dim), 2.0))
        assert torch.allclose(fused[q_out + k_out :], torch.full((v_out, in_dim), 3.0))

    def test_vision_prefix_with_qkv_fusion(self):
        adapter_weights, adapter_config = _make_qkv_adapter(
            self.Q_OUT, self.K_OUT, self.V_OUT, self.IN_DIM
        )
        state_dict = {
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": torch.zeros(
                self.FUSED_ROWS, self.IN_DIM
            ),
        }
        profile = self._make_profile(has_language_model_prefix=True)
        ops = plan_merge_ops(adapter_weights, adapter_config, set(state_dict.keys()), profile)
        for op_list in ops.values():
            for op in op_list:
                apply_merge_op(state_dict, op)

        fused = state_dict["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"]
        assert torch.allclose(fused[: self.Q_OUT], torch.full((self.Q_OUT, self.IN_DIM), 1.0))

    def test_validate_passes_for_sliced_ops(self):
        fused_key = "model.layers.0.linear_attn.in_proj_qkv.weight"
        ops = {
            fused_key: [
                MergeOp(
                    target_key=fused_key,
                    lora_A=torch.ones(1, self.IN_DIM),
                    lora_B=torch.ones(self.Q_OUT, 1),
                    slice_start=0,
                ),
                MergeOp(
                    target_key=fused_key,
                    lora_A=torch.ones(1, self.IN_DIM),
                    lora_B=torch.ones(self.K_OUT, 1),
                    slice_start=self.Q_OUT,
                ),
                MergeOp(
                    target_key=fused_key,
                    lora_A=torch.ones(1, self.IN_DIM),
                    lora_B=torch.ones(self.V_OUT, 1),
                    slice_start=self.Q_OUT + self.K_OUT,
                ),
            ]
        }
        shapes = {fused_key: (self.FUSED_ROWS, self.IN_DIM)}
        validate_merge_op_shapes(ops, shapes)  # should not raise

    def test_validate_rejects_slice_overflow(self):
        ops = {
            "in_proj_qkv.weight": [
                MergeOp(
                    target_key="in_proj_qkv.weight",
                    lora_A=torch.ones(1, self.IN_DIM),
                    lora_B=torch.ones(self.Q_OUT, 1),
                    slice_start=self.FUSED_ROWS - 1,
                )
            ]
        }
        shapes = {"in_proj_qkv.weight": (self.FUSED_ROWS, self.IN_DIM)}
        with pytest.raises(WeightsMergeError, match=r"Shape mismatch.*in_proj_qkv"):
            validate_merge_op_shapes(ops, shapes)

    def test_validate_rejects_in_dim_mismatch(self):
        ops = {
            "in_proj_qkv.weight": [
                MergeOp(
                    target_key="in_proj_qkv.weight",
                    lora_A=torch.ones(1, self.IN_DIM + 1),
                    lora_B=torch.ones(self.Q_OUT, 1),
                    slice_start=0,
                )
            ]
        }
        shapes = {"in_proj_qkv.weight": (self.FUSED_ROWS, self.IN_DIM)}
        with pytest.raises(WeightsMergeError, match=r"Shape mismatch.*in_proj_qkv"):
            validate_merge_op_shapes(ops, shapes)


# ---------------------------------------------------------------------------
# unembed_tokens remapping for vision models (tied vs non-tied embeddings)
# ---------------------------------------------------------------------------


def _make_unembed_adapter(
    vocab: int, hidden: int, rank: int = 1
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Build a minimal adapter with a single unembed_tokens LoRA."""
    prefix = "base_model.model.model.unembed_tokens"
    adapter_weights = {
        f"{prefix}.lora_A.weight": torch.ones(rank, hidden),
        f"{prefix}.lora_B.weight": torch.ones(vocab, rank),
    }
    return adapter_weights, {"lora_alpha": 1, "r": rank}


class TestUnembedTokensVisionRemap:
    """Tests for unembed_tokens → lm_head / embed_tokens remap in vision models."""

    VOCAB = 8
    HIDDEN = 4

    def _make_profile(self, **kwargs: Any) -> MergeProfile:
        return MergeProfile(model_family="qwen3_5", split_qkv_projections=True, **kwargs)

    def test_tied_embeddings_merges_into_embed_tokens(self):
        embed = torch.zeros(self.VOCAB, self.HIDDEN)
        state_dict: dict[str, torch.Tensor] = {
            "model.language_model.embed_tokens.weight": embed,
        }
        profile = self._make_profile(has_language_model_prefix=True)
        adapter_weights, config = _make_unembed_adapter(self.VOCAB, self.HIDDEN)
        ops = plan_merge_ops(adapter_weights, config, set(state_dict.keys()), profile)
        for op_list in ops.values():
            for op in op_list:
                apply_merge_op(state_dict, op)
        assert torch.allclose(embed, torch.ones(self.VOCAB, self.HIDDEN))

    def test_non_tied_embeddings_merges_into_lm_head(self):
        embed = torch.zeros(self.VOCAB, self.HIDDEN)
        lm_head = torch.zeros(self.VOCAB, self.HIDDEN)
        state_dict: dict[str, torch.Tensor] = {
            "model.language_model.embed_tokens.weight": embed,
            "lm_head.weight": lm_head,
        }
        profile = self._make_profile(has_language_model_prefix=True)
        adapter_weights, config = _make_unembed_adapter(self.VOCAB, self.HIDDEN)
        ops = plan_merge_ops(adapter_weights, config, set(state_dict.keys()), profile)
        for op_list in ops.values():
            for op in op_list:
                apply_merge_op(state_dict, op)
        assert torch.allclose(lm_head, torch.ones(self.VOCAB, self.HIDDEN))
        assert torch.allclose(embed, torch.zeros(self.VOCAB, self.HIDDEN))
