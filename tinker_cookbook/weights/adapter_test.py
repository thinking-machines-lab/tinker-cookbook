"""Unit tests for build_lora_adapter (Tinker → PEFT adapter conversion).

Uses synthetic safetensors files and adapter weights to test the conversion
pipeline without requiring real HF models or network access.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tinker_cookbook.exceptions import WeightsAdapterError
from tinker_cookbook.weights._adapter import build_lora_adapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RANK = 2
HIDDEN = 8
OUT_DIM = 16
ALPHA = 4


def _create_synthetic_model(
    model_dir: Path,
    config_dict: dict,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Create a minimal synthetic HF model directory (config.json + safetensors)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config_dict))
    save_file(state_dict, str(model_dir / "model.safetensors"))


def _create_adapter(
    adapter_dir: Path,
    weights: dict[str, torch.Tensor],
    *,
    lora_alpha: int = ALPHA,
    rank: int = RANK,
) -> None:
    """Create a synthetic Tinker adapter directory."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"lora_alpha": lora_alpha, "r": rank})
    )


def _load_peft_output(output_dir: Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Load PEFT adapter output (weights + config)."""
    weights = load_file(str(output_dir / "adapter_model.safetensors"))
    with open(output_dir / "adapter_config.json") as f:
        config = json.load(f)
    return weights, config


# ---------------------------------------------------------------------------
# Standard dense model config
# ---------------------------------------------------------------------------

_DENSE_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": HIDDEN,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
}

_DENSE_STATE_DICT = {
    "model.layers.0.self_attn.q_proj.weight": torch.zeros(OUT_DIM, HIDDEN),
    "model.layers.0.self_attn.k_proj.weight": torch.zeros(OUT_DIM, HIDDEN),
    "model.layers.0.self_attn.v_proj.weight": torch.zeros(OUT_DIM, HIDDEN),
    "model.layers.0.self_attn.o_proj.weight": torch.zeros(HIDDEN, OUT_DIM),
    "model.layers.0.mlp.gate_proj.weight": torch.zeros(OUT_DIM, HIDDEN),
    "model.layers.0.mlp.up_proj.weight": torch.zeros(OUT_DIM, HIDDEN),
    "model.layers.0.mlp.down_proj.weight": torch.zeros(HIDDEN, OUT_DIM),
    "model.embed_tokens.weight": torch.zeros(100, HIDDEN),
    "lm_head.weight": torch.zeros(100, HIDDEN),
}


def _make_dense_adapter_weights() -> dict[str, torch.Tensor]:
    """Create adapter weights targeting q_proj and gate_proj."""
    return {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(RANK, HIDDEN),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(OUT_DIM, RANK)
        * 2,
        "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight": torch.ones(RANK, HIDDEN) * 3,
        "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight": torch.ones(OUT_DIM, RANK)
        * 4,
    }


# ---------------------------------------------------------------------------
# Tests: Dense model conversion
# ---------------------------------------------------------------------------


class TestDenseConversion:
    def test_output_has_correct_peft_keys(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _DENSE_CONFIG, _DENSE_STATE_DICT)
        _create_adapter(adapter_dir, _make_dense_adapter_weights())

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        weights, config = _load_peft_output(output_dir)

        # Check PEFT key naming convention.
        expected_keys = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
            "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight",
            "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight",
        }
        assert set(weights.keys()) == expected_keys

    def test_tensors_are_not_scaled(self, tmp_path: Path) -> None:
        """PEFT adapters store raw tensors; scaling is applied by the serving framework."""
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _DENSE_CONFIG, _DENSE_STATE_DICT)
        adapter_weights = _make_dense_adapter_weights()
        _create_adapter(adapter_dir, adapter_weights)

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        peft_weights, _ = _load_peft_output(output_dir)

        # lora_B should NOT be scaled by alpha/r.
        orig_B = adapter_weights["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
        peft_B = peft_weights["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
        assert torch.equal(orig_B, peft_B)

    def test_peft_config_has_required_fields(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _DENSE_CONFIG, _DENSE_STATE_DICT)
        _create_adapter(adapter_dir, _make_dense_adapter_weights())

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        _, config = _load_peft_output(output_dir)

        assert config["peft_type"] == "LORA"
        assert config["r"] == RANK
        assert config["lora_alpha"] == ALPHA
        assert config["task_type"] == "CAUSAL_LM"
        assert config["bias"] == "none"
        assert isinstance(config["target_modules"], list)
        assert "q_proj" in config["target_modules"]
        assert "gate_proj" in config["target_modules"]

    def test_target_modules_derived_from_adapter_keys(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _DENSE_CONFIG, _DENSE_STATE_DICT)
        _create_adapter(adapter_dir, _make_dense_adapter_weights())

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        _, config = _load_peft_output(output_dir)
        assert sorted(config["target_modules"]) == ["gate_proj", "q_proj"]


# ---------------------------------------------------------------------------
# Tests: MoE separate expert expansion
# ---------------------------------------------------------------------------

_MOE_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "hidden_size": HIDDEN,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
}

NUM_EXPERTS = 3


def _make_moe_state_dict() -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(OUT_DIM, HIDDEN),
    }
    for e in range(NUM_EXPERTS):
        state_dict[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = torch.zeros(
            OUT_DIM, HIDDEN
        )
        state_dict[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = torch.zeros(OUT_DIM, HIDDEN)
        state_dict[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = torch.zeros(
            HIDDEN, OUT_DIM
        )
    return state_dict


def _make_moe_adapter_weights() -> dict[str, torch.Tensor]:
    """Create adapter with 3D expert LoRA tensors (w1=gate, w3=up)."""
    return {
        "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight": torch.ones(
            NUM_EXPERTS, RANK, HIDDEN
        ),
        "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight": torch.ones(
            NUM_EXPERTS, OUT_DIM, RANK
        ),
        "base_model.model.model.layers.0.mlp.experts.w3.lora_A.weight": torch.ones(
            NUM_EXPERTS, RANK, HIDDEN
        )
        * 2,
        "base_model.model.model.layers.0.mlp.experts.w3.lora_B.weight": torch.ones(
            NUM_EXPERTS, OUT_DIM, RANK
        )
        * 2,
    }


class TestMoEExpertExpansion:
    def test_3d_expanded_to_per_expert_2d_keys(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _MOE_CONFIG, _make_moe_state_dict())
        _create_adapter(adapter_dir, _make_moe_adapter_weights())

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        weights, config = _load_peft_output(output_dir)

        # Should have per-expert keys (w1→gate_proj, w3→up_proj).
        for e in range(NUM_EXPERTS):
            assert (
                f"base_model.model.model.layers.0.mlp.experts.{e}.gate_proj.lora_A.weight"
                in weights
            )
            assert (
                f"base_model.model.model.layers.0.mlp.experts.{e}.gate_proj.lora_B.weight"
                in weights
            )
            assert (
                f"base_model.model.model.layers.0.mlp.experts.{e}.up_proj.lora_A.weight" in weights
            )
            assert (
                f"base_model.model.model.layers.0.mlp.experts.{e}.up_proj.lora_B.weight" in weights
            )

        # Should NOT have the original 3D keys.
        assert not any("experts.w1" in k for k in weights)
        assert not any("experts.w3" in k for k in weights)

    def test_per_expert_tensors_are_2d(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _MOE_CONFIG, _make_moe_state_dict())
        _create_adapter(adapter_dir, _make_moe_adapter_weights())

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        weights, _ = _load_peft_output(output_dir)

        for key, tensor in weights.items():
            assert tensor.ndim == 2, f"{key} has {tensor.ndim}D, expected 2D"

    def test_expert_target_modules(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _MOE_CONFIG, _make_moe_state_dict())
        _create_adapter(adapter_dir, _make_moe_adapter_weights())

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        _, config = _load_peft_output(output_dir)
        assert sorted(config["target_modules"]) == ["gate_proj", "up_proj"]

    def test_broadcast_lora_a_expanded_correctly(self, tmp_path: Path) -> None:
        """Real Tinker pattern: lora_A shared (1, rank, dim), lora_B per-expert.

        Verifies that broadcast expansion produces correct per-expert 2D keys
        with the right shapes and values.
        """
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _MOE_CONFIG, _make_moe_state_dict())

        prefix = "base_model.model.model.layers.0.mlp.experts"
        # w1: A shared (1, rank, hidden), B per-expert (num_experts, out_dim, rank)
        # w2: A per-expert (num_experts, rank, out_dim), B shared (1, hidden, rank)
        broadcast_adapter = {
            f"{prefix}.w1.lora_A.weight": torch.ones(1, RANK, HIDDEN) * 0.1,
            f"{prefix}.w1.lora_B.weight": torch.ones(NUM_EXPERTS, OUT_DIM, RANK),
            f"{prefix}.w3.lora_A.weight": torch.ones(1, RANK, HIDDEN) * 0.2,
            f"{prefix}.w3.lora_B.weight": torch.ones(NUM_EXPERTS, OUT_DIM, RANK),
            f"{prefix}.w2.lora_A.weight": torch.ones(NUM_EXPERTS, RANK, OUT_DIM) * 0.3,
            f"{prefix}.w2.lora_B.weight": torch.ones(1, HIDDEN, RANK),
        }
        _create_adapter(adapter_dir, broadcast_adapter)

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        weights, config = _load_peft_output(output_dir)

        # All per-expert keys should exist
        for e in range(NUM_EXPERTS):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                a_key = f"base_model.model.model.layers.0.mlp.experts.{e}.{proj}.lora_A.weight"
                b_key = f"base_model.model.model.layers.0.mlp.experts.{e}.{proj}.lora_B.weight"
                assert a_key in weights, f"Missing {a_key}"
                assert b_key in weights, f"Missing {b_key}"
                assert weights[a_key].ndim == 2
                assert weights[b_key].ndim == 2

        # Verify broadcast: each expert's lora_A for w1 should be identical
        # (all expanded from the same shared tensor)
        a0 = weights[f"{prefix}.0.gate_proj.lora_A.weight"]
        a1 = weights[f"{prefix}.1.gate_proj.lora_A.weight"]
        assert torch.equal(a0, a1), "Broadcast expansion produced different values"

        assert sorted(config["target_modules"]) == [
            "down_proj",
            "gate_proj",
            "up_proj",
        ]


# ---------------------------------------------------------------------------
# Tests: Vision model prefix
# ---------------------------------------------------------------------------

_VISION_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": HIDDEN,
}

_VISION_STATE_DICT = {
    "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(OUT_DIM, HIDDEN),
    "model.language_model.embed_tokens.weight": torch.zeros(100, HIDDEN),
    "lm_head.weight": torch.zeros(100, HIDDEN),
}


class TestVisionPrefix:
    def test_language_model_prefix_in_output_keys(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _VISION_CONFIG, _VISION_STATE_DICT)
        _create_adapter(
            adapter_dir,
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(
                    RANK, HIDDEN
                ),
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(
                    OUT_DIM, RANK
                ),
            },
        )

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        weights, _ = _load_peft_output(output_dir)
        assert (
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
            in weights
        )


# ---------------------------------------------------------------------------
# Tests: GPT-OSS key remap (.attn → .self_attn)
# ---------------------------------------------------------------------------

_GPT_OSS_CONFIG = {
    "architectures": ["GptOssForCausalLM"],
    "model_type": "gpt_oss",
    "hidden_size": HIDDEN,
}

_GPT_OSS_STATE_DICT = {
    "model.layers.0.self_attn.q_proj.weight": torch.zeros(OUT_DIM, HIDDEN),
    "model.layers.0.self_attn.o_proj.weight": torch.zeros(HIDDEN, OUT_DIM),
    "model.embed_tokens.weight": torch.zeros(100, HIDDEN),
    "lm_head.weight": torch.zeros(100, HIDDEN),
}


class TestGptOssRemap:
    def test_attn_remapped_to_self_attn(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _GPT_OSS_CONFIG, _GPT_OSS_STATE_DICT)
        # Tinker adapter uses .attn (GPT-OSS internal naming).
        _create_adapter(
            adapter_dir,
            {
                "base_model.model.model.layers.0.attn.q_proj.lora_A.weight": torch.ones(
                    RANK, HIDDEN
                ),
                "base_model.model.model.layers.0.attn.q_proj.lora_B.weight": torch.ones(
                    OUT_DIM, RANK
                ),
            },
        )

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        weights, _ = _load_peft_output(output_dir)
        # Output should use .self_attn (HF naming).
        assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in weights
        assert not any(".attn." in k for k in weights)


# ---------------------------------------------------------------------------
# Tests: Qwen3.5 split QKV
# ---------------------------------------------------------------------------

_QWEN35_CONFIG = {
    "architectures": ["Qwen3_5ForCausalLM"],
    "model_type": "qwen3_5",
    "hidden_size": HIDDEN,
}

_QWEN35_STATE_DICT = {
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": torch.zeros(
        OUT_DIM * 3, HIDDEN
    ),
    "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(OUT_DIM, HIDDEN),
    "model.language_model.embed_tokens.weight": torch.zeros(100, HIDDEN),
    "lm_head.weight": torch.zeros(100, HIDDEN),
}


class TestQwen35SplitQkv:
    def test_split_qkv_keys_preserved_as_separate(self, tmp_path: Path) -> None:
        """Split in_proj_q/k/v should be output as separate PEFT keys (Option A)."""
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _QWEN35_CONFIG, _QWEN35_STATE_DICT)
        _create_adapter(
            adapter_dir,
            {
                "base_model.model.model.layers.0.linear_attn.in_proj_q.lora_A.weight": torch.ones(
                    RANK, HIDDEN
                ),
                "base_model.model.model.layers.0.linear_attn.in_proj_q.lora_B.weight": torch.ones(
                    OUT_DIM, RANK
                ),
                "base_model.model.model.layers.0.linear_attn.in_proj_k.lora_A.weight": torch.ones(
                    RANK, HIDDEN
                ),
                "base_model.model.model.layers.0.linear_attn.in_proj_k.lora_B.weight": torch.ones(
                    OUT_DIM, RANK
                ),
                "base_model.model.model.layers.0.linear_attn.in_proj_v.lora_A.weight": torch.ones(
                    RANK, HIDDEN
                ),
                "base_model.model.model.layers.0.linear_attn.in_proj_v.lora_B.weight": torch.ones(
                    OUT_DIM, RANK
                ),
            },
        )

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        weights, config = _load_peft_output(output_dir)

        # Each component should be a separate PEFT key.
        for proj in ("in_proj_q", "in_proj_k", "in_proj_v"):
            assert any(proj in k for k in weights), f"Missing {proj} in PEFT output"

        # target_modules should include the split names.
        for proj in ("in_proj_q", "in_proj_k", "in_proj_v"):
            assert proj in config["target_modules"]


# ---------------------------------------------------------------------------
# Tests: Unsupported models
# ---------------------------------------------------------------------------


class TestUnsupportedModels:
    def test_deepseek_v3_raises_error(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(
            model_dir,
            {"architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3"},
            {"model.layers.0.self_attn.q_proj.weight": torch.zeros(OUT_DIM, HIDDEN)},
        )
        _create_adapter(
            adapter_dir,
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(
                    RANK, HIDDEN
                ),
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(
                    OUT_DIM, RANK
                ),
            },
        )

        with pytest.raises(WeightsAdapterError, match="DeepSeek"):
            build_lora_adapter(
                base_model=str(model_dir),
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
            )

    def test_nemotron_backbone_prefix_remapped(self, tmp_path: Path) -> None:
        """Nemotron HF checkpoints use 'backbone.*' but vLLM remaps to 'model.*'.

        The adapter conversion must produce PEFT keys with 'model.*' prefix
        (matching vLLM's internal parameter names) not 'backbone.*' (the HF
        checkpoint prefix).
        """
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(
            model_dir,
            {"architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h"},
            {"backbone.layers.0.mixer.q_proj.weight": torch.zeros(OUT_DIM, HIDDEN)},
        )
        _create_adapter(
            adapter_dir,
            {
                "base_model.model.backbone.layers.0.mixer.q_proj.lora_A.weight": torch.ones(
                    RANK, HIDDEN
                ),
                "base_model.model.backbone.layers.0.mixer.q_proj.lora_B.weight": torch.ones(
                    OUT_DIM, RANK
                ),
            },
        )

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        peft_weights, peft_config = _load_peft_output(output_dir)

        # PEFT keys should use model.* (vLLM's internal names), not backbone.*
        assert "base_model.model.model.layers.0.mixer.q_proj.lora_A.weight" in peft_weights
        assert "base_model.model.model.layers.0.mixer.q_proj.lora_B.weight" in peft_weights
        assert not any("backbone" in k for k in peft_weights), (
            "PEFT keys should not contain 'backbone' — vLLM remaps to 'model'"
        )
        assert "q_proj" in peft_config["target_modules"]


# ---------------------------------------------------------------------------
# Tests: Nemotron MoE expert conversion
# ---------------------------------------------------------------------------

# Nemotron MoE has only up_proj and down_proj per expert (no gate_proj).
# Tinker maps: w1=up_proj, w2=down_proj, w3=empty placeholder.
_NEMOTRON_MOE_CONFIG = {
    "architectures": ["NemotronHForCausalLM"],
    "model_type": "nemotron_h",
    "hidden_size": HIDDEN,
}

NEMOTRON_NUM_EXPERTS = 4
NEMOTRON_INTERMEDIATE = 12  # moe_intermediate_size
# in_proj fuses: gate (NEMOTRON_INTERMEDIATE) + x (NEMOTRON_INTERMEDIATE) + B + C + dt
NEMOTRON_IN_PROJ_OUT = NEMOTRON_INTERMEDIATE * 2 + 10  # +10 for B+C+dt


def _make_nemotron_moe_state_dict() -> dict[str, torch.Tensor]:
    """Create a Nemotron-like model with Mamba + attention + MoE layers."""
    state_dict: dict[str, torch.Tensor] = {
        # Mamba layer (layer 0) — in_proj fuses gate_proj + x_proj + B + C + dt
        "backbone.layers.0.mixer.in_proj.weight": torch.zeros(NEMOTRON_IN_PROJ_OUT, HIDDEN),
        "backbone.layers.0.mixer.out_proj.weight": torch.zeros(HIDDEN, NEMOTRON_INTERMEDIATE),
        # Attention layer (layer 2)
        "backbone.layers.2.mixer.q_proj.weight": torch.zeros(OUT_DIM, HIDDEN),
    }
    # MoE layer (layer 1) — only up_proj and down_proj per expert
    for e in range(NEMOTRON_NUM_EXPERTS):
        state_dict[f"backbone.layers.1.mixer.experts.{e}.up_proj.weight"] = torch.zeros(
            NEMOTRON_INTERMEDIATE, HIDDEN
        )
        state_dict[f"backbone.layers.1.mixer.experts.{e}.down_proj.weight"] = torch.zeros(
            HIDDEN, NEMOTRON_INTERMEDIATE
        )
    # Shared experts (non-MoE, standard dense keys)
    state_dict["backbone.layers.1.mixer.shared_experts.up_proj.weight"] = torch.zeros(
        NEMOTRON_INTERMEDIATE, HIDDEN
    )
    state_dict["backbone.layers.1.mixer.shared_experts.down_proj.weight"] = torch.zeros(
        HIDDEN, NEMOTRON_INTERMEDIATE
    )
    return state_dict


def _make_nemotron_moe_adapter_weights() -> dict[str, torch.Tensor]:
    """Create adapter weights matching real Tinker output for Nemotron MoE.

    Includes:
    - Dense attention LoRA (layer 0, q_proj)
    - Expert LoRA w1 (up_proj, 3D with broadcast) and w2 (down_proj, 3D with broadcast)
    - Empty w3 placeholder (no gate_proj in Nemotron)
    - Shared expert LoRA (standard 2D keys)
    """
    prefix = "base_model.model.backbone"
    return {
        # Mamba layer (layer 0) — gate_proj and x_proj trained separately
        f"{prefix}.layers.0.mixer.gate_proj.lora_A.weight": torch.ones(RANK, HIDDEN) * 2,
        f"{prefix}.layers.0.mixer.gate_proj.lora_B.weight": torch.ones(NEMOTRON_INTERMEDIATE, RANK)
        * 2,
        f"{prefix}.layers.0.mixer.x_proj.lora_A.weight": torch.ones(RANK, HIDDEN) * 3,
        f"{prefix}.layers.0.mixer.x_proj.lora_B.weight": torch.ones(NEMOTRON_INTERMEDIATE, RANK)
        * 3,
        # Attention layer (layer 2)
        f"{prefix}.layers.2.mixer.q_proj.lora_A.weight": torch.ones(RANK, HIDDEN),
        f"{prefix}.layers.2.mixer.q_proj.lora_B.weight": torch.ones(OUT_DIM, RANK),
        # Expert w1 (up_proj): lora_A shared (1 expert), lora_B per-expert
        f"{prefix}.layers.1.mixer.experts.w1.lora_A.weight": torch.ones(1, RANK, HIDDEN),
        f"{prefix}.layers.1.mixer.experts.w1.lora_B.weight": torch.ones(
            NEMOTRON_NUM_EXPERTS, NEMOTRON_INTERMEDIATE, RANK
        ),
        # Expert w2 (down_proj): lora_A per-expert, lora_B shared (1 expert)
        f"{prefix}.layers.1.mixer.experts.w2.lora_A.weight": torch.ones(
            NEMOTRON_NUM_EXPERTS, RANK, NEMOTRON_INTERMEDIATE
        ),
        f"{prefix}.layers.1.mixer.experts.w2.lora_B.weight": torch.ones(1, HIDDEN, RANK),
        # Expert w3 (gate_proj): empty — Nemotron has no gate_proj
        f"{prefix}.layers.1.mixer.experts.w3.lora_A.weight": torch.empty(0),
        f"{prefix}.layers.1.mixer.experts.w3.lora_B.weight": torch.empty(0),
        # Shared experts (standard 2D, not routed through expert expansion)
        f"{prefix}.layers.1.mixer.shared_experts.up_proj.lora_A.weight": torch.ones(RANK, HIDDEN),
        f"{prefix}.layers.1.mixer.shared_experts.up_proj.lora_B.weight": torch.ones(
            NEMOTRON_INTERMEDIATE, RANK
        ),
        f"{prefix}.layers.1.mixer.shared_experts.down_proj.lora_A.weight": torch.ones(
            RANK, NEMOTRON_INTERMEDIATE
        ),
        f"{prefix}.layers.1.mixer.shared_experts.down_proj.lora_B.weight": torch.ones(HIDDEN, RANK),
    }


class TestNemotronMoE:
    def test_empty_expert_tensors_skipped(self, tmp_path: Path) -> None:
        """Empty w3 expert LoRA tensors should be skipped without error."""
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _NEMOTRON_MOE_CONFIG, _make_nemotron_moe_state_dict())
        _create_adapter(adapter_dir, _make_nemotron_moe_adapter_weights())

        # This should not raise WeightsAdapterError about non-3D tensors.
        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        weights, _ = _load_peft_output(output_dir)
        # No keys should reference w3 or gate_proj (Nemotron doesn't have it).
        assert not any("w3" in k for k in weights)
        assert not any("gate_proj" in k for k in weights)

    def test_expert_keys_mapped_to_up_and_down_proj(self, tmp_path: Path) -> None:
        """Nemotron w1→up_proj, w2→down_proj (not w1→gate_proj)."""
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _NEMOTRON_MOE_CONFIG, _make_nemotron_moe_state_dict())
        _create_adapter(adapter_dir, _make_nemotron_moe_adapter_weights())

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        weights, config = _load_peft_output(output_dir)

        # Per-expert keys should use up_proj and down_proj.
        for e in range(NEMOTRON_NUM_EXPERTS):
            assert (
                f"base_model.model.model.layers.1.mixer.experts.{e}.up_proj.lora_A.weight"
                in weights
            )
            assert (
                f"base_model.model.model.layers.1.mixer.experts.{e}.down_proj.lora_A.weight"
                in weights
            )

        # backbone.* should be remapped to model.* for serving.
        assert not any("backbone" in k for k in weights)

        # target_modules should include up_proj, down_proj, q_proj, in_proj.
        assert "up_proj" in config["target_modules"]
        assert "down_proj" in config["target_modules"]
        assert "q_proj" in config["target_modules"]
        assert "in_proj" in config["target_modules"]

    def test_mamba_gate_x_merged_into_in_proj(self, tmp_path: Path) -> None:
        """gate_proj and x_proj LoRA should be merged into fused in_proj."""
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _NEMOTRON_MOE_CONFIG, _make_nemotron_moe_state_dict())
        _create_adapter(adapter_dir, _make_nemotron_moe_adapter_weights())

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        weights, config = _load_peft_output(output_dir)

        # gate_proj and x_proj should NOT appear as separate PEFT keys.
        assert not any("gate_proj" in k for k in weights if "experts" not in k)
        assert not any("x_proj" in k for k in weights)

        # Instead, in_proj should be present with doubled rank.
        in_proj_A = weights["base_model.model.model.layers.0.mixer.in_proj.lora_A.weight"]
        in_proj_B = weights["base_model.model.model.layers.0.mixer.in_proj.lora_B.weight"]
        assert in_proj_A.shape == (RANK * 2, HIDDEN), (
            f"in_proj lora_A should have rank 2*{RANK}={RANK * 2}, got {in_proj_A.shape}"
        )
        assert in_proj_B.shape == (NEMOTRON_IN_PROJ_OUT, RANK * 2), (
            f"in_proj lora_B should have shape ({NEMOTRON_IN_PROJ_OUT}, {RANK * 2}), got {in_proj_B.shape}"
        )

        # Verify block-diagonal structure: gate in first rank columns, x in second.
        # gate_proj values are 2, x_proj values are 3 (from _make_nemotron_moe_adapter_weights).
        # Row [0:NEMOTRON_INTERMEDIATE, 0:RANK] should be gate_lora_B (all 2s).
        assert torch.allclose(
            in_proj_B[:NEMOTRON_INTERMEDIATE, :RANK],
            torch.ones(NEMOTRON_INTERMEDIATE, RANK) * 2,
        )
        # Row [NEMOTRON_INTERMEDIATE:2*NEMOTRON_INTERMEDIATE, RANK:2*RANK] should be x_lora_B (all 3s).
        assert torch.allclose(
            in_proj_B[NEMOTRON_INTERMEDIATE : 2 * NEMOTRON_INTERMEDIATE, RANK : 2 * RANK],
            torch.ones(NEMOTRON_INTERMEDIATE, RANK) * 3,
        )
        # Remaining rows should be zero (B, C, dt don't have LoRA).
        assert in_proj_B[2 * NEMOTRON_INTERMEDIATE :].abs().sum() == 0

        # PEFT config should have rank_pattern and alpha_pattern for in_proj.
        assert config["rank_pattern"]["in_proj"] == RANK * 2
        assert config["alpha_pattern"]["in_proj"] == ALPHA * 2

    def test_per_expert_tensors_are_2d(self, tmp_path: Path) -> None:
        """All output tensors should be 2D (3D experts expanded to per-expert 2D)."""
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _NEMOTRON_MOE_CONFIG, _make_nemotron_moe_state_dict())
        _create_adapter(adapter_dir, _make_nemotron_moe_adapter_weights())

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        weights, _ = _load_peft_output(output_dir)
        for key, tensor in weights.items():
            assert tensor.ndim == 2, f"{key} has {tensor.ndim}D, expected 2D"

    def test_shared_experts_alongside_routed_experts(self, tmp_path: Path) -> None:
        """Shared expert keys (standard 2D) should coexist with routed expert keys."""
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _NEMOTRON_MOE_CONFIG, _make_nemotron_moe_state_dict())
        _create_adapter(adapter_dir, _make_nemotron_moe_adapter_weights())

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        weights, _ = _load_peft_output(output_dir)

        # Shared experts should be present as standard 2D keys.
        assert (
            "base_model.model.model.layers.1.mixer.shared_experts.up_proj.lora_A.weight" in weights
        )
        assert (
            "base_model.model.model.layers.1.mixer.shared_experts.down_proj.lora_A.weight"
            in weights
        )


# ---------------------------------------------------------------------------
# Tests: Partial LoRA coverage (user configures train_attn/train_mlp)
# ---------------------------------------------------------------------------


class TestNemotronPartialLora:
    """Test adapter conversion when only a subset of module groups has LoRA.

    In Tinker, users can set ``train_attn=False`` or ``train_mlp=False``
    in LoraConfig. Untrained modules are completely absent from the adapter
    (not saved as zeros). The conversion must handle any subset gracefully.
    """

    def test_attn_only_no_experts(self, tmp_path: Path) -> None:
        """train_attn=True, train_mlp=False: only Mamba + attention, no experts."""
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _NEMOTRON_MOE_CONFIG, _make_nemotron_moe_state_dict())

        # Adapter with only ATTN group modules (gate_proj, x_proj, q_proj)
        prefix = "base_model.model.backbone"
        _create_adapter(
            adapter_dir,
            {
                f"{prefix}.layers.0.mixer.gate_proj.lora_A.weight": torch.ones(RANK, HIDDEN),
                f"{prefix}.layers.0.mixer.gate_proj.lora_B.weight": torch.ones(
                    NEMOTRON_INTERMEDIATE, RANK
                ),
                f"{prefix}.layers.0.mixer.x_proj.lora_A.weight": torch.ones(RANK, HIDDEN),
                f"{prefix}.layers.0.mixer.x_proj.lora_B.weight": torch.ones(
                    NEMOTRON_INTERMEDIATE, RANK
                ),
                f"{prefix}.layers.2.mixer.q_proj.lora_A.weight": torch.ones(RANK, HIDDEN),
                f"{prefix}.layers.2.mixer.q_proj.lora_B.weight": torch.ones(OUT_DIM, RANK),
            },
        )

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        weights, config = _load_peft_output(output_dir)

        # Should have in_proj (merged gate+x) and q_proj, nothing else.
        assert any("in_proj" in k for k in weights)
        assert any("q_proj" in k for k in weights)
        assert not any("experts" in k for k in weights)
        assert not any("shared_experts" in k for k in weights)
        assert sorted(config["target_modules"]) == ["in_proj", "q_proj"]

    def test_mlp_only_no_attention(self, tmp_path: Path) -> None:
        """train_mlp=True, train_attn=False: only experts, no Mamba/attention."""
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _NEMOTRON_MOE_CONFIG, _make_nemotron_moe_state_dict())

        # Adapter with only MLP group modules (experts + shared experts)
        prefix = "base_model.model.backbone"
        _create_adapter(
            adapter_dir,
            {
                f"{prefix}.layers.1.mixer.experts.w1.lora_A.weight": torch.ones(1, RANK, HIDDEN),
                f"{prefix}.layers.1.mixer.experts.w1.lora_B.weight": torch.ones(
                    NEMOTRON_NUM_EXPERTS, NEMOTRON_INTERMEDIATE, RANK
                ),
                f"{prefix}.layers.1.mixer.experts.w2.lora_A.weight": torch.ones(
                    NEMOTRON_NUM_EXPERTS, RANK, NEMOTRON_INTERMEDIATE
                ),
                f"{prefix}.layers.1.mixer.experts.w2.lora_B.weight": torch.ones(1, HIDDEN, RANK),
                f"{prefix}.layers.1.mixer.experts.w3.lora_A.weight": torch.empty(0),
                f"{prefix}.layers.1.mixer.experts.w3.lora_B.weight": torch.empty(0),
                f"{prefix}.layers.1.mixer.shared_experts.up_proj.lora_A.weight": torch.ones(
                    RANK, HIDDEN
                ),
                f"{prefix}.layers.1.mixer.shared_experts.up_proj.lora_B.weight": torch.ones(
                    NEMOTRON_INTERMEDIATE, RANK
                ),
            },
        )

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        weights, config = _load_peft_output(output_dir)

        # Should have expert and shared_expert keys, no in_proj or q_proj.
        assert any("experts" in k for k in weights)
        assert any("shared_experts" in k for k in weights)
        assert not any("in_proj" in k for k in weights)
        assert not any("q_proj" in k for k in weights)

    def test_unembed_only(self, tmp_path: Path) -> None:
        """train_unembed=True, train_attn=False, train_mlp=False: only lm_head."""
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        # Need lm_head in the model state dict
        state_dict = _make_nemotron_moe_state_dict()
        state_dict["lm_head.weight"] = torch.zeros(100, HIDDEN)
        _create_synthetic_model(model_dir, _NEMOTRON_MOE_CONFIG, state_dict)

        # Real Tinker adapters use base_model.model.model.lm_head (not unembed_tokens)
        _create_adapter(
            adapter_dir,
            {
                "base_model.model.model.lm_head.lora_A.weight": torch.ones(RANK, HIDDEN),
                "base_model.model.model.lm_head.lora_B.weight": torch.ones(100, RANK),
            },
        )

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        weights, config = _load_peft_output(output_dir)

        assert any("lm_head" in k for k in weights)
        assert len(weights) == 2  # just lora_A + lora_B for lm_head


# ---------------------------------------------------------------------------
# Tests: Edge cases and validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_output_exists_raises_file_exists_error(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with pytest.raises(FileExistsError, match="already exists"):
            build_lora_adapter(
                base_model=str(tmp_path),
                adapter_path=str(tmp_path),
                output_path=str(output_dir),
            )

    def test_missing_adapter_config_key_raises(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _DENSE_CONFIG, _DENSE_STATE_DICT)

        adapter_dir.mkdir(parents=True)
        save_file(
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(
                    RANK, HIDDEN
                )
            },
            str(adapter_dir / "adapter_model.safetensors"),
        )
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"lora_alpha": 1})  # Missing "r"
        )

        with pytest.raises(WeightsAdapterError, match=r"missing required key.*'r'"):
            build_lora_adapter(
                base_model=str(model_dir),
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
            )

    def test_output_files_exist(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _DENSE_CONFIG, _DENSE_STATE_DICT)
        _create_adapter(adapter_dir, _make_dense_adapter_weights())

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        assert (output_dir / "adapter_model.safetensors").exists()
        assert (output_dir / "adapter_config.json").exists()

    def test_unembed_tokens_remapped_to_lm_head(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        _create_synthetic_model(model_dir, _DENSE_CONFIG, _DENSE_STATE_DICT)
        _create_adapter(
            adapter_dir,
            {
                "base_model.model.model.unembed_tokens.lora_A.weight": torch.ones(RANK, HIDDEN),
                "base_model.model.model.unembed_tokens.lora_B.weight": torch.ones(100, RANK),
            },
        )

        build_lora_adapter(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
        )

        weights, _ = _load_peft_output(output_dir)
        assert "base_model.model.lm_head.lora_A.weight" in weights
