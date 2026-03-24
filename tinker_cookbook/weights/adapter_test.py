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
