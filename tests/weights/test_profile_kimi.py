"""Profile detection tests against real Kimi K2 and K2.5 HuggingFace configs.

Downloads only config.json and the safetensors index (no weights) to verify
that profile detection, virtual key creation, name remapping, and format
detection produce the expected results against the actual model metadata.

These tests guard against:
- Upstream HF config changes (new fields, renamed model_type)
- Regressions in the detector chain (wrong family, wrong expert layout)
- Virtual key creation producing wrong key names for real models
- Format detection false-positives on non-pack-quantized models
"""

import json

import pytest
from huggingface_hub import hf_hub_download

from tinker_cookbook.weights._merge import detect_merge_profile
from tinker_cookbook.weights._merge_kimi_k25 import (
    _build_kimi_k25_name_remaps,
)
from tinker_cookbook.weights._merge_utils import (
    create_virtual_weight_keys,
    find_quantization_config,
    is_pack_quantized,
)


def _load_real_config(repo_id: str) -> dict:
    path = hf_hub_download(repo_id, "config.json")
    with open(path) as f:
        return json.load(f)


def _load_real_keys(repo_id: str) -> set[str]:
    path = hf_hub_download(repo_id, "model.safetensors.index.json")
    with open(path) as f:
        return set(json.load(f)["weight_map"].keys())


class TestKimiK2RealConfig:
    """Profile detection against real moonshotai/Kimi-K2-Thinking."""

    REPO = "moonshotai/Kimi-K2-Thinking"

    @pytest.fixture(scope="class")
    def config(self) -> dict:
        return _load_real_config(self.REPO)

    @pytest.fixture(scope="class")
    def keys(self) -> set[str]:
        return _load_real_keys(self.REPO)

    def test_model_type(self, config: dict) -> None:
        assert config["model_type"] == "kimi_k2"

    def test_profile_is_default_not_deepseek(self, config: dict, keys: set[str]) -> None:
        """K2 must use default profile (not deepseek, which blocks adapter conversion)."""
        profile = detect_merge_profile(config, keys)
        assert profile.model_family == "default"
        assert profile.expert_layout == "separate"

    def test_is_pack_quantized(self, config: dict) -> None:
        assert is_pack_quantized(config)

    def test_virtual_keys_created_for_packed_experts(self, keys: set[str]) -> None:
        augmented, packed_map = create_virtual_weight_keys(keys)
        # Real model has 384 experts × 3 projections × 60 MoE layers = 69,120
        assert len(packed_map) > 60_000
        # Spot-check a specific virtual key
        assert "model.layers.1.mlp.experts.0.gate_proj.weight" in augmented
        assert "model.layers.1.mlp.experts.0.gate_proj.weight" in packed_map

    def test_standard_key_prefix(self, keys: set[str]) -> None:
        """K2 uses standard model.* prefix (no language_model. wrapper)."""
        assert any(k.startswith("model.layers.") for k in keys)
        assert not any(k.startswith("language_model.") for k in keys)

    def test_has_packed_expert_keys(self, keys: set[str]) -> None:
        assert any(k.endswith(".weight_packed") for k in keys)

    def test_has_bf16_attention_keys(self, keys: set[str]) -> None:
        assert "model.layers.0.self_attn.q_a_proj.weight" in keys


class TestKimiK25RealConfig:
    """Profile detection against real moonshotai/Kimi-K2.5."""

    REPO = "moonshotai/Kimi-K2.5"

    @pytest.fixture(scope="class")
    def config(self) -> dict:
        return _load_real_config(self.REPO)

    @pytest.fixture(scope="class")
    def keys(self) -> set[str]:
        return _load_real_keys(self.REPO)

    def test_model_type(self, config: dict) -> None:
        assert config["model_type"] == "kimi_k25"

    def test_profile_is_kimi_k25(self, config: dict, keys: set[str]) -> None:
        profile = detect_merge_profile(config, keys)
        assert profile.model_family == "kimi_k25"
        assert profile.expert_layout == "separate"
        assert profile.has_language_model_prefix is False

    def test_is_pack_quantized_via_text_config(self, config: dict) -> None:
        """K2.5 nests quantization_config under text_config."""
        assert is_pack_quantized(config)
        # Top-level has no quantization_config
        assert config.get("quantization_config") is None
        # It's in text_config
        quant = find_quantization_config(config)
        assert quant is not None
        assert quant["format"] == "pack-quantized"

    def test_virtual_keys_created_for_packed_experts(self, keys: set[str]) -> None:
        augmented, packed_map = create_virtual_weight_keys(keys)
        assert len(packed_map) > 60_000
        assert "language_model.model.layers.1.mlp.experts.0.gate_proj.weight" in augmented

    def test_language_model_prefix(self, keys: set[str]) -> None:
        """K2.5 uses language_model.model.* prefix (VL model)."""
        assert any(k.startswith("language_model.model.layers.") for k in keys)
        assert not any(k.startswith("model.layers.") for k in keys)

    def test_lm_head_under_language_model(self, keys: set[str]) -> None:
        assert "language_model.lm_head.weight" in keys

    def test_has_vision_keys(self, keys: set[str]) -> None:
        assert any(k.startswith("vision_tower.") for k in keys)

    def test_name_remap_attention(self) -> None:
        """Adapter attention key remaps to language_model.model.* in real model."""
        from tinker_cookbook.weights._merge_utils import remap_adapter_name

        remaps = _build_kimi_k25_name_remaps()
        result = remap_adapter_name(
            "base_model.model.model.layers.1.self_attn.q_a_proj.weight", remaps
        )
        assert result == "language_model.model.layers.1.self_attn.q_a_proj.weight"

    def test_name_remap_unembed(self) -> None:
        from tinker_cookbook.weights._merge_utils import remap_adapter_name

        remaps = _build_kimi_k25_name_remaps()
        result = remap_adapter_name("base_model.model.model.unembed_tokens.weight", remaps)
        assert result == "language_model.lm_head.weight"

    def test_name_remap_expert(self) -> None:
        from tinker_cookbook.weights._merge_utils import remap_adapter_name

        remaps = _build_kimi_k25_name_remaps()
        result = remap_adapter_name("base_model.model.model.layers.1.mlp.experts.w1.weight", remaps)
        assert result == "language_model.model.layers.1.mlp.experts.w1.weight"


class TestNonKimiModelsNotAffected:
    """Verify other model configs are NOT matched by K2/K2.5 detection."""

    @pytest.mark.parametrize(
        "repo_id",
        [
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3.5-4B",
            "deepseek-ai/DeepSeek-V3.1",
        ],
    )
    def test_not_pack_quantized(self, repo_id: str) -> None:
        config = _load_real_config(repo_id)
        assert not is_pack_quantized(config)

    @pytest.mark.parametrize(
        "repo_id,expected_family",
        [
            ("Qwen/Qwen3-8B", "default"),
            ("Qwen/Qwen3.5-4B", "qwen3_5"),
            ("deepseek-ai/DeepSeek-V3.1", "deepseek"),
        ],
    )
    def test_profile_family_unchanged(self, repo_id: str, expected_family: str) -> None:
        """Adding K2/K2.5 support must not change profile detection for other models."""
        config = _load_real_config(repo_id)
        # Use a minimal key set — profile detection for these models
        # doesn't depend on weight keys (uses model_type / architectures).
        profile = detect_merge_profile(config, set())
        assert profile.model_family == expected_family
