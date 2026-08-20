import logging

import pytest

from tinker_cookbook.model_info import (
    get_model_attributes,
    get_recommended_renderer_name,
    get_recommended_renderer_names,
    warn_if_renderer_not_recommended,
)


class TestQwen3_6:
    """Qwen3.6 models are architecturally identical to their Qwen3.5
    counterparts (same tokenizer, chat template, and ``qwen3_5`` /
    ``qwen3_5_moe`` model_type) and therefore reuse the qwen3_5 renderer."""

    @pytest.mark.parametrize("size_str", ["27B", "35B-A3B"])
    def test_qwen3_6_uses_qwen3_5_renderer(self, size_str: str):
        assert get_recommended_renderer_name(f"Qwen/Qwen3.6-{size_str}") == "qwen3_5"

    @pytest.mark.parametrize("size_str", ["27B", "35B-A3B"])
    def test_qwen3_6_attributes(self, size_str: str):
        attrs = get_model_attributes(f"Qwen/Qwen3.6-{size_str}")
        assert attrs.organization == "Qwen"
        assert attrs.version_str == "3.6"
        assert attrs.size_str == size_str
        assert attrs.is_chat is True
        assert attrs.is_vl is True
        assert attrs.is_audio_in is False


class TestQwen3_8:
    """Qwen3.8 keeps the Qwen3.5/3.6 tokenizer and preprocessor but has its own
    chat template (reasoning-effort instructions, preserve-thinking default), so
    it gets the dedicated qwen3_8 renderer family."""

    def test_qwen3_8_uses_qwen3_8_renderer(self):
        assert get_recommended_renderer_name("Qwen/Qwen3.8-27B") == "qwen3_8_xhigh_reasoning"

    def test_qwen3_8_recommended_renderers(self):
        names = get_recommended_renderer_names("Qwen/Qwen3.8-27B")
        assert "qwen3_8_disable_thinking" in names
        assert "qwen3_8_medium_reasoning" in names
        assert "qwen3_8_low_reasoning" in names

    def test_qwen3_8_attributes(self):
        attrs = get_model_attributes("Qwen/Qwen3.8-27B")
        assert attrs.organization == "Qwen"
        assert attrs.version_str == "3.8"
        assert attrs.size_str == "27B"
        assert attrs.is_chat is True
        assert attrs.is_vl is True
        assert attrs.is_audio_in is False


class TestNemotron3:
    def test_lightning_uses_ultra_format_renderer(self):
        assert (
            get_recommended_renderer_name("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
            == "nemotron3_ultra"
        )

    def test_lightning_peft_suffix_uses_ultra_format_renderer(self):
        assert (
            get_recommended_renderer_name(
                "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16:peft:262144"
            )
            == "nemotron3_ultra"
        )

    def test_lightning_attributes(self):
        model_name = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
        attrs = get_model_attributes(model_name)
        assert attrs.organization == "nvidia"
        assert attrs.version_str == "3.5"
        assert attrs.size_str == "30B-A3B"
        assert attrs.is_chat is True
        assert attrs.is_vl is False
        assert get_recommended_renderer_names(model_name) == [
            "nemotron3_ultra",
            "nemotron3_ultra_disable_thinking",
            "nemotron3_ultra_preserve_thinking",
        ]

    def test_ultra_uses_nemotron3_ultra_renderer(self):
        assert (
            get_recommended_renderer_name("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16")
            == "nemotron3_ultra"
        )

    def test_ultra_peft_suffix_uses_nemotron3_ultra_renderer(self):
        assert (
            get_recommended_renderer_name(
                "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16:peft:262144"
            )
            == "nemotron3_ultra"
        )

    def test_ultra_attributes(self):
        attrs = get_model_attributes("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16")
        assert attrs.organization == "nvidia"
        assert attrs.version_str == "3"
        assert attrs.size_str == "550B-A55B"
        assert attrs.is_chat is True
        assert attrs.is_vl is False
        assert get_recommended_renderer_names("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16") == [
            "nemotron3_ultra",
            "nemotron3_ultra_disable_thinking",
            "nemotron3_ultra_medium_thinking",
            "nemotron3_ultra_preserve_thinking",
        ]


class TestTmlModels:
    @pytest.mark.parametrize(
        "model_name",
        [
            "thinkingmachines/Inkling",
            "thinkingmachines/Inkling:peft:131072",
        ],
    )
    def test_tml_renderers_models_use_tml_v0_renderer(self, model_name: str):
        assert get_recommended_renderer_name(model_name) == "tml_v0"

    def test_inkling_attributes_route_to_tml_renderers(self):
        attrs = get_model_attributes("thinkingmachines/Inkling")
        assert attrs.is_chat is True
        assert attrs.is_vl is True
        assert attrs.is_audio_in is True
        assert attrs.recommended_renderers == ("tml_v0",)


class TestWarnIfRendererNotRecommended:
    def test_no_warning_when_renderer_is_none(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING):
            warn_if_renderer_not_recommended("Qwen/Qwen3.5-4B", None)
        assert caplog.text == ""

    def test_no_warning_when_renderer_is_recommended(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING):
            warn_if_renderer_not_recommended("Qwen/Qwen3.5-4B", "qwen3_5")
        assert caplog.text == ""

    def test_warning_when_renderer_not_recommended(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING):
            warn_if_renderer_not_recommended("Qwen/Qwen3.5-4B", "qwen3_disable_thinking")
        assert "not recommended" in caplog.text
        assert "qwen3_disable_thinking" in caplog.text
        assert "qwen3_5" in caplog.text

    def test_no_warning_for_unknown_model(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING):
            warn_if_renderer_not_recommended("unknown/model", "qwen3")
        assert caplog.text == ""

    def test_warning_for_thinking_renderer_on_thinking_model_alt(
        self, caplog: pytest.LogCaptureFixture
    ):
        """qwen3_disable_thinking is valid for Qwen3-8B (a thinking model)."""
        with caplog.at_level(logging.WARNING):
            warn_if_renderer_not_recommended("Qwen/Qwen3-8B", "qwen3_disable_thinking")
        assert caplog.text == ""

    def test_warning_for_wrong_family(self, caplog: pytest.LogCaptureFixture):
        """llama3 renderer is not recommended for a Qwen model."""
        with caplog.at_level(logging.WARNING):
            warn_if_renderer_not_recommended("Qwen/Qwen3-8B", "llama3")
        assert "not recommended" in caplog.text
