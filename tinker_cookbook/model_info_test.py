import logging

import pytest

from tinker_cookbook.model_info import (
    get_model_attributes,
    get_recommended_renderer_name,
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


class TestWarnIfRendererNotRecommended:
    def test_no_warning_when_renderer_is_none(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING):
            warn_if_renderer_not_recommended("Qwen/Qwen3-4B-Instruct-2507", None)
        assert caplog.text == ""

    def test_no_warning_when_renderer_is_recommended(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING):
            warn_if_renderer_not_recommended("Qwen/Qwen3-4B-Instruct-2507", "qwen3_instruct")
        assert caplog.text == ""

    def test_warning_when_renderer_not_recommended(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING):
            warn_if_renderer_not_recommended(
                "Qwen/Qwen3-4B-Instruct-2507", "qwen3_disable_thinking"
            )
        assert "not recommended" in caplog.text
        assert "qwen3_disable_thinking" in caplog.text
        assert "qwen3_instruct" in caplog.text

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
