from unittest.mock import MagicMock, patch

import pytest

from tinker_cookbook.image_processing_utils import get_image_processor


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear the lru_cache between tests so env var changes take effect."""
    get_image_processor.cache_clear()


@patch("transformers.models.auto.image_processing_auto.AutoImageProcessor")
def test_kimi_k25_trusts_remote_code_without_env(
    mock_auto: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hardcoded Kimi K2.5 should pass trust_remote_code=True without the env var."""
    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)
    get_image_processor("moonshotai/Kimi-K2.5")
    mock_auto.from_pretrained.assert_called_once_with(
        "moonshotai/Kimi-K2.5",
        trust_remote_code=True,
        revision="3367c8d1c68584429fab7faf845a32d5195b6ac1",
    )


@patch("transformers.models.auto.image_processing_auto.AutoImageProcessor")
def test_no_trust_remote_code_by_default(
    mock_auto: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without env var, generic models should NOT get trust_remote_code."""
    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)
    get_image_processor("some-org/some-model")
    mock_auto.from_pretrained.assert_called_once_with(
        "some-org/some-model",
    )


@pytest.mark.parametrize("env_value", ["1", "true", "TRUE", "yes"])
@patch("transformers.models.auto.image_processing_auto.AutoImageProcessor")
def test_env_var_enables_trust_remote_code(
    mock_auto: MagicMock, monkeypatch: pytest.MonkeyPatch, env_value: str
) -> None:
    """HF_TRUST_REMOTE_CODE env var should enable trust_remote_code for any model."""
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", env_value)
    get_image_processor("some-org/some-model")
    mock_auto.from_pretrained.assert_called_once_with(
        "some-org/some-model",
        trust_remote_code=True,
    )
