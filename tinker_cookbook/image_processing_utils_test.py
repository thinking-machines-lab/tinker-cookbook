from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from tinker_cookbook.image_processing_utils import get_image_processor, image_to_data_uri


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


@patch("transformers.models.auto.image_processing_auto.AutoImageProcessor")
def test_inkling_has_no_hf_image_processor(mock_auto: MagicMock) -> None:
    assert get_image_processor("thinkingmachines/Inkling") is None
    mock_auto.from_pretrained.assert_not_called()


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


@patch("tinker_cookbook.image_processing_utils.Image.open")
def test_image_to_data_uri_closes_local_file_handles(mock_open: MagicMock) -> None:
    opened = MagicMock()
    opened.__enter__.return_value.copy.return_value = Image.new("RGB", (1, 1), "white")
    mock_open.return_value = opened

    result = image_to_data_uri("/tmp/example.png")

    assert result.startswith("data:image/jpeg;base64,")
    opened.__exit__.assert_called_once()
