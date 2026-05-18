from unittest.mock import MagicMock, patch

import pytest

from tinker_cookbook.tokenizer_utils import _get_hf_tokenizer


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear the lru_cache between tests so env var changes take effect."""
    _get_hf_tokenizer.cache_clear()


@patch("transformers.models.auto.tokenization_auto.AutoTokenizer")
def test_kimi_k26_trusts_remote_code_without_env(
    mock_auto: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hardcoded Kimi K2.6 should pass trust_remote_code=True without the env var."""
    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)
    _get_hf_tokenizer("moonshotai/Kimi-K2.6")
    mock_auto.from_pretrained.assert_called_once_with(
        "moonshotai/Kimi-K2.6",
        trust_remote_code=True,
        revision="b5aabbfb20227ed42becbf5541dbffd213942c58",
    )


@patch("transformers.models.auto.tokenization_auto.AutoTokenizer")
def test_no_trust_remote_code_by_default(
    mock_auto: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without env var, generic models should NOT get trust_remote_code."""
    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)
    _get_hf_tokenizer("some-org/some-model")
    mock_auto.from_pretrained.assert_called_once_with(
        "some-org/some-model",
    )


@pytest.mark.parametrize("env_value", ["1", "true", "TRUE", "yes", "Yes"])
@patch("transformers.models.auto.tokenization_auto.AutoTokenizer")
def test_env_var_enables_trust_remote_code(
    mock_auto: MagicMock, monkeypatch: pytest.MonkeyPatch, env_value: str
) -> None:
    """HF_TRUST_REMOTE_CODE env var should enable trust_remote_code for any model."""
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", env_value)
    _get_hf_tokenizer("some-org/some-model")
    mock_auto.from_pretrained.assert_called_once_with(
        "some-org/some-model",
        trust_remote_code=True,
    )


@pytest.mark.parametrize("env_value", ["0", "false", "no", ""])
@patch("transformers.models.auto.tokenization_auto.AutoTokenizer")
def test_env_var_falsy_values_do_not_enable(
    mock_auto: MagicMock, monkeypatch: pytest.MonkeyPatch, env_value: str
) -> None:
    """Falsy values for HF_TRUST_REMOTE_CODE should not enable trust_remote_code."""
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", env_value)
    _get_hf_tokenizer("some-org/some-model")
    mock_auto.from_pretrained.assert_called_once_with(
        "some-org/some-model",
    )
