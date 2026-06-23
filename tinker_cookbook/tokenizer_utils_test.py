from unittest.mock import MagicMock, patch

import pytest

from tinker_cookbook.tokenizer_utils import _get_hf_tokenizer


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear the lru_cache between tests so env var changes take effect."""
    _get_hf_tokenizer.cache_clear()


@pytest.mark.parametrize(
    ("model_name", "revision"),
    [
        ("moonshotai/Kimi-K2-Thinking", "a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55"),
        ("moonshotai/Kimi-K2.5", "2426b45b6af0da48d0dcce71bbce6225e5c73adc"),
        ("moonshotai/Kimi-K2.6", "b5aabbfb20227ed42becbf5541dbffd213942c58"),
    ],
)
@patch("transformers.dynamic_module_utils.get_class_from_dynamic_module")
@patch("transformers.models.auto.tokenization_auto.AutoTokenizer")
def test_kimi_tokenizers_bypass_auto_tokenizer(
    mock_auto: MagicMock,
    mock_get_class: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    revision: str,
) -> None:
    """Hardcoded Kimi models should load TikTokenTokenizer directly."""
    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)
    mock_tokenizer = object()
    mock_tokenizer_cls = MagicMock()
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    mock_get_class.return_value = mock_tokenizer_cls

    assert _get_hf_tokenizer(model_name) is mock_tokenizer

    mock_auto.from_pretrained.assert_not_called()
    mock_get_class.assert_called_once_with(
        "tokenization_kimi.TikTokenTokenizer", model_name, revision=revision
    )
    mock_tokenizer_cls.from_pretrained.assert_called_once_with(model_name, revision=revision)


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
