from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import pytest

from tinker_cookbook import tokenizer_utils
from tinker_cookbook.tokenizer_utils import _get_hf_tokenizer


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear the lru_cache between tests so env var changes take effect."""
    _get_hf_tokenizer.cache_clear()


@pytest.mark.parametrize(
    "model_name,revision",
    [
        ("moonshotai/Kimi-K2-Thinking", "a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55"),
        ("moonshotai/Kimi-K2.5", "2426b45b6af0da48d0dcce71bbce6225e5c73adc"),
        ("moonshotai/Kimi-K2.6", "b5aabbfb20227ed42becbf5541dbffd213942c58"),
    ],
)
@patch("transformers.dynamic_module_utils.get_class_from_dynamic_module")
@patch("transformers.models.auto.tokenization_auto.AutoTokenizer")
def test_kimi_loads_custom_class_directly(
    mock_auto: MagicMock,
    mock_get_class: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    revision: str,
) -> None:
    """Kimi K2 models load the custom TikTokenTokenizer directly at the pinned
    revision, bypassing AutoTokenizer (which fails on some transformers releases)."""
    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)
    _get_hf_tokenizer(model_name)
    mock_get_class.assert_called_once_with(
        "tokenization_kimi.TikTokenTokenizer", model_name, revision=revision
    )
    mock_get_class.return_value.from_pretrained.assert_called_once_with(
        model_name, revision=revision
    )
    mock_auto.from_pretrained.assert_not_called()


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


@patch("tinker_cookbook.tokenizer_utils.TmlRenderersTokenizerAdapter")
def test_inkling_uses_tml_renderers_tokenizer_adapter(mock_adapter: MagicMock) -> None:
    tokenizer = tokenizer_utils.get_tokenizer("thinkingmachines/Inkling")

    mock_adapter.assert_called_once_with("thinkingmachines/Inkling")
    assert tokenizer is mock_adapter.return_value


class _MinimalTmlRendererTokenizer:
    def encode_ordinary(self, text: str) -> Sequence[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class _FullTmlTokenizer(_MinimalTmlRendererTokenizer):
    bos_token = "<bos>"
    eos_token = "<eos>"

    def encode_special(self, text: str) -> int:
        assert text == self.eos_token
        return 42


class _BrokenFullTmlTokenizer(_FullTmlTokenizer):
    def encode_special(self, text: str) -> int:
        raise ValueError("unknown special token")


def test_tml_tokenizer_adapter_exposes_full_tokenizer_special_tokens() -> None:
    adapter = tokenizer_utils.TmlRenderersTokenizerAdapter.from_tokenizer(_FullTmlTokenizer())

    assert adapter.bos_token == "<bos>"
    assert adapter.eos_token == "<eos>"
    assert adapter.eos_token_id == 42


def test_tml_tokenizer_adapter_marks_minimal_tokenizer_special_tokens_unavailable() -> None:
    adapter = tokenizer_utils.TmlRenderersTokenizerAdapter.from_tokenizer(
        _MinimalTmlRendererTokenizer()
    )

    assert adapter.bos_token is None
    assert adapter.eos_token is None
    assert adapter.eos_token_id is None


def test_tml_tokenizer_adapter_does_not_hide_special_token_errors() -> None:
    with pytest.raises(ValueError, match="unknown special token"):
        tokenizer_utils.TmlRenderersTokenizerAdapter.from_tokenizer(_BrokenFullTmlTokenizer())
