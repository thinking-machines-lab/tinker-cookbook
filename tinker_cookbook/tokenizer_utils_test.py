import sys
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


def test_tml_renderers_source_dir_rolls_back_failed_sys_path_insert(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # Temporary regression coverage for the TML_RENDERERS_SOURCE_DIR import shim.
    # Delete with ensure_tml_renderers_importable once tml-renderers is a normal dependency.
    source_dir = tmp_path / "tml-renderers"
    (source_dir / "tml_renderers").mkdir(parents=True)
    source_str = str(source_dir)
    monkeypatch.setenv("TML_RENDERERS_SOURCE_DIR", source_str)
    monkeypatch.setattr(tokenizer_utils.importlib.util, "find_spec", lambda _name: None)
    original_sys_path = list(sys.path)

    with pytest.raises(ModuleNotFoundError):
        tokenizer_utils.ensure_tml_renderers_importable()

    assert sys.path == original_sys_path


@patch("tinker_cookbook.tokenizer_utils.TmlRenderersTokenizerAdapter")
def test_inkling_uses_tml_renderers_tokenizer_adapter(mock_adapter: MagicMock) -> None:
    tokenizer = tokenizer_utils.get_tokenizer("thinkingmachines/Inkling")

    mock_adapter.assert_called_once_with("thinkingmachines/Inkling")
    assert tokenizer is mock_adapter.return_value
