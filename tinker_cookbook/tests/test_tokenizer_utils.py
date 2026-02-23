"""Tests for tokenizer_utils.get_tokenizer."""

from unittest.mock import MagicMock, patch


def _fresh_get_tokenizer():
    """Import get_tokenizer with a cleared cache to avoid cross-test pollution."""
    import importlib

    import tinker_cookbook.tokenizer_utils as mod

    importlib.reload(mod)
    return mod.get_tokenizer


@patch("tinker_cookbook.tokenizer_utils.AutoTokenizer", create=True)
def test_trust_remote_code_always_passed(mock_auto):
    """trust_remote_code=True should be passed for every model, not just specific ones."""
    mock_auto = MagicMock()
    with patch("transformers.models.auto.tokenization_auto.AutoTokenizer", mock_auto):
        get_tokenizer = _fresh_get_tokenizer()
        get_tokenizer("some-org/some-model")

    mock_auto.from_pretrained.assert_called_once_with(
        "some-org/some-model", use_fast=True, trust_remote_code=True
    )


@patch("tinker_cookbook.tokenizer_utils.AutoTokenizer", create=True)
def test_kimi_k2_gets_revision_pin(mock_auto):
    """Kimi-K2-Thinking should still get its revision pin alongside trust_remote_code."""
    mock_auto = MagicMock()
    with patch("transformers.models.auto.tokenization_auto.AutoTokenizer", mock_auto):
        get_tokenizer = _fresh_get_tokenizer()
        get_tokenizer("moonshotai/Kimi-K2-Thinking")

    mock_auto.from_pretrained.assert_called_once_with(
        "moonshotai/Kimi-K2-Thinking",
        use_fast=True,
        trust_remote_code=True,
        revision="612681931a8c906ddb349f8ad0f582cb552189cd",
    )


@patch("tinker_cookbook.tokenizer_utils.AutoTokenizer", create=True)
def test_llama3_redirect(mock_auto):
    """Llama-3 models should be redirected but still get trust_remote_code."""
    mock_auto = MagicMock()
    with patch("transformers.models.auto.tokenization_auto.AutoTokenizer", mock_auto):
        get_tokenizer = _fresh_get_tokenizer()
        get_tokenizer("meta-llama/Llama-3-8B-Instruct")

    mock_auto.from_pretrained.assert_called_once_with(
        "thinkingmachineslabinc/meta-llama-3-instruct-tokenizer",
        use_fast=True,
        trust_remote_code=True,
    )


@patch("tinker_cookbook.tokenizer_utils.AutoTokenizer", create=True)
def test_colon_suffix_stripped(mock_auto):
    """Model names with ':suffix' should have the suffix stripped."""
    mock_auto = MagicMock()
    with patch("transformers.models.auto.tokenization_auto.AutoTokenizer", mock_auto):
        get_tokenizer = _fresh_get_tokenizer()
        get_tokenizer("some-org/some-model:latest")

    mock_auto.from_pretrained.assert_called_once_with(
        "some-org/some-model", use_fast=True, trust_remote_code=True
    )
