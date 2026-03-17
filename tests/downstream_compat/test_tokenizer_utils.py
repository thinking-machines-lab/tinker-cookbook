"""Downstream compatibility tests for tinker_cookbook.tokenizer_utils.

Validates that the tokenizer registry API remains stable.
"""

import inspect

from tinker_cookbook.tokenizer_utils import (
    Tokenizer,
    get_registered_tokenizer_names,
    get_tokenizer,
    is_tokenizer_registered,
    register_tokenizer,
    unregister_tokenizer,
)


class TestTokenizerRegistryAPI:
    def test_get_tokenizer_signature(self):
        sig = inspect.signature(get_tokenizer)
        params = list(sig.parameters.keys())
        assert "model_name" in params or len(params) >= 1

    def test_register_tokenizer_callable(self):
        assert callable(register_tokenizer)

    def test_unregister_tokenizer_callable(self):
        assert callable(unregister_tokenizer)

    def test_is_tokenizer_registered_callable(self):
        assert callable(is_tokenizer_registered)

    def test_get_registered_tokenizer_names_callable(self):
        assert callable(get_registered_tokenizer_names)

    def test_tokenizer_type_alias_exists(self):
        assert Tokenizer is not None

    def test_register_and_unregister_roundtrip(self):
        name = "__test_downstream_compat_tokenizer__"
        assert not is_tokenizer_registered(name)

        register_tokenizer(name, lambda: None)  # type: ignore[arg-type]
        assert is_tokenizer_registered(name)
        assert name in get_registered_tokenizer_names()

        assert unregister_tokenizer(name) is True
        assert not is_tokenizer_registered(name)
