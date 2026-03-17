"""Downstream compatibility tests for tinker_cookbook.completers.

Validates that completer interfaces and types remain stable.
"""

import inspect

from tinker_cookbook.completers import (
    MessageCompleter,
    StopCondition,
    TinkerTokenCompleter,
    TokenCompleter,
    TokensWithLogprobs,
)


class TestTokensWithLogprobs:
    def test_fields(self):
        t = TokensWithLogprobs(tokens=[1, 2, 3], maybe_logprobs=[0.1, 0.2, 0.3])
        assert t.tokens == [1, 2, 3]
        assert t.maybe_logprobs == [0.1, 0.2, 0.3]

    def test_logprobs_property(self):
        t = TokensWithLogprobs(tokens=[1], maybe_logprobs=[0.5])
        assert t.logprobs == [0.5]

    def test_logprobs_raises_when_none(self):
        import pytest

        t = TokensWithLogprobs(tokens=[1], maybe_logprobs=None)
        with pytest.raises(ValueError):
            _ = t.logprobs

    def test_none_logprobs(self):
        t = TokensWithLogprobs(tokens=[1, 2], maybe_logprobs=None)
        assert t.maybe_logprobs is None


class TestTokenCompleter:
    def test_is_callable(self):
        assert callable(TokenCompleter)

    def test_call_is_async(self):
        assert inspect.iscoroutinefunction(TokenCompleter.__call__)

    def test_call_signature(self):
        sig = inspect.signature(TokenCompleter.__call__)
        params = list(sig.parameters.keys())
        assert "model_input" in params
        assert "stop" in params


class TestMessageCompleter:
    def test_is_callable(self):
        assert callable(MessageCompleter)

    def test_call_is_async(self):
        assert inspect.iscoroutinefunction(MessageCompleter.__call__)


class TestTinkerTokenCompleter:
    def test_is_subclass_of_token_completer(self):
        assert issubclass(TinkerTokenCompleter, TokenCompleter)

    def test_has_expected_fields(self):
        # TinkerTokenCompleter is a dataclass with these fields
        annotations = TinkerTokenCompleter.__dataclass_fields__
        assert "sampling_client" in annotations
        assert "max_tokens" in annotations
        assert "temperature" in annotations


class TestStopCondition:
    def test_is_type_alias(self):
        # StopCondition should accept list[str] or list[int]
        val_str: StopCondition = ["<stop>"]
        val_int: StopCondition = [0]
        assert isinstance(val_str, list)
        assert isinstance(val_int, list)
