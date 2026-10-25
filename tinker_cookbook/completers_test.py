"""Tests for the per-call ``max_tokens`` override on token completers."""

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any

import pytest
import tinker

from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter


@dataclass
class _FakeSequence:
    tokens: list[int]
    logprobs: list[float]
    stop_reason: str = "stop"


@dataclass
class _FakeSampleResult:
    sequences: list[_FakeSequence]


class _RecordingSamplingClient:
    """Stub sampling client that records the sampling params of each request."""

    def __init__(self) -> None:
        self.requests: list[tinker.SamplingParams] = []

    async def sample_async(
        self,
        prompt: tinker.ModelInput,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
    ) -> _FakeSampleResult:
        self.requests.append(sampling_params)
        return _FakeSampleResult(sequences=[_FakeSequence(tokens=[1, 2], logprobs=[-0.1, -0.2])])


def _make_completer(**kwargs: Any) -> tuple[TinkerTokenCompleter, _RecordingSamplingClient]:
    client = _RecordingSamplingClient()
    completer = TinkerTokenCompleter(sampling_client=client, **kwargs)  # type: ignore[arg-type]
    return completer, client


def _prompt(n_tokens: int) -> tinker.ModelInput:
    return tinker.ModelInput.from_ints(list(range(n_tokens)))


class TestTinkerTokenCompleterMaxTokensOverride:
    def test_default_uses_completer_max_tokens(self):
        completer, client = _make_completer(max_tokens=128)
        asyncio.run(completer(_prompt(4), stop=["<stop>"]))
        assert client.requests[0].max_tokens == 128

    def test_override_smaller_wins(self):
        completer, client = _make_completer(max_tokens=128)
        asyncio.run(completer(_prompt(4), stop=["<stop>"], max_tokens=16))
        assert client.requests[0].max_tokens == 16

    def test_override_larger_is_clamped_to_completer_limit(self):
        completer, client = _make_completer(max_tokens=128)
        asyncio.run(completer(_prompt(4), stop=["<stop>"], max_tokens=1024))
        assert client.requests[0].max_tokens == 128

    def test_context_window_still_caps_override(self):
        completer, client = _make_completer(max_tokens=128, context_window=20)
        asyncio.run(completer(_prompt(10), stop=["<stop>"], max_tokens=64))
        assert client.requests[0].max_tokens == 10  # context_window - prompt length

    def test_context_window_overflow_still_raises(self):
        completer, _ = _make_completer(max_tokens=128, context_window=8)
        with pytest.raises(ValueError, match="exceeds context window"):
            asyncio.run(completer(_prompt(10), stop=["<stop>"], max_tokens=4))


class TestTokenCompleterInterface:
    def test_base_call_accepts_max_tokens_kwarg(self):
        with pytest.raises(NotImplementedError):
            asyncio.run(TokenCompleter()(_prompt(1), stop=["<stop>"], max_tokens=4))

    def test_implementations_accept_max_tokens_keyword_only(self):
        from tinker_cookbook.rl.play_w_env import ManualPolicy

        for impl in [TokenCompleter, TinkerTokenCompleter, ManualPolicy]:
            params = inspect.signature(impl.__call__).parameters
            assert "max_tokens" in params, impl
            param = params["max_tokens"]
            assert param.kind is inspect.Parameter.KEYWORD_ONLY, impl
            assert param.default is None, impl
