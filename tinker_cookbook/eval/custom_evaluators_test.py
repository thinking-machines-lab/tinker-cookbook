"""Offline import-safety and explicit demonstration controls."""

import asyncio
import importlib
import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace

import pytest
import tinker
from tinker import types

from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.renderers.base import Message

_MODULE = "tinker_cookbook.eval.custom_evaluators"


@pytest.fixture
def external_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def deny(operation: str) -> Callable[..., None]:
        def blocked(*args: object, **kwargs: object) -> None:
            calls.append(operation)
            raise AssertionError(f"Unexpected external operation: {operation}")

        return blocked

    monkeypatch.setattr(tokenizer_utils, "get_tokenizer", deny("get_tokenizer"))
    monkeypatch.setattr(tinker.ServiceClient, "__init__", deny("ServiceClient"))
    monkeypatch.setattr(
        tinker.ServiceClient, "create_sampling_client", deny("create_sampling_client")
    )
    monkeypatch.setattr(tinker.SamplingClient, "sample_async", deny("sample_async"))
    return calls


def _fresh_import(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> ModuleType:
    monkeypatch.delitem(sys.modules, _MODULE, raising=False)
    package = importlib.import_module("tinker_cookbook.eval")
    monkeypatch.delattr(package, "custom_evaluators", raising=False)

    def cleanup_import() -> None:
        sys.modules.pop(_MODULE, None)
        package.__dict__.pop("custom_evaluators", None)

    request.addfinalizer(cleanup_import)
    return importlib.import_module(_MODULE)


def test_import_does_not_initialize_demo(
    monkeypatch: pytest.MonkeyPatch,
    external_calls: list[str],
    request: pytest.FixtureRequest,
) -> None:
    module = _fresh_import(monkeypatch, request)
    assert callable(module.CustomEvaluator)
    assert callable(module.main)
    assert len(module.QA_DATASET) == 3
    assert external_calls == []


def test_import_inside_running_event_loop(
    monkeypatch: pytest.MonkeyPatch,
    external_calls: list[str],
    request: pytest.FixtureRequest,
) -> None:
    async def import_example() -> ModuleType:
        return _fresh_import(monkeypatch, request)

    module = asyncio.run(import_example())
    assert module.grader_fn("The capital is PARIS.", "Paris")
    assert external_calls == []


def test_explicit_main_preserves_demo_flow(
    monkeypatch: pytest.MonkeyPatch,
    external_calls: list[str],
    capsys: pytest.CaptureFixture[str],
    request: pytest.FixtureRequest,
) -> None:
    module = _fresh_import(monkeypatch, request)
    tokenizer = object()
    events: list[tuple[object, ...]] = []
    prompts: list[list[Message]] = []
    samples: list[tuple[types.ModelInput, int, types.SamplingParams]] = []
    responses = ["Paris", "Munich", "Rome"]

    class FakeRenderer:
        def get_stop_sequences(self) -> list[str]:
            return ["<stop>"]

        def build_generation_prompt(self, messages: list[Message]) -> types.ModelInput:
            prompts.append(messages)
            return types.ModelInput.from_ints([len(prompts)])

        def parse_response(self, tokens: list[int]) -> tuple[Message, renderers.ParseTermination]:
            return {
                "role": "assistant",
                "content": responses[tokens[0] - 1],
            }, renderers.ParseTermination.STOP_SEQUENCE

    class FakeSamplingClient:
        async def sample_async(
            self,
            *,
            prompt: types.ModelInput,
            num_samples: int,
            sampling_params: types.SamplingParams,
        ) -> SimpleNamespace:
            samples.append((prompt, num_samples, sampling_params))
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=[len(samples)])])

    class FakeServiceClient:
        def create_sampling_client(self, *, base_model: str) -> FakeSamplingClient:
            events.append(("sampling_client", base_model))
            return FakeSamplingClient()

    def get_tokenizer(model_name: str) -> object:
        events.append(("tokenizer", model_name))
        return tokenizer

    def get_renderer(*, name: str, tokenizer: object) -> FakeRenderer:
        events.append(("renderer", name, tokenizer))
        return FakeRenderer()

    def service_client() -> FakeServiceClient:
        events.append(("service_client",))
        return FakeServiceClient()

    monkeypatch.setattr(module, "get_tokenizer", get_tokenizer)
    monkeypatch.setattr(renderers, "get_renderer", get_renderer)
    monkeypatch.setattr(tinker, "ServiceClient", service_client)

    asyncio.run(module.main())

    assert events == [
        ("tokenizer", "Qwen/Qwen3.5-9B"),
        ("renderer", "qwen3_5_disable_thinking", tokenizer),
        ("service_client",),
        ("sampling_client", "Qwen/Qwen3.5-9B"),
    ]
    assert prompts == [[{"role": "user", "content": row["input"]}] for row in module.QA_DATASET]
    assert len(samples) == 3
    for index, (prompt, num_samples, params) in enumerate(samples, start=1):
        assert prompt.to_ints() == [index]
        assert num_samples == 1
        assert params.max_tokens == 100
        assert params.temperature == 0.7
        assert params.top_p == 1.0
        assert params.stop == ["<stop>"]
    assert capsys.readouterr().out == f"{{'accuracy': {2 / 3}}}\n"
    assert external_calls == []
