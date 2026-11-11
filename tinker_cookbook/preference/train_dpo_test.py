from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any, cast

import pytest
import tinker
import torch

from tinker_cookbook.preference.train_dpo import _run_dpo_optimizer_step

_DpoLossFn = Callable[
    [list[tinker.Datum], list[torch.Tensor]], tuple[torch.Tensor, dict[str, float]]
]


@dataclass
class _FakeResult:
    metrics: dict[str, float]


class _FakeFuture:
    """Minimal stand-in for ``tinker.APIFuture``: awaitable, resolves inertly."""

    def __init__(self, name: str, events: list[str], result: _FakeResult):
        self.name = name
        self.events = events
        self.result = result

    def __await__(self) -> Generator[Any, None, _FakeResult]:
        return self._resolve().__await__()

    async def _resolve(self) -> _FakeResult:
        self.events.append(f"consume:{self.name}")
        return self.result


class _FakeTrainingClient:
    def __init__(self, events: list[str]):
        self.events = events
        self.fwd_result = _FakeResult(metrics={"dpo_loss": 0.25})
        self.optim_result = _FakeResult(metrics={"learning_rate": 1e-5})
        self.data: list[tinker.Datum] | None = None
        self.loss_fn: _DpoLossFn | None = None
        self.adam_params: tinker.AdamParams | None = None

    def forward_backward_custom(self, data: list[tinker.Datum], loss_fn: _DpoLossFn) -> _FakeFuture:
        self.events.append("enqueue:forward_backward_custom")
        self.data = data
        self.loss_fn = loss_fn
        return _FakeFuture("forward_backward_custom", self.events, self.fwd_result)

    def optim_step(self, adam_params: tinker.AdamParams) -> _FakeFuture:
        self.events.append("enqueue:optim_step")
        self.adam_params = adam_params
        return _FakeFuture("optim_step", self.events, self.optim_result)


def _loss_fn(
    data: list[tinker.Datum], logprobs: list[torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    del data, logprobs
    return torch.tensor(0.0), {}


@pytest.mark.asyncio
async def test_dpo_optimizer_is_enqueued_before_either_result_is_consumed() -> None:
    """Both requests must be submitted before either future is awaited.

    Awaiting the backward result before submitting ``optim_step`` costs a full
    round trip and can push the optimizer into a later worker-pool scheduling
    slot, so the submission order is the behavior worth pinning.
    """
    events: list[str] = []
    client = _FakeTrainingClient(events)
    data: list[tinker.Datum] = []
    adam_params = tinker.AdamParams(learning_rate=1e-5)

    fwd_result, optim_result = await _run_dpo_optimizer_step(
        cast(tinker.TrainingClient, client),
        data,
        _loss_fn,
        adam_params,
    )

    assert events[:2] == ["enqueue:forward_backward_custom", "enqueue:optim_step"]
    assert set(events[2:]) == {"consume:forward_backward_custom", "consume:optim_step"}
    assert fwd_result is client.fwd_result
    assert optim_result is client.optim_result


@pytest.mark.asyncio
async def test_dpo_optimizer_step_forwards_its_arguments_unchanged() -> None:
    """The scheduling change must not alter what is sent to the service."""
    events: list[str] = []
    client = _FakeTrainingClient(events)
    data: list[tinker.Datum] = []
    adam_params = tinker.AdamParams(learning_rate=1e-5)

    await _run_dpo_optimizer_step(
        cast(tinker.TrainingClient, client),
        data,
        _loss_fn,
        adam_params,
    )

    assert client.data is data
    assert client.loss_fn is _loss_fn
    assert client.adam_params is adam_params
