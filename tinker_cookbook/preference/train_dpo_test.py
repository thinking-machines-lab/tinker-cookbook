from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

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
    def __init__(self, name: str, events: list[str], result: _FakeResult):
        self.name = name
        self.events = events
        self.result = result

    async def result_async(self) -> _FakeResult:
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

    async def forward_backward_custom_async(
        self, data: list[tinker.Datum], loss_fn: _DpoLossFn
    ) -> _FakeFuture:
        self.events.append("enqueue:forward_backward_custom")
        self.data = data
        self.loss_fn = loss_fn
        return _FakeFuture("forward_backward_custom", self.events, self.fwd_result)

    async def optim_step_async(self, adam_params: tinker.AdamParams) -> _FakeFuture:
        self.events.append("enqueue:optim_step")
        self.adam_params = adam_params
        return _FakeFuture("optim_step", self.events, self.optim_result)


@pytest.mark.asyncio
async def test_dpo_optimizer_is_enqueued_before_either_result_is_consumed() -> None:
    events: list[str] = []
    client = _FakeTrainingClient(events)
    data: list[tinker.Datum] = []

    def loss_fn(
        data: list[tinker.Datum], logprobs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del data, logprobs
        return torch.tensor(0.0), {}

    adam_params = tinker.AdamParams(learning_rate=1e-5)
    fwd_result, optim_result = await _run_dpo_optimizer_step(
        cast(tinker.TrainingClient, client),
        data,
        loss_fn,
        adam_params,
    )

    assert events[:2] == ["enqueue:forward_backward_custom", "enqueue:optim_step"]
    assert set(events[2:]) == {"consume:forward_backward_custom", "consume:optim_step"}
    assert fwd_result is client.fwd_result
    assert optim_result is client.optim_result
    assert client.data is data
    assert client.loss_fn is loss_fn
    assert client.adam_params is adam_params
