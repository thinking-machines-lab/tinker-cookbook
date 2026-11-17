import math
from collections.abc import Generator
from types import SimpleNamespace
from typing import cast

import pytest
import tinker
import torch

from tinker_cookbook.lr_calibration import sweep
from tinker_cookbook.lr_calibration.sweep import RunResult, SweepConfig
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import Tokenizer


class _Done:
    def __init__(self, value: object = None):
        self.value = value

    def __await__(self) -> Generator[None, None, object]:
        return self.value
        yield


class _FakeTrainingClient:
    def forward_backward(self, batch: list[tinker.Datum], loss_fn: str) -> _Done:
        return _Done(SimpleNamespace(loss_fn_outputs=[]))

    def optim_step(self, params: tinker.AdamParams) -> _Done:
        return _Done()


class _FakeServiceClient:
    def create_lora_training_client(self, **kwargs: object) -> _Done:
        return _Done(_FakeTrainingClient())


async def _run(
    monkeypatch: pytest.MonkeyPatch, train_bpb: list[float], eval_bpb: float
) -> RunResult:
    """Run ``_run_one`` against a fake client whose steps report ``train_bpb`` in order."""
    steps = iter(train_bpb)
    monkeypatch.setattr(sweep, "_bits_per_byte", lambda outputs, data, tokenizer: next(steps))

    async def evaluate(tc: object, data: object, tokenizer: object) -> float:
        return eval_bpb

    monkeypatch.setattr(sweep, "_evaluate", evaluate)
    datum = datum_from_model_input_weights(tinker.ModelInput.from_ints([1, 2]), torch.ones(2))
    config = SweepConfig(
        batch_size=1,
        steps=len(train_bpb),
        lora_rank=1,
        n_eval=1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
    )
    return await sweep._run_one(
        cast(tinker.ServiceClient, _FakeServiceClient()),
        "org/model",
        1e-4,
        config,
        [datum] * len(train_bpb),
        [datum],
        cast(Tokenizer, object()),
    )


@pytest.mark.asyncio
async def test_healthy_run_is_evaluated(monkeypatch):
    result = await _run(monkeypatch, [0.5, 0.45, 0.4], eval_bpb=0.42)
    assert not result.diverged and result.eval_bpb == 0.42
    assert result.train_bpb == [0.5, 0.45, 0.4]


@pytest.mark.asyncio
async def test_divergence_in_the_last_steps_is_caught(monkeypatch):
    result = await _run(monkeypatch, [0.5, 3.0, 3.0, 3.0], eval_bpb=0.42)
    assert result.diverged and math.isnan(result.eval_bpb)


@pytest.mark.asyncio
async def test_nonfinite_eval_counts_as_diverged(monkeypatch):
    result = await _run(monkeypatch, [0.5, 0.45], eval_bpb=math.nan)
    assert result.diverged
