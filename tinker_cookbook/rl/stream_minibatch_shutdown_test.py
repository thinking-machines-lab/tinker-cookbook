"""Async streaming-minibatch training must train every full batch before shutting down."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder


class _Dataset(RLDataset):
    def __init__(self, groups_per_batch: int):
        self.groups_per_batch = groups_per_batch

    def get_batch(self, index: int) -> list[Any]:
        return [MagicMock(name=f"builder_{index}_{i}") for i in range(self.groups_per_batch)]

    def __len__(self) -> int:
        return 1_000


def _run(tmp_path, monkeypatch: pytest.MonkeyPatch, num_batches: int, groups_per_batch: int):
    trained_steps: list[int] = []

    async def fake_save_checkpoint(**kwargs: Any) -> dict[str, str]:
        return {"sampler_path": "sampler", "state_path": "state"}

    async def fake_rollout(*args: Any, **kwargs: Any) -> MagicMock:
        return MagicMock(name="trajectory_group")

    async def fake_prepare_minibatch(*args: Any, **kwargs: Any):
        return [], {}

    async def fake_full_batch_metrics(training_client, checkpoint_mgr, step, *args, **kwargs):
        trained_steps.append(step - 1)
        # Let rollouts (and the shutdown sentinel) queue up while this step trains.
        await asyncio.sleep(0.01)
        return MagicMock(name="sampling_client"), {}

    monkeypatch.setattr(train.checkpoint_utils, "save_checkpoint_async", fake_save_checkpoint)
    monkeypatch.setattr(train, "do_group_rollout_and_filter_constant_reward", fake_rollout)
    monkeypatch.setattr(train, "prepare_minibatch", fake_prepare_minibatch)
    monkeypatch.setattr(train, "_training_logprobs_from_fwd_bwd", lambda *a, **k: [])
    monkeypatch.setattr(
        train, "compute_full_batch_metrics_and_get_sampling_client", fake_full_batch_metrics
    )
    monkeypatch.setattr(train, "compute_trajectory_metrics", lambda *a, **k: {})
    monkeypatch.setattr(train, "compute_sampling_client_metrics", lambda *a, **k: {})
    monkeypatch.setattr(train, "_maybe_export_rollout_summary_jsonl", lambda **k: None)

    config = train.Config(
        learning_rate=1e-4,
        dataset_builder=MagicMock(spec=RLDatasetBuilder),
        model_name="test-model",
        recipe_name="test",
        max_tokens=16,
        log_path=str(tmp_path),
        eval_every=0,
        async_config=train.AsyncConfig(
            max_steps_off_policy=num_batches, groups_per_batch=groups_per_batch
        ),
        stream_minibatch_config=train.StreamMinibatchConfig(
            groups_per_batch=groups_per_batch, num_minibatches=1
        ),
    )
    future = MagicMock()
    future.result_async = AsyncMock(return_value=MagicMock(metrics={}))
    training_client = MagicMock()
    training_client.forward_backward_async = AsyncMock(return_value=future)
    training_client.optim_step_async = AsyncMock(return_value=future)

    asyncio.run(
        train.do_async_training(
            start_batch=0,
            end_batch=num_batches,
            num_batches=num_batches,
            config=config,
            training_client=training_client,
            kl_reference_client=None,
            evaluators=[],
            dataset=_Dataset(groups_per_batch),
            ml_logger=MagicMock(),
            tokenizer=MagicMock(),
        )
    )
    return trained_steps


@pytest.mark.parametrize("num_batches,groups_per_batch", [(1, 2), (3, 2), (2, 4)])
def test_streaming_trains_every_batch(
    tmp_path, monkeypatch: pytest.MonkeyPatch, num_batches: int, groups_per_batch: int
):
    trained_steps = _run(tmp_path, monkeypatch, num_batches, groups_per_batch)
    assert trained_steps == list(range(num_batches))
