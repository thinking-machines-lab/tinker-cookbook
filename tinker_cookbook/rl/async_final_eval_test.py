"""Async RL training must always evaluate the weights produced by the last train step."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder

GROUPS_PER_BATCH = 2


class _Dataset(RLDataset):
    def get_batch(self, index: int) -> list[Any]:
        return [MagicMock(name=f"builder_{index}_{i}") for i in range(GROUPS_PER_BATCH)]

    def __len__(self) -> int:
        return 1_000


def _run(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    end_batch: int,
    eval_every: int,
) -> list[int]:
    """Run async training with stubbed Tinker calls and return the evaluated steps."""
    eval_steps: list[int] = []

    async def fake_save_checkpoint(**kwargs: Any) -> dict[str, str]:
        return {"sampler_path": "sampler", "state_path": "state"}

    async def fake_rollout(*args: Any, **kwargs: Any) -> MagicMock:
        return MagicMock(name="trajectory_group")

    async def fake_train_step(config, i_batch, *args: Any, **kwargs: Any):
        return MagicMock(name=f"sampling_client_{i_batch + 1}"), {}

    async def fake_run_evals(evaluators, sampling_client, config, step, **kwargs: Any):
        eval_steps.append(step)
        return {}

    monkeypatch.setattr(train.checkpoint_utils, "save_checkpoint_async", fake_save_checkpoint)
    monkeypatch.setattr(train, "do_group_rollout_and_filter_constant_reward", fake_rollout)
    monkeypatch.setattr(train, "do_train_step_and_get_sampling_client", fake_train_step)
    monkeypatch.setattr(train, "run_evaluations_parallel", fake_run_evals)
    monkeypatch.setattr(train, "compute_sampling_client_metrics", lambda *a, **k: {})
    monkeypatch.setattr(train, "_maybe_export_rollout_summary_jsonl", lambda **k: None)

    config = train.Config(
        learning_rate=1e-4,
        dataset_builder=MagicMock(spec=RLDatasetBuilder),
        model_name="test-model",
        recipe_name="test",
        max_tokens=16,
        log_path=str(tmp_path),
        eval_every=eval_every,
        async_config=train.AsyncConfig(max_steps_off_policy=2, groups_per_batch=GROUPS_PER_BATCH),
    )

    asyncio.run(
        train.do_async_training(
            start_batch=0,
            end_batch=end_batch,
            num_batches=end_batch,
            config=config,
            training_client=MagicMock(),
            kl_reference_client=None,
            evaluators=[MagicMock()],
            dataset=_Dataset(),
            ml_logger=MagicMock(),
            tokenizer=MagicMock(),
        )
    )
    return eval_steps


def test_final_step_off_cadence_is_evaluated(tmp_path, monkeypatch: pytest.MonkeyPatch):
    eval_steps = _run(tmp_path, monkeypatch, end_batch=7, eval_every=5)
    assert eval_steps[-1] == 7
    assert len(eval_steps) == len(set(eval_steps))


def test_final_step_on_cadence_is_evaluated_once(tmp_path, monkeypatch: pytest.MonkeyPatch):
    eval_steps = _run(tmp_path, monkeypatch, end_batch=10, eval_every=5)
    assert eval_steps[-1] == 10
    assert len(eval_steps) == len(set(eval_steps))
