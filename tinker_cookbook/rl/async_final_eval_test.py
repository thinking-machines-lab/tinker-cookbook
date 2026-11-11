"""Async RL training must always evaluate the weights produced by the last train step."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.utils import ml_log

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
    ml_logger: ml_log.Logger | None = None,
    delay_eval: bool = False,
    deterministic_clock: bool = False,
    delayed_step: int = 0,
    delay_eval_until_final: bool = False,
    before_delayed_eval_returns: Callable[[], None] | None = None,
) -> list[int]:
    """Run async training with stubbed Tinker calls and return the evaluated steps."""
    eval_steps: list[int] = []
    evaluation_started = asyncio.Event()
    training_ahead = asyncio.Event()
    clock = [100.0]
    if deterministic_clock:
        monkeypatch.setattr(train.time, "monotonic", lambda: clock[0])

    async def fake_save_checkpoint(**kwargs: Any) -> dict[str, str]:
        return {"sampler_path": "sampler", "state_path": "state"}

    async def fake_rollout(*args: Any, **kwargs: Any) -> MagicMock:
        return MagicMock(name="trajectory_group")

    async def fake_train_step(config, i_batch, *args: Any, **kwargs: Any):
        if delay_eval and i_batch == delayed_step:
            await evaluation_started.wait()
        if i_batch == (end_batch - 1 if delay_eval_until_final else delayed_step + 1):
            training_ahead.set()
        clock[0] = 100.0 + (i_batch + 1) * 10.0
        return MagicMock(name=f"sampling_client_{i_batch + 1}"), {
            "train/score": float(i_batch),
            "cost/cumulative_usd": float(i_batch + 1) * 2,
        }

    async def fake_run_evals(evaluators, sampling_client, config, step, **kwargs: Any):
        eval_steps.append(step)
        if step > 0:
            assert sampling_client._mock_name == f"sampling_client_{step}"
        if delay_eval and step == delayed_step:
            evaluation_started.set()
            await training_ahead.wait()
            if before_delayed_eval_returns is not None:
                before_delayed_eval_returns()
        return {"eval/score": float(step)}

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
            ml_logger=ml_logger if ml_logger is not None else MagicMock(),
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


def test_async_training_streams_before_initial_eval_finishes(tmp_path, monkeypatch):
    fake_wandb = MagicMock()
    monkeypatch.setattr(ml_log, "wandb", fake_wandb)
    monkeypatch.setattr(ml_log, "_wandb_available", True)
    logger = ml_log.WandbLogger(project="test")

    def assert_streamed():
        rows = [call.args[0] for call in fake_wandb.log.call_args_list]
        assert [r["training/step"] for r in rows] == [0, 1, 2, 3]
        assert all("eval/score" not in r for r in rows)

    evaluated = _run(
        tmp_path,
        monkeypatch,
        end_batch=4,
        eval_every=1,
        ml_logger=logger,
        delay_eval=True,
        delay_eval_until_final=True,
        before_delayed_eval_returns=assert_streamed,
    )
    rows = [call.args[0] for call in fake_wandb.log.call_args_list]
    eval_rows = [r for r in rows if "eval/score" in r]
    assert [r["eval/snapshot/training_step"] for r in eval_rows] == evaluated
    assert all("training/step" not in r for r in eval_rows)
    assert "eval/snapshot/cumulative_cost_usd" not in eval_rows[0]
    logger.log_metrics({"cost/final": 2.0}, step=4)
    assert fake_wandb.log.call_args_list[-1].args[0] == {"cost/final": 2.0, "training/step": 4}
    assert all(call.kwargs == {"commit": True} for call in fake_wandb.log.call_args_list)
    count = fake_wandb.log.call_count
    logger.close()
    assert fake_wandb.log.call_count == count
    fake_wandb.init.return_value.summary.update.assert_not_called()


def test_delayed_eval_keeps_published_weight_cost_and_time(tmp_path, monkeypatch):
    fake_wandb = MagicMock()
    monkeypatch.setattr(ml_log, "wandb", fake_wandb)
    monkeypatch.setattr(ml_log, "_wandb_available", True)
    logger = ml_log.WandbLogger(project="test")
    _run(
        tmp_path,
        monkeypatch,
        end_batch=4,
        eval_every=1,
        ml_logger=logger,
        delay_eval=True,
        delayed_step=1,
        deterministic_clock=True,
    )
    rows = {
        call.args[0]["eval/snapshot/training_step"]: call.args[0]
        for call in fake_wandb.log.call_args_list
        if "eval/score" in call.args[0]
    }
    snapshot = rows[1]
    assert snapshot["eval/snapshot/training_step"] == 1
    assert snapshot["eval/snapshot/cumulative_cost_usd"] == 2.0
    assert snapshot["eval/snapshot/elapsed_seconds"] == 10.0
    assert snapshot["eval/completed_elapsed_seconds"] >= 20.0
    assert "cost/cumulative_usd" not in snapshot
    training_rows = {
        call.args[0]["training/step"]: call.args[0]
        for call in fake_wandb.log.call_args_list
        if "train/score" in call.args[0]
    }
    assert training_rows[1]["cost/cumulative_usd"] == 4.0
    for axis in ("step", "time", "cost"):
        assert snapshot[f"eval/by_{axis}/eval/score"] == 1.0
    assert rows[4]["eval/snapshot/cumulative_cost_usd"] == 8.0
    assert rows[4]["eval/snapshot/elapsed_seconds"] == 40.0
