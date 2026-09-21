"""Offline tests for supervised checkpoint resume.

These tests drive :func:`tinker_cookbook.supervised.train.main` against an
in-process fake ``TrainingClient``. They do not need ``TINKER_API_KEY`` or
network access.

The production loop pipelines ``submit_ahead`` batches (default 1). Checkpoints
must (1) sit on Tinker's request queue immediately after the batch they
snapshot and (2) record the *next* ``(epoch, batch)`` to execute, matching the
comment in ``train.main``.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import chz
import pytest
import tinker

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder

EVENTS: list[tuple[str, Any]] = []


class SimulatedCrash(Exception):
    """Raised by the fake dataset to interrupt a run after a checkpoint."""


def _make_datum(batch_idx: int) -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1]),
        loss_fn_inputs={
            "weights": tinker.TensorData(data=[1.0], dtype="float32", shape=[1]),
            "target_tokens": tinker.TensorData(data=[100 + batch_idx], dtype="int64", shape=[1]),
        },
    )


class FakeDataset(SupervisedDataset):
    def __init__(self, n_batches: int, crash_at_batch: int | None = None):
        self.n_batches = n_batches
        self.crash_at_batch = crash_at_batch

    def get_batch(self, index: int) -> list[tinker.Datum]:
        if self.crash_at_batch is not None and index == self.crash_at_batch:
            raise SimulatedCrash(f"crash before batch {index}")
        EVENTS.append(("get_batch", index))
        return [_make_datum(index)]

    def __len__(self) -> int:
        return self.n_batches

    def set_epoch(self, seed: int = 0) -> None:
        return None


@chz.chz
class FakeDatasetBuilder(SupervisedDatasetBuilder):
    n_batches: int = 10
    crash_at_batch: int | None = None

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        return FakeDataset(self.n_batches, self.crash_at_batch), None


class FakeTokenizer:
    def decode(self, ids: list[int] | tuple[int, ...], **_: Any) -> str:
        return " ".join(str(i) for i in ids)

    def encode(self, s: str, add_special_tokens: bool = True) -> list[int]:
        return [0]


class FakeResult:
    def __init__(self, **kwargs: Any):
        self.__dict__.update(kwargs)


class FakeFuture:
    def __init__(self, result: FakeResult):
        self._result = result

    async def result_async(self) -> FakeResult:
        return self._result


class FakeTrainingClient:
    model_id = "fake-model"

    async def forward_backward_async(
        self, data: list[tinker.Datum], loss_fn: str, loss_fn_config: Any = None
    ) -> FakeFuture:
        idx = int(data[0].loss_fn_inputs["target_tokens"].data[0]) - 100
        EVENTS.append(("forward_backward", idx))
        n = len(data[0].loss_fn_inputs["target_tokens"].data)
        return FakeFuture(
            FakeResult(
                loss_fn_outputs=[
                    {"logprobs": tinker.TensorData(data=[0.0] * n, dtype="float32", shape=[n])}
                ]
            )
        )

    async def optim_step_async(self, adam_params: tinker.AdamParams) -> FakeFuture:
        EVENTS.append(("optim_step", None))
        return FakeFuture(FakeResult(metrics={}))

    async def save_state_async(self, name: str, ttl_seconds: int | None = None) -> FakeFuture:
        EVENTS.append(("save_state", name))
        return FakeFuture(FakeResult(path=f"tinker://fake-run/weights/{name}"))

    async def save_weights_for_sampler_async(
        self, name: str, ttl_seconds: int | None = None
    ) -> FakeFuture:
        EVENTS.append(("save_weights_for_sampler", name))
        return FakeFuture(FakeResult(path=f"tinker://fake-run/sampler/{name}"))


class FakeServiceClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def create_lora_training_client_async(
        self, base_model: str, rank: int, user_metadata: dict[str, str] | None = None
    ) -> FakeTrainingClient:
        EVENTS.append(("create_lora", base_model))
        return FakeTrainingClient()

    async def create_training_client_from_state_with_optimizer_async(
        self, state_path: str, user_metadata: dict[str, str] | None = None
    ) -> FakeTrainingClient:
        EVENTS.append(("resume_from_state", state_path))
        return FakeTrainingClient()


def _patched_run(config: train.Config) -> None:
    with (
        patch("tinker_cookbook.supervised.train.tinker.ServiceClient", FakeServiceClient),
        patch("tinker_cookbook.supervised.train.get_tokenizer", lambda _name: FakeTokenizer()),
        patch("tinker_cookbook.supervised.train.model_info.warn_if_renderer_not_recommended"),
    ):
        asyncio.run(train.main(config))


def _fwd_batches() -> list[int]:
    return [idx for name, idx in EVENTS if name == "forward_backward"]


def _ops_before_save(save_name: str) -> list[str]:
    """Return event names from the start of EVENTS through the named save_state."""
    names: list[str] = []
    for name, payload in EVENTS:
        names.append(name if name != "save_state" else f"save_state:{payload}")
        if name == "save_state" and payload == save_name:
            return names
    raise AssertionError(f"save_state {save_name!r} not found in {EVENTS}")


@pytest.mark.parametrize(
    ("epoch_idx", "batch_idx", "n_batches", "expected"),
    [
        (0, 4, 10, {"epoch": 0, "batch": 5, "elapsed_tokens": 7}),
        (0, 9, 10, {"epoch": 1, "batch": 0, "elapsed_tokens": 7}),
        (2, 0, 1, {"epoch": 3, "batch": 0, "elapsed_tokens": 7}),
    ],
)
def test_next_supervised_loop_state(
    epoch_idx: int, batch_idx: int, n_batches: int, expected: dict[str, int]
) -> None:
    assert (
        train.next_supervised_loop_state(epoch_idx, batch_idx, n_batches, elapsed_tokens=7)
        == expected
    )


@pytest.mark.parametrize("submit_ahead", [0, 1])
def test_resume_does_not_retrain_the_checkpointed_batch(submit_ahead: int) -> None:
    """Crash after the save_every batch and resume: that batch must not run again.

    ``submit_ahead=0`` is the #935 repro. ``submit_ahead=1`` is the cookbook default.
    """
    global EVENTS
    log_dir = tempfile.mkdtemp(prefix="sl_resume_")
    try:
        EVENTS = []
        config = train.Config(
            log_path=log_dir,
            model_name="fake/model",
            recipe_name="test_supervised_resume",
            renderer_name=None,
            dataset_builder=FakeDatasetBuilder(n_batches=10, crash_at_batch=6),
            num_epochs=1,
            save_every=5,
            eval_every=0,
            infrequent_eval_every=0,
            wandb_project=None,
            submit_ahead=submit_ahead,
        )
        with pytest.raises(SimulatedCrash):
            _patched_run(config)

        record = checkpoint_utils.get_last_checkpoint(log_dir)
        assert record is not None
        assert record.name == "000005"
        assert record.batch == 6
        assert record.epoch == 0

        first_fwd = _fwd_batches()
        assert first_fwd == [0, 1, 2, 3, 4, 5]
        save_ops = _ops_before_save("000005")
        assert save_ops[-3:] == ["forward_backward", "optim_step", "save_state:000005"]
        save_idx = next(
            i
            for i, (name, payload) in enumerate(EVENTS)
            if name == "save_state" and payload == "000005"
        )
        later_fwd = [
            payload for name, payload in EVENTS[save_idx + 1 :] if name == "forward_backward"
        ]
        assert later_fwd == []

        EVENTS = []
        config = train.Config(
            log_path=log_dir,
            model_name="fake/model",
            recipe_name="test_supervised_resume",
            renderer_name=None,
            dataset_builder=FakeDatasetBuilder(n_batches=10, crash_at_batch=None),
            num_epochs=1,
            save_every=5,
            eval_every=0,
            infrequent_eval_every=0,
            wandb_project=None,
            submit_ahead=submit_ahead,
        )
        _patched_run(config)

        resumed_fwd = _fwd_batches()
        assert 5 not in resumed_fwd, f"batch 5 was trained again on resume: {resumed_fwd}"
        assert resumed_fwd == [6, 7, 8, 9]
        assert ("resume_from_state", "tinker://fake-run/weights/000005") in EVENTS
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)


def test_epoch_wrap_is_recorded_on_the_last_batch_of_an_epoch() -> None:
    global EVENTS
    log_dir = tempfile.mkdtemp(prefix="sl_resume_epoch_")
    try:
        EVENTS = []
        config = train.Config(
            log_path=log_dir,
            model_name="fake/model",
            recipe_name="test_supervised_resume",
            renderer_name=None,
            dataset_builder=FakeDatasetBuilder(n_batches=5, crash_at_batch=None),
            num_epochs=2,
            save_every=4,
            eval_every=0,
            infrequent_eval_every=0,
            wandb_project=None,
            submit_ahead=1,
            max_steps=5,
        )
        _patched_run(config)

        records = checkpoint_utils.load_checkpoints_file(log_dir)
        periodic = [r for r in records if r.name == "000004"]
        assert periodic, f"expected periodic checkpoint 000004, got {records}"
        assert periodic[0].epoch == 1
        assert periodic[0].batch == 0
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)


def test_complete_run_does_not_duplicate_batches() -> None:
    global EVENTS
    log_dir = tempfile.mkdtemp(prefix="sl_resume_full_")
    try:
        EVENTS = []
        config = train.Config(
            log_path=log_dir,
            model_name="fake/model",
            recipe_name="test_supervised_resume",
            renderer_name=None,
            dataset_builder=FakeDatasetBuilder(n_batches=8, crash_at_batch=None),
            num_epochs=1,
            save_every=4,
            eval_every=0,
            infrequent_eval_every=0,
            wandb_project=None,
            submit_ahead=1,
        )
        _patched_run(config)
        fwd = _fwd_batches()
        assert fwd == list(range(8))
        assert len(fwd) == len(set(fwd))
        assert Path(log_dir, "checkpoints.jsonl").exists()
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)
