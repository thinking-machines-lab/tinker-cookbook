import asyncio
import inspect
import logging
from typing import Any, cast

import pytest

from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.rl import train
from tinker_cookbook.rl.dataset_utils import EpochCycledRLDataset, cycle_permutation
from tinker_cookbook.rl.sync_final_eval_test import (  # noqa: F401  (evaluated is a fixture)
    LOOPS,
    _Builder,
    _CheckpointManager,
    _config,
    _Logger,
    evaluated,
)
from tinker_cookbook.rl.train import _resolve_end_batch
from tinker_cookbook.rl.types import RLDataset
from tinker_cookbook.supervised.train import _resolve_total_steps


class _RecordingDataset(RLDataset):
    def __init__(self, n: int):
        self.n = n
        self.requested: list[int] = []

    def get_batch(self, index: int) -> list[Any]:
        assert 0 <= index < self.n
        self.requested.append(index)
        return [_Builder()]

    def __len__(self) -> int:
        return self.n


def test_rl_default_is_one_epoch():
    assert inspect.signature(train.Config).parameters["num_epochs"].default == 1
    assert _resolve_end_batch(None, 75) == 75


def test_rl_num_epochs_multiplies_steps():
    assert _resolve_end_batch(None, 75, num_epochs=2) == 150
    assert _resolve_end_batch(100, 75, num_epochs=2) == 100


def test_rl_num_epochs_must_be_positive():
    with pytest.raises(ConfigurationError):
        _resolve_end_batch(None, 75, num_epochs=0)


def test_epoch_cycled_dataset_maps_across_epoch_boundary():
    inner = _RecordingDataset(3)
    ds = EpochCycledRLDataset(inner, num_epochs=2)
    assert len(ds) == 6
    for i in range(6):
        ds.get_batch(i)
    assert inner.requested == [0, 1, 2, 0, 1, 2]


@pytest.mark.usefixtures("evaluated")
@pytest.mark.parametrize("loop", LOOPS)
def test_resume_across_epoch_boundary(loop, tmp_path):
    inner = _RecordingDataset(2)
    logger = _Logger()
    asyncio.run(
        loop(
            start_batch=1,
            end_batch=5,
            num_batches=6,
            config=_config(0, tmp_path),
            training_client=cast(Any, None),
            kl_reference_client=None,
            evaluators=[],
            dataset=EpochCycledRLDataset(inner, num_epochs=3),
            ml_logger=cast(Any, logger),
            tokenizer=cast(Any, None),
            checkpoint_mgr=cast(Any, _CheckpointManager()),
        )
    )
    assert inner.requested == [1, 0, 1, 0]
    assert sorted({step for step, _ in logger.logged}) == [1, 2, 3, 4]


def test_rl_max_steps_beyond_epochs_warns_and_names_num_epochs(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="tinker_cookbook.rl.train"):
        assert _resolve_end_batch(80, 75) == 75
    assert "max_steps=80" in caplog.text
    assert "1 * 75 = 75" in caplog.text
    assert "stop after 75 steps" in caplog.text
    assert "num_epochs" in caplog.text


@pytest.mark.parametrize("max_steps, expected", [(None, 75), (75, 75), (10, 10)])
def test_rl_max_steps_within_dataset_is_silent(
    caplog: pytest.LogCaptureFixture, max_steps: int | None, expected: int
):
    with caplog.at_level(logging.WARNING, logger="tinker_cookbook.rl.train"):
        assert _resolve_end_batch(max_steps, 75) == expected
    assert caplog.text == ""


def test_sl_max_steps_beyond_epochs_warns_and_caps(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="tinker_cookbook.supervised.train"):
        assert _resolve_total_steps(100, 30, 2) == 60
    assert "stop after 60 steps" in caplog.text


@pytest.mark.parametrize("max_steps, expected", [(None, 60), (60, 60), (45, 45)])
def test_sl_max_steps_within_epochs_is_silent(
    caplog: pytest.LogCaptureFixture, max_steps: int | None, expected: int
):
    with caplog.at_level(logging.WARNING, logger="tinker_cookbook.supervised.train"):
        assert _resolve_total_steps(max_steps, 30, 2) == expected
    assert caplog.text == ""


def test_one_epoch_is_identity():
    inner = _RecordingDataset(3)
    ds = EpochCycledRLDataset(inner, num_epochs=1)
    assert len(ds) == 3
    for i in range(3):
        ds.get_batch(i)
    assert inner.requested == [0, 1, 2]
    with pytest.raises(IndexError):
        ds.get_batch(3)


def test_shuffled_epochs_are_permutations_and_resume_exact():
    inner = _RecordingDataset(5)
    ds = EpochCycledRLDataset(inner, num_epochs=3, shuffle_seed=7)
    for i in range(15):
        ds.get_batch(i)
    order = list(inner.requested)
    epochs = [order[k : k + 5] for k in range(0, 15, 5)]
    assert all(sorted(e) == [0, 1, 2, 3, 4] for e in epochs)
    assert len({tuple(e) for e in epochs}) > 1
    assert epochs[1] == list(cycle_permutation(5, 7, 1))

    # A fresh dataset (as after a restart) resolves later indices identically.
    resumed_inner = _RecordingDataset(5)
    resumed = EpochCycledRLDataset(resumed_inner, num_epochs=3, shuffle_seed=7)
    for i in range(7, 15):
        resumed.get_batch(i)
    assert resumed_inner.requested == order[7:]


def test_decorators_stay_on_functions():
    """A class must never pick up a function decorator such as @trace.scope."""
    assert hasattr(train.run_single_evaluation, "__wrapped__")
    for name, obj in vars(train).items():
        assert not inspect.isclass(getattr(obj, "__wrapped__", None)), name
