# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the continual-learning experiment runner."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tinker_cookbook.recipes.sdft import run_continual_learning
from tinker_cookbook.recipes.sdft.run_continual_learning import ExperimentConfig, cli_main


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_checkpoints", [True, False])
async def test_dry_run_does_not_write_checkpoint_map(
    tmp_path: Path, existing_checkpoints: bool
) -> None:
    log_root = tmp_path / "experiment"
    checkpoint_file = log_root / "stage1_checkpoints.json"
    original = json.dumps({"stage1_sft_lr5e-04": "tinker://previous-run/weights/final"})
    if existing_checkpoints:
        log_root.mkdir()
        checkpoint_file.write_text(original)

    await cli_main(
        ExperimentConfig(
            log_root=str(log_root),
            methods="sft",
            learning_rates="5e-4",
            stages="1,2",
            dry_run=True,
        )
    )

    if existing_checkpoints:
        assert checkpoint_file.read_text() == original
    else:
        assert not log_root.exists()


@pytest.mark.asyncio
async def test_training_persists_checkpoints_for_stage_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard the real persistence and resume path while replacing paid training."""
    checkpoint = "tinker://trained-run/state/final"
    train = AsyncMock(return_value=checkpoint)
    monkeypatch.setattr(run_continual_learning, "run_sft_stage", train)

    await cli_main(
        ExperimentConfig(log_root=str(tmp_path), methods="sft", learning_rates="5e-4", stages="1")
    )
    saved = json.loads((tmp_path / "stage1_checkpoints.json").read_text())
    assert saved == {"stage1_sft_lr5e-04": checkpoint}

    await cli_main(
        ExperimentConfig(log_root=str(tmp_path), methods="sft", learning_rates="5e-4", stages="2")
    )
    assert train.await_count == 2
    assert train.await_args is not None
    assert train.await_args.kwargs["load_checkpoint_path"] == checkpoint
