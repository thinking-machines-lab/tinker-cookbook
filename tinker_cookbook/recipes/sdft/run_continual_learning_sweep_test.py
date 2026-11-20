# SPDX-License-Identifier: Apache-2.0
"""Run and checkpoint identity in continual-learning sweeps."""

import json
from pathlib import Path

import pytest

from tinker_cookbook.recipes.sdft import run_continual_learning
from tinker_cookbook.recipes.sdft.run_continual_learning import ExperimentConfig, cli_main


@pytest.mark.asyncio
async def test_learning_rate_sweep_keeps_distinct_runs_and_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[tuple[str, float, str, str | None]] = []
    checkpoints: dict[float, str] = {}

    async def train_stage(
        config: ExperimentConfig,
        task: str,
        learning_rate: float,
        renderer_name: str,
        log_path: str,
        wandb_name: str | None = None,
        session_name: str | None = None,
        load_checkpoint_path: str | None = None,
    ) -> str:
        runs.append((task, learning_rate, log_path, load_checkpoint_path))
        checkpoint = f"tinker://test-{learning_rate}-{task}/weights/final"
        if task == "tooluse":
            checkpoints[learning_rate] = checkpoint
        return checkpoint

    monkeypatch.setattr(run_continual_learning, "run_sft_stage", train_stage)
    await cli_main(
        ExperimentConfig(
            log_root=str(tmp_path), methods="sft", learning_rates="1.1e-3,1.2e-3", stages="1,2"
        )
    )

    assert len({log_path for _, _, log_path, _ in runs}) == 4
    for task, learning_rate, log_path, loaded in runs:
        result = json.loads((Path(log_path) / "run_result.json").read_text())
        assert result["learning_rate"] == learning_rate
        if task == "science":
            assert loaded == checkpoints[learning_rate]
    saved = json.loads((tmp_path / "stage1_checkpoints.json").read_text())
    assert set(saved.values()) == set(checkpoints.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("learning_rate, slug", [(5e-4, "5e-04"), (1e-3, "1e-03")])
async def test_default_learning_rates_keep_existing_run_ids(
    learning_rate: float, slug: str
) -> None:
    result = await run_continual_learning.run_single(
        ExperimentConfig(dry_run=True), "sft", learning_rate, 1, "qwen3", {}
    )
    assert result["run_id"] == f"stage1_sft_lr{slug}"
