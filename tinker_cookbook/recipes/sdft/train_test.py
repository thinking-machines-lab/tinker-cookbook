# SPDX-License-Identifier: Apache-2.0
"""Test SDFT CLI contracts without model or dataset downloads."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import tinker

from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.distillation import sdft
from tinker_cookbook.recipes.sdft import datasets, train
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.rl.types import RLDataset


@pytest.mark.asyncio
@pytest.mark.parametrize("dataset_name", ["sciknoweval", "toolalpaca"])
async def test_cli_held_out_data_runs_in_rl_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dataset_name: str
) -> None:
    renderer = MagicMock(spec=Renderer)
    renderer.tokenizer = MagicMock()
    renderer.get_stop_sequences.return_value = [0]
    renderer.build_generation_prompt.return_value = tinker.ModelInput.from_ints([10, 11])
    dataset = datasets.SDFTDataset(
        questions=["question 1", "question 2", "question 3"],
        golden_answers=["A", "B", "C"],
        batch_size=2,
        group_size=1,
        renderer=renderer,
    )
    builder_cls = (
        datasets.SciKnowEvalSDFTBuilder
        if dataset_name == "sciknoweval"
        else datasets.ToolAlpacaSDFTBuilder
    )
    monkeypatch.setattr(builder_cls, "__call__", AsyncMock(return_value=(dataset, dataset)))
    policy = AsyncMock(
        spec=TokenCompleter, return_value=TokensWithLogprobs(tokens=[12], maybe_logprobs=[-1.0])
    )

    async def evaluate_held_out(
        config: sdft.Config, training_data: datasets.SDFTDataset, held_out: RLDataset | None
    ) -> None:
        assert training_data is dataset
        assert held_out is not None
        evaluator = RLTestSetEvaluator(held_out, max_tokens=config.max_tokens, num_groups_to_log=0)
        await evaluator.eval_token_completer(policy)

    monkeypatch.setattr(train.sdft, "main", evaluate_held_out)
    await train.cli_main(
        train.CLIConfig(
            dataset=dataset_name,
            renderer_name="qwen3",
            log_path=str(tmp_path / "run"),
            behavior_if_log_dir_exists="resume",
        )
    )
    assert policy.await_count == 3
