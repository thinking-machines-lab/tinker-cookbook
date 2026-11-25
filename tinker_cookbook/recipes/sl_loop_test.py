# SPDX-License-Identifier: Apache-2.0
"""Regression tests for checkpoint progress in the minimal SL loop."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import chz
import datasets
import pytest
import tinker
import torch

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.recipes import sl_loop
from tinker_cookbook.supervised.common import datum_from_model_input_weights


@pytest.mark.parametrize("initial_steps", [0, 1, 3])
def test_resume_after_max_steps_trains_remaining_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, initial_steps: int
) -> None:
    dataset = datasets.Dataset.from_dict(
        {"messages": [[{"role": "assistant", "content": str(i)}] for i in range(3)]}
    )
    monkeypatch.setattr(
        sl_loop.datasets, "load_dataset", lambda _: datasets.DatasetDict({"train": dataset})
    )
    monkeypatch.setattr(sl_loop.ml_log, "setup_logging", MagicMock())
    monkeypatch.setattr(sl_loop, "get_tokenizer", MagicMock())
    monkeypatch.setattr(sl_loop.renderers, "get_renderer", MagicMock())
    monkeypatch.setattr(sl_loop, "compute_bpb", MagicMock(return_value=1.0))

    datum = datum_from_model_input_weights(
        tinker.ModelInput.from_ints([1, 2, 3]), torch.tensor([0.0, 1.0, 1.0]), max_length=3
    )
    convert = MagicMock(return_value=datum)
    monkeypatch.setattr(sl_loop, "conversation_to_datum", convert)
    training_client = MagicMock()
    training_client.forward_backward.return_value.result.return_value = SimpleNamespace(
        loss_fn_outputs=[{"logprobs": tinker.TensorData.from_torch(torch.tensor([-1.0, -1.0]))}]
    )
    training_client.optim_step.return_value.result.return_value = SimpleNamespace(metrics={})
    training_client.save_state = AsyncMock(
        return_value=SimpleNamespace(path="tinker://test/weights/final")
    )
    training_client.save_weights_for_sampler = AsyncMock(
        return_value=SimpleNamespace(path="tinker://test/sampler_weights/final")
    )
    service_client = MagicMock()
    service_client.create_lora_training_client.return_value.result.return_value = training_client
    service_client.create_training_client_from_state_with_optimizer.return_value.result.return_value = training_client
    monkeypatch.setattr(sl_loop.tinker, "ServiceClient", MagicMock(return_value=service_client))

    config = sl_loop.Config(
        log_path=str(tmp_path), batch_size=1, max_steps=initial_steps, save_every=0
    )
    sl_loop.main(config)
    checkpoint = checkpoint_utils.get_last_checkpoint(str(tmp_path))
    assert checkpoint is not None
    assert checkpoint.batch == initial_steps
    assert training_client.optim_step.call_count == initial_steps

    sl_loop.main(chz.replace(config, max_steps=None))
    assert training_client.optim_step.call_count == len(dataset)
    assert [call.args[0] for call in convert.call_args_list] == list(
        dataset.shuffle(seed=0)["messages"]
    )

    # Resuming with an already-reached cap must preserve completed progress.
    sl_loop.main(chz.replace(config, max_steps=1))
    checkpoint = checkpoint_utils.get_last_checkpoint(str(tmp_path))
    assert checkpoint is not None
    assert checkpoint.batch == len(dataset)
    assert training_client.optim_step.call_count == len(dataset)
