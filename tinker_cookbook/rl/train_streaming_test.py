import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
import tinker

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl import train
from tinker_cookbook.tokenizer_utils import Tokenizer


@pytest.mark.asyncio
async def test_streaming_step_skips_update_when_all_minibatches_are_filtered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = cast(
        train.Config,
        SimpleNamespace(
            stream_minibatch_config=train.StreamMinibatchConfig(
                groups_per_batch=4,
                num_minibatches=2,
            ),
            num_substeps=1,
            learning_rate=1e-5,
        ),
    )
    trajectory_groups_queue: asyncio.Queue[
        train.WrappedTrajectoryGroup | train._Shutdown | None
    ] = asyncio.Queue()
    for _ in range(4):
        trajectory_groups_queue.put_nowait(None)

    training_client_mock = MagicMock()
    training_client_mock.forward_backward_async = AsyncMock()
    training_client_mock.optim_step_async = AsyncMock()
    training_client = cast(tinker.TrainingClient, training_client_mock)
    checkpoint_mgr = cast(checkpoint_utils.CheckpointManager, MagicMock())
    sampling_client = cast(tinker.SamplingClient, MagicMock())
    save_checkpoint = AsyncMock(return_value=(sampling_client, {"checkpoint/skipped_step": 1}))
    monkeypatch.setattr(train, "save_checkpoint_and_get_sampling_client", save_checkpoint)

    result = await train.do_train_step_streaming_and_get_sampling_client(
        config=config,
        i_batch=7,
        trajectory_groups_queue=trajectory_groups_queue,
        training_client=training_client,
        checkpoint_mgr=checkpoint_mgr,
        kl_reference_client=None,
        tokenizer=cast(Tokenizer, MagicMock()),
    )

    assert result == (sampling_client, {"checkpoint/skipped_step": 1}, [])
    training_client_mock.forward_backward_async.assert_not_awaited()
    training_client_mock.optim_step_async.assert_not_awaited()
    save_checkpoint.assert_awaited_once_with(training_client, checkpoint_mgr, 8)
