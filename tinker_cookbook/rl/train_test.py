from concurrent.futures import ThreadPoolExecutor

import chz
import pytest

from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder


class _EmptyDataset(RLDataset):
    def get_batch(self, index):
        return []

    def __len__(self):
        return 0


@chz.chz
class _EmptyDatasetBuilder(RLDatasetBuilder):
    async def __call__(self):
        return _EmptyDataset(), None


def _config(**kwargs):
    return train.Config(
        learning_rate=1e-5,
        dataset_builder=_EmptyDatasetBuilder(),
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        max_tokens=8,
        log_path="/tmp/tinker-train-test",
        **kwargs,
    )


def test_validate_rollout_execution_config_rejects_nonpositive_sample_cap():
    with pytest.raises(ConfigurationError, match="max_concurrent_samples must be positive"):
        train._validate_rollout_execution_config(_config(max_concurrent_samples=0), None)


def test_validate_rollout_execution_config_rejects_executor_with_sample_cap():
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        with pytest.raises(ConfigurationError, match="not supported with rollout_executor"):
            train._validate_rollout_execution_config(_config(max_concurrent_samples=1), executor)
    finally:
        executor.shutdown(wait=False)


def test_validate_rollout_execution_config_accepts_executor_without_sample_cap():
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        train._validate_rollout_execution_config(_config(), executor)
    finally:
        executor.shutdown(wait=False)
