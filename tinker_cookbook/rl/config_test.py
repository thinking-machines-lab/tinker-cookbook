import chz

from tinker_cookbook.rl.train import Config
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder


class _DummyDataset(RLDataset):
    def get_batch(self, index):
        return []

    def __len__(self):
        return 0


@chz.chz
class _DummyDatasetBuilder(RLDatasetBuilder):
    async def __call__(self):
        return _DummyDataset(), None


def test_config_log_path_preserves_cloud_uri():
    config = Config(
        learning_rate=1e-5,
        dataset_builder=_DummyDatasetBuilder(),
        model_name="Qwen/Qwen3-8B",
        recipe_name="test",
        max_tokens=16,
        log_path="gs://bucket/path/to/run",
    )

    assert config.log_path == "gs://bucket/path/to/run"
