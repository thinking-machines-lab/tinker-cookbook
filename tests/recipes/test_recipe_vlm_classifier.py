import pytest

from tests.helpers import run_recipe


@pytest.mark.integration
def test_vlm_classifier():
    run_recipe(
        "tinker_cookbook.recipes.vlm_classifier.train",
        [
            "experiment_dir=/tmp/tinker-smoke-test/vlm_classifier",
            "model_name=Qwen/Qwen3.6-35B-A3B",
            "renderer_name=qwen3_5_disable_thinking",
            "batch_size=16",
            "num_epochs=1",
            "n_eval=16",
            "behavior_if_log_dir_exists=delete",
        ],
    )
