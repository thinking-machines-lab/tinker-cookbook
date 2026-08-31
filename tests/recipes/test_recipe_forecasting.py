import pytest

from tests.helpers import run_recipe

MODULE = "tinker_cookbook.recipes.forecasting.train"


@pytest.mark.integration
def test_forecasting():
    run_recipe(
        MODULE,
        [
            "model_name=Qwen/Qwen3.5-4B",
            "renderer_name=qwen3_5",
            "groups_per_batch=8",
            "group_size=4",
            "max_tokens=5",
            "max_steps=1",
            "behavior_if_log_dir_exists=delete",
        ],
    )
