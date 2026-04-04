import pytest

from tests.helpers import run_recipe

MODULE = "tinker_cookbook.recipes.math_rl.train"


@pytest.mark.integration
def test_math_rl_sync():
    run_recipe(
        MODULE,
        [
            "model_name=Qwen/Qwen3.5-4B",
            "groups_per_batch=8",
            "group_size=4",
            "max_tokens=5",
            "behavior_if_log_dir_exists=delete",
        ],
    )


@pytest.mark.integration
def test_math_rl_async():
    run_recipe(
        MODULE,
        [
            "model_name=Qwen/Qwen3.5-4B",
            "groups_per_batch=8",
            "group_size=4",
            "max_tokens=5",
            "max_steps_off_policy=2",
            "behavior_if_log_dir_exists=delete",
        ],
    )


@pytest.mark.integration
def test_math_rl_stream_minibatch():
    run_recipe(
        MODULE,
        [
            "model_name=Qwen/Qwen3.5-4B",
            "groups_per_batch=8",
            "group_size=4",
            "max_tokens=5",
            "stream_minibatch_config.groups_per_batch=8",
            "stream_minibatch_config.num_minibatches=2",
            "behavior_if_log_dir_exists=delete",
        ],
    )


@pytest.mark.integration
def test_math_rl_reinforce_pp():
    run_recipe(
        MODULE,
        [
            "model_name=Qwen/Qwen3.5-4B",
            "groups_per_batch=8",
            "group_size=4",
            "max_tokens=5",
            "advantage_method=reinforce_pp_baseline",
            "behavior_if_log_dir_exists=delete",
        ],
    )
