import pytest

from tests.helpers import run_recipe


@pytest.mark.integration
def test_on_policy_multi_teacher():
    run_recipe(
        "tinker_cookbook.recipes.distillation.on_policy_multi_teacher",
        [
            "deepmath_groups_per_batch=4",
            "tulu3_groups_per_batch=4",
            "behavior_if_log_dir_exists=delete",
        ],
        max_steps=1,
    )
