from tests.smoke.helpers import run_recipe


def test_on_policy_multi_teacher():
    run_recipe(
        "tinker_cookbook.recipes.distillation.on_policy_multi_teacher",
        [
            "deepmath_groups_per_batch=16",
            "tulu3_groups_per_batch=16",
            "behavior_if_log_dir_exists=delete",
        ],
    )
