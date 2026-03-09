from tests.smoke.conftest import run_recipe


def test_on_policy_multi_teacher():
    run_recipe(
        "tinker_cookbook.recipes.distillation.on_policy_multi_teacher",
        ["behavior_if_log_dir_exists=delete"],
    )
