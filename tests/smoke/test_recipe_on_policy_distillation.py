from tests.smoke.conftest import run_recipe


def test_on_policy_distillation():
    run_recipe(
        "tinker_cookbook.recipes.distillation.on_policy_distillation",
        ["behavior_if_log_dir_exists=delete"],
    )
