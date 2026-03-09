from tests.smoke.conftest import run_recipe


def test_dpo():
    run_recipe(
        "tinker_cookbook.recipes.preference.dpo.train",
        ["behavior_if_log_dir_exists=delete"],
    )
