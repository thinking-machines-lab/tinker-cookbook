from tests.helpers import run_recipe


def test_shorter():
    run_recipe(
        "tinker_cookbook.recipes.preference.shorter.train",
        ["behavior_if_log_dir_exists=delete"],
    )
