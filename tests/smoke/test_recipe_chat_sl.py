from tests.smoke.helpers import run_recipe


def test_chat_sl():
    run_recipe(
        "tinker_cookbook.recipes.chat_sl.train",
        ["behavior_if_log_dir_exists=delete"],
    )
