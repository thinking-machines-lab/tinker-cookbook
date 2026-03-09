from tests.smoke.conftest import run_recipe


def test_twenty_questions():
    run_recipe(
        "tinker_cookbook.recipes.multiplayer_rl.twenty_questions.train",
        ["behavior_if_log_dir_exists=delete"],
    )
