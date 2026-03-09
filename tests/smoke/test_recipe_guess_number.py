from tests.smoke.conftest import run_recipe


def test_guess_number():
    run_recipe(
        "tinker_cookbook.recipes.multiplayer_rl.guess_number.train",
        ["behavior_if_log_dir_exists=delete"],
    )
