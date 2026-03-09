from tests.smoke.conftest import run_recipe


def test_text_arena():
    run_recipe(
        "tinker_cookbook.recipes.multiplayer_rl.text_arena.train",
        ["behavior_if_log_dir_exists=delete"],
    )
