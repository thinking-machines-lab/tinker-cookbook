from tests.helpers import run_recipe


def test_guess_number():
    run_recipe(
        "tinker_cookbook.recipes.multiplayer_rl.guess_number.train",
        [
            "batch_size=8",
            "group_size=2",
        ],
    )
