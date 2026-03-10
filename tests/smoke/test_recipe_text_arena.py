from tests.smoke.helpers import run_recipe


def test_text_arena():
    run_recipe(
        "tinker_cookbook.recipes.multiplayer_rl.text_arena.train",
        [
            "batch_size=16",
            "num_train_datapoints=128",
        ],
    )
