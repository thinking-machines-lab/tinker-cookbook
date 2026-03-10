from tests.helpers import run_recipe


def test_twenty_questions():
    run_recipe(
        "tinker_cookbook.recipes.multiplayer_rl.twenty_questions.train",
        [
            "batch_size=8",
            "group_size=2",
            "num_epochs=1",
            "eval_every=999",
        ],
    )
