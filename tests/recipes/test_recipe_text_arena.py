import pytest

from tests.helpers import run_recipe


@pytest.mark.integration
def test_text_arena():
    run_recipe(
        "tinker_cookbook.recipes.multiplayer_rl.text_arena.train",
        [
            "batch_size=16",
            "num_train_datapoints=128",
        ],
    )


@pytest.mark.integration
def test_text_arena_random_opponent():
    run_recipe(
        "tinker_cookbook.recipes.multiplayer_rl.text_arena.train",
        [
            "batch_size=16",
            "num_train_datapoints=128",
            "test_opponent=random",
        ],
    )
