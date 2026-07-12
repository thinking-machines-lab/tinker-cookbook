"""Tests for guess_number token DB capture metadata."""

from typing import Any, cast

from tinker_cookbook.recipes.multiplayer_rl.guess_number.env import GuessNumberEnvGroupBuilder


def test_metadata_game_and_row_id():
    builder = GuessNumberEnvGroupBuilder(answer=17, renderer=cast(Any, None), num_envs=2)
    assert builder.metadata() == {"game": "guess_number", "row_id": "guess_number-17"}
