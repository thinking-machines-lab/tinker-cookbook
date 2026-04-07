import asyncio

from tinker_cookbook.recipes.multiplayer_rl.text_arena.env import RandomOpponent
from tinker_cookbook.renderers import Message


def test_random_opponent_picks_valid_move():
    opponent = RandomOpponent()
    observation = (
        "[GAME] Current Board:\n"
        " O | 1 | 2\n"
        "---+---+---\n"
        " 3 | X | 5\n"
        "---+---+---\n"
        " 6 | 7 | 8\n"
        "\n"
        "Available Moves: '[1]', '[2]', '[3]', '[5]', '[6]', '[7]', '[8]'"
    )
    messages: list[Message] = [{"role": "user", "content": observation}]
    response = asyncio.run(opponent(messages))
    assert response["role"] == "assistant"
    assert response["content"] in ["[1]", "[2]", "[3]", "[5]", "[6]", "[7]", "[8]"]


def test_random_opponent_single_move():
    opponent = RandomOpponent()
    observation = "Available Moves: '[4]'"
    messages: list[Message] = [{"role": "user", "content": observation}]
    response = asyncio.run(opponent(messages))
    assert response["content"] == "[4]"


def test_random_opponent_fallback():
    opponent = RandomOpponent()
    observation = "No valid moves shown here"
    messages: list[Message] = [{"role": "user", "content": observation}]
    response = asyncio.run(opponent(messages))
    assert response["content"] == "[0]"
