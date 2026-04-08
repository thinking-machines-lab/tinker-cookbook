import asyncio

from tinker_cookbook.recipes.multiplayer_rl.text_arena.env import (
    OptimalOpponent,
    RandomOpponent,
)
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


def _make_observation(board_rows: list[str], player_symbol: str = "X") -> str:
    """Helper to build a TextArena-style observation for testing."""
    opp = "O" if player_symbol == "X" else "X"
    board = "\n".join(board_rows)
    # Collect available moves from the board
    moves = []
    for row in board_rows:
        for cell in row.split("|"):
            cell = cell.strip()
            if cell.isdigit():
                moves.append(f"'[{cell}]'")
    return (
        f"As Player 1, you will be '{player_symbol}', while your opponent is '{opp}'.\n"
        f"\n[GAME] Current Board:\n\n{board}\n\n"
        f"Available Moves: {', '.join(moves)}"
    )


def test_optimal_blocks_winning_move():
    """Optimal opponent should block when the other player is about to win."""
    opponent = OptimalOpponent()
    # O has two in a row (0, 1), optimal (playing as X) must block at 2
    observation = _make_observation(
        [" O | O | 2 ", "---+---+---", " 3 | X | 5 ", "---+---+---", " 6 | 7 | 8 "],
        player_symbol="X",
    )
    messages: list[Message] = [{"role": "user", "content": observation}]
    response = asyncio.run(opponent(messages))
    assert response["content"] == "[2]", f"Should block at 2, got {response['content']}"


def test_optimal_takes_winning_move():
    """Optimal opponent should take the win when available."""
    opponent = OptimalOpponent()
    # X has 0 and 3 (column 0), playing [6] wins. O threatens nothing.
    observation = _make_observation(
        [" X | O | O ", "---+---+---", " X | O | 5 ", "---+---+---", " 6 | 7 | 8 "],
        player_symbol="X",
    )
    messages: list[Message] = [{"role": "user", "content": observation}]
    response = asyncio.run(opponent(messages))
    assert response["content"] == "[6]", f"Should win at 6, got {response['content']}"


def test_optimal_takes_center_on_empty_board():
    """On an empty board, optimal play should take the center."""
    opponent = OptimalOpponent()
    observation = _make_observation(
        [" 0 | 1 | 2 ", "---+---+---", " 3 | 4 | 5 ", "---+---+---", " 6 | 7 | 8 "],
        player_symbol="O",
    )
    messages: list[Message] = [{"role": "user", "content": observation}]
    response = asyncio.run(opponent(messages))
    # Center (4) or corner (0,2,6,8) are both optimal opening moves
    assert response["content"] in ["[0]", "[2]", "[4]", "[6]", "[8]"]


def test_optimal_vs_optimal_always_draws():
    """Two optimal opponents playing each other should always draw."""

    opponent_o = OptimalOpponent()
    opponent_x = OptimalOpponent()

    # Simulate a full game
    import textarena as ta

    env = ta.make("TicTacToe-v0")
    env.reset(num_players=2)

    for _ in range(9):  # max 9 moves
        player_id, obs = env.get_observation()
        assert isinstance(obs, str)
        messages: list[Message] = [{"role": "user", "content": obs}]
        if player_id == 0:
            response = asyncio.run(opponent_o(messages))
        else:
            response = asyncio.run(opponent_x(messages))
        move = response["content"]
        assert isinstance(move, str)
        done, info = env.step(move)
        if done:
            break

    # Should be a draw
    rewards = env.state.rewards
    assert rewards is not None
    assert rewards[0] == 0 and rewards[1] == 0, f"Expected draw, got {rewards}"
