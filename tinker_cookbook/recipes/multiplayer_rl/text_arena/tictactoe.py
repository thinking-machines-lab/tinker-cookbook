"""Tic-tac-toe-specific opponents for the TextArena recipe.

Provides random and optimal (minimax) opponents for evaluation, plus a factory
function to construct any opponent type by name.
"""

import random
import re
from typing import Literal

import tinker

from tinker_cookbook.completers import MessageCompleter, StopCondition, TinkerMessageCompleter
from tinker_cookbook.renderers import Message, Renderer

OpponentType = Literal["base_model", "random", "optimal"]


def _parse_available_moves(observation: str) -> list[str]:
    """Parse available moves from a TextArena observation string."""
    return re.findall(r"'(\[\d+\])'", observation)


def _parse_board(observation: str) -> list[str]:
    """Parse the 3x3 board state from a TextArena tic-tac-toe observation.

    The observation may contain multiple "Current Board:" sections (the initial
    board + updates after each move). We parse the last one to get the current state.

    Returns a list of 9 strings: 'O', 'X', or the position number ('0'-'8') if empty.
    """
    board: list[str] = []
    in_board = False
    for line in observation.split("\n"):
        if "Current Board:" in line:
            in_board = True
            board = []  # Reset — take the last board section
            continue
        if in_board and "Available Moves:" in line:
            in_board = False
        if in_board and "|" in line and "---" not in line:
            cells = [c.strip() for c in line.split("|")]
            board.extend(cells)
    return board


class RandomOpponent(MessageCompleter):
    """Opponent that picks a random legal move from the TextArena observation.

    Parses the "Available Moves" line from the observation text and selects
    one uniformly at random. This is useful for cheap evaluation: a trained
    policy should consistently beat a random player.
    """

    async def __call__(self, messages: list[Message]) -> Message:
        observation = messages[-1]["content"]
        assert isinstance(observation, str)
        moves = _parse_available_moves(observation)
        if not moves:
            move = "[0]"
        else:
            move = random.choice(moves)
        return {"role": "assistant", "content": move}


class OptimalOpponent(MessageCompleter):
    """Perfect tic-tac-toe player using minimax.

    Always plays the optimal move. Against optimal play, tic-tac-toe always
    results in a draw. A trained model that consistently draws against this
    opponent has learned perfect play.
    """

    async def __call__(self, messages: list[Message]) -> Message:
        observation = messages[-1]["content"]
        assert isinstance(observation, str)

        board = _parse_board(observation)
        assert len(board) == 9, f"Expected 9 cells, got {len(board)}: {board}"

        # Determine which symbol we are from the observation
        if "you will be 'O'" in observation:
            my_symbol, opp_symbol = "O", "X"
        else:
            my_symbol, opp_symbol = "X", "O"

        move = self._best_move(board, my_symbol, opp_symbol)
        return {"role": "assistant", "content": f"[{move}]"}

    def _best_move(self, board: list[str], my_symbol: str, opp_symbol: str) -> int:
        best_score = -2
        best_pos = -1
        for i in range(9):
            if board[i] not in ("O", "X"):
                board[i] = my_symbol
                score = self._minimax(board, my_symbol, opp_symbol, is_my_turn=False)
                board[i] = str(i)
                if score > best_score:
                    best_score = score
                    best_pos = i
        return best_pos

    def _minimax(self, board: list[str], my_symbol: str, opp_symbol: str, is_my_turn: bool) -> int:
        """Returns +1 for win, -1 for loss, 0 for draw."""
        winner = self._check_winner(board)
        if winner == my_symbol:
            return 1
        if winner == opp_symbol:
            return -1
        empty = [i for i in range(9) if board[i] not in ("O", "X")]
        if not empty:
            return 0

        if is_my_turn:
            best = -2
            for i in empty:
                board[i] = my_symbol
                best = max(best, self._minimax(board, my_symbol, opp_symbol, False))
                board[i] = str(i)
            return best
        else:
            best = 2
            for i in empty:
                board[i] = opp_symbol
                best = min(best, self._minimax(board, my_symbol, opp_symbol, True))
                board[i] = str(i)
            return best

    _WIN_LINES = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),  # rows
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),  # cols
        (0, 4, 8),
        (2, 4, 6),  # diagonals
    ]

    @staticmethod
    def _check_winner(board: list[str]) -> str | None:
        for a, b, c in OptimalOpponent._WIN_LINES:
            if board[a] == board[b] == board[c] and board[a] in ("O", "X"):
                return board[a]
        return None


def make_opponent(
    opponent_type: OpponentType,
    model_name: str,
    renderer: Renderer,
    base_url: str | None = None,
    stop_condition: StopCondition | None = None,
) -> MessageCompleter:
    """Construct an opponent policy by type name.

    - ``"random"``: picks a random legal move (no API calls).
    - ``"optimal"``: perfect minimax player (no API calls).
    - ``"base_model"``: samples from the untrained base model via Tinker.
    """
    if opponent_type == "random":
        return RandomOpponent()
    if opponent_type == "optimal":
        return OptimalOpponent()
    service_client = tinker.ServiceClient(base_url=base_url)
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    return TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=64,
        stop_condition=stop_condition,
    )
