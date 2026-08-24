from __future__ import annotations

import pytest

from diamond.contract.board import standard_board
from diamond.contract.state import EMPTY, GameState


@pytest.fixture(scope="session")
def board():
    return standard_board()


def make_state(board, pieces: dict[int, int], current_player_id: int = 1, turn: int = 1):
    """Build an arbitrary state: ``pieces`` maps position id -> player id."""
    occupancy = [EMPTY] * len(board)
    for position_id, player_id in pieces.items():
        occupancy[position_id] = player_id
    return GameState(
        occupancy=tuple(occupancy), current_player_id=current_player_id, turn_number=turn
    )
