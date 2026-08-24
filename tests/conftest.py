from __future__ import annotations

from diamond.contract.camps import PLAYABLE_HOLES
from diamond.contract.state import EMPTY, GameState


def make_state(pieces: dict[int, int], current_player_id: int = 1, turn: int = 1):
    """Build an arbitrary state: ``pieces`` maps position id -> player id."""
    occupancy = [EMPTY] * PLAYABLE_HOLES
    for position_id, player_id in pieces.items():
        occupancy[position_id] = player_id
    return GameState(
        occupancy=tuple(occupancy), current_player_id=current_player_id, turn_number=turn
    )
