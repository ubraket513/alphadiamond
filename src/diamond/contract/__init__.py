"""What a Diamond position *is*, independent of who applies the rules.

The board's geometry, the seats, a move and a position. No rules: nothing here
decides which moves are legal, whose turn is next or who has won -- the C++ core
decides all of that, and ``diamond.game`` keeps the Python reading of it that
the golden corpus was generated from.

**This package exists so that deleting the Python engine does not mean
rewriting every module that merely describes a position.** Those two things
were the same import before, which is why the retirement ledger had to count
them separately (see docs/architecture/retiring_the_python_engine.md). They are
now separate packages, and the dependency points away from the implementation
being retired: ``diamond.game`` imports from here, not the other way round.

The geometry is still generated in Python -- ``standard_board()`` is what
produces the topology tables the extension is configured with and the
deployment artifact ships. That is the next thing to move, not something this
package settles.
"""

from __future__ import annotations

from .board import CAMP_SIZE, PLAYABLE_HOLES, Board, BoardPosition, Camp, standard_board
from .coordinates import DIRECTIONS, NUM_DIRECTIONS, Cube
from .move import IllegalMoveError, Move, MoveKind
from .state import (
    DEFAULT_PLAYERS,
    EMPTY,
    SEAT_LAYOUTS,
    GameState,
    GameStatus,
    PlayerKind,
    PlayerSpec,
    build_players,
    initial_state,
    player_by_id,
    seat_ids_for,
)

__all__ = [
    "CAMP_SIZE",
    "DEFAULT_PLAYERS",
    "DIRECTIONS",
    "EMPTY",
    "NUM_DIRECTIONS",
    "PLAYABLE_HOLES",
    "SEAT_LAYOUTS",
    "Board",
    "BoardPosition",
    "Camp",
    "Cube",
    "GameState",
    "GameStatus",
    "IllegalMoveError",
    "Move",
    "MoveKind",
    "PlayerKind",
    "PlayerSpec",
    "build_players",
    "initial_state",
    "player_by_id",
    "seat_ids_for",
    "standard_board",
]
