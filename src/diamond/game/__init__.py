"""Pure game engine: board topology, rules, state and history.

**No longer the source of truth.** The C++ core under ``native/`` is the
authority for rules, encoding, search and self-play; this package is kept as
the oracle that generates ``tests/golden/`` and as the other half of the bridge
parity gates, and it is scheduled for deletion once neither job needs it. New
production code should reach for ``diamond.alphazero.native`` instead --
``tests/test_engine_retirement.py`` fails if it does not.

See ``docs/architecture/retiring_the_python_engine.md``.

The package stays independent of any GUI toolkit so it can be tested headlessly.
"""

from .board import Board, BoardPosition, Camp, standard_board
from .coordinates import DIRECTIONS, Cube
from .history import MoveRecord
from .move import Move, MoveKind
from .rules import IllegalMoveError, find_legal_move, find_winner, legal_moves, moves_from
from .session import SCHEMA_VERSION, GameSession
from .state import (
    DEFAULT_PLAYERS,
    EMPTY,
    GameState,
    GameStatus,
    PlayerKind,
    PlayerSpec,
    initial_state,
    next_player_id,
    player_by_id,
)

__all__ = [
    "DEFAULT_PLAYERS",
    "DIRECTIONS",
    "EMPTY",
    "SCHEMA_VERSION",
    "Board",
    "BoardPosition",
    "Camp",
    "Cube",
    "GameSession",
    "GameState",
    "GameStatus",
    "IllegalMoveError",
    "Move",
    "MoveKind",
    "MoveRecord",
    "PlayerKind",
    "PlayerSpec",
    "find_legal_move",
    "find_winner",
    "initial_state",
    "legal_moves",
    "moves_from",
    "next_player_id",
    "player_by_id",
    "standard_board",
]
