"""What a Diamond position *is*, independent of who applies the rules.

The seats, a move and a position. No rules and no geometry: the C++ core
decides which moves are legal, whose turn is next and who has won, and it
generates the board (``native/src/topology_gen.cpp``). Python reads the tables
through :mod:`diamond.alphazero.native.topology`.

This package exists so that deleting the Python engine did not mean rewriting
every module that merely describes a position. The engine is gone; these types
stayed exactly where the callers already pointed.
"""

from __future__ import annotations

from .camps import CAMP_INDEX, CAMP_ORDER, CAMP_SIZE, NUM_DIRECTIONS, PLAYABLE_HOLES, Camp
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
    "CAMP_INDEX",
    "CAMP_ORDER",
    "CAMP_SIZE",
    "DEFAULT_PLAYERS",
    "EMPTY",
    "NUM_DIRECTIONS",
    "PLAYABLE_HOLES",
    "SEAT_LAYOUTS",
    "Camp",
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
]
