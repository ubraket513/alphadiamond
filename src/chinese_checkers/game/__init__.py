"""Pure game engine: board topology, rules, state and history.

This package must stay free of PySide6/QML imports so it can be unit-tested
headlessly and reused by a future AlphaZero training pipeline.
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
    "Board",
    "BoardPosition",
    "Camp",
    "Cube",
    "DEFAULT_PLAYERS",
    "DIRECTIONS",
    "EMPTY",
    "GameSession",
    "GameState",
    "GameStatus",
    "IllegalMoveError",
    "Move",
    "MoveKind",
    "MoveRecord",
    "PlayerKind",
    "PlayerSpec",
    "SCHEMA_VERSION",
    "find_legal_move",
    "find_winner",
    "initial_state",
    "legal_moves",
    "moves_from",
    "next_player_id",
    "player_by_id",
    "standard_board",
]
