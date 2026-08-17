"""Authoritative, immutable game state and the 3-player match setup."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from .board import CAMP_SIZE, Board, Camp, standard_board
from .move import Move

EMPTY = 0
"""Occupancy sentinel for an empty hole (player ids start at 1)."""


class PlayerKind(Enum):
    HUMAN = "human"
    AI = "ai"


class GameStatus(Enum):
    IN_PROGRESS = "in_progress"
    FINISHED = "finished"


@dataclass(frozen=True, slots=True)
class PlayerSpec:
    """Static configuration of one seat.

    Camps are data, not hard-coded logic, so a different orientation (or a
    different number of seats) only needs a different ``PlayerSpec`` list.
    """

    id: int
    name: str
    kind: PlayerKind
    camp: Camp
    target_camp: Camp
    color: str


# The three active camps sit 120 degrees apart on the star; each player aims at
# the camp directly across the board.  Colours follow the reference board art,
# where a camp and its opposite share one colour.
DEFAULT_PLAYERS: tuple[PlayerSpec, ...] = (
    PlayerSpec(1, "Player 1", PlayerKind.HUMAN, Camp.Z_POS, Camp.Z_NEG, "#D1394F"),
    PlayerSpec(2, "Player 2", PlayerKind.HUMAN, Camp.Y_POS, Camp.Y_NEG, "#D9CF1E"),
    PlayerSpec(3, "Player 3", PlayerKind.AI, Camp.X_POS, Camp.X_NEG, "#3E88A8"),
)


@dataclass(frozen=True, slots=True)
class GameState:
    """A complete authoritative snapshot.

    ``occupancy`` is a tuple of length 121 holding ``EMPTY`` or a player id, so
    a state is cheap to copy, hashable-in-spirit and safe to keep in history.
    """

    occupancy: tuple[int, ...]
    current_player_id: int
    turn_number: int
    status: GameStatus = GameStatus.IN_PROGRESS
    winner_id: int | None = None

    def occupant(self, position_id: int) -> int:
        return self.occupancy[position_id]

    def is_empty(self, position_id: int) -> bool:
        return self.occupancy[position_id] == EMPTY

    def positions_of(self, player_id: int) -> tuple[int, ...]:
        return tuple(i for i, v in enumerate(self.occupancy) if v == player_id)

    def apply(self, move: Move, *, next_player_id: int, advance_turn: bool) -> "GameState":
        """Return a new state with ``move`` applied.  Never mutates ``self``."""
        occupancy = list(self.occupancy)
        if occupancy[move.source] != move.player_id:
            raise ValueError("source hole is not occupied by the moving player")
        if occupancy[move.destination] != EMPTY:
            raise ValueError("destination hole is not empty")
        occupancy[move.source] = EMPTY
        occupancy[move.destination] = move.player_id
        return replace(
            self,
            occupancy=tuple(occupancy),
            current_player_id=next_player_id,
            turn_number=self.turn_number + (1 if advance_turn else 0),
        )

    def finished(self, winner_id: int) -> "GameState":
        return replace(self, status=GameStatus.FINISHED, winner_id=winner_id)


def initial_state(
    players: tuple[PlayerSpec, ...] = DEFAULT_PLAYERS,
    board: Board | None = None,
) -> GameState:
    """Fill every player's home camp with their ten pieces."""
    board = board or standard_board()
    occupancy = [EMPTY] * len(board)
    for spec in players:
        camp = board.camp_positions(spec.camp)
        if len(camp) != CAMP_SIZE:
            raise AssertionError(f"camp {spec.camp} has {len(camp)} holes, expected {CAMP_SIZE}")
        for pid in camp:
            occupancy[pid] = spec.id
    return GameState(
        occupancy=tuple(occupancy),
        current_player_id=players[0].id,
        turn_number=1,
    )


def next_player_id(players: tuple[PlayerSpec, ...], current_id: int) -> int:
    ids = [p.id for p in players]
    return ids[(ids.index(current_id) + 1) % len(ids)]


def player_by_id(players: tuple[PlayerSpec, ...], player_id: int) -> PlayerSpec:
    for spec in players:
        if spec.id == player_id:
            return spec
    raise KeyError(f"unknown player id {player_id}")
