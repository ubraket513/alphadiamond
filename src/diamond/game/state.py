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


RED = "#D1394F"
YELLOW = "#D9CF1E"
BLUE = "#3E88A8"

# Which camps are in play depends on how many seats there are, because every
# player must aim at the camp directly across the board:
#
#   3 players -- the three corners of triangle "up", 120 degrees apart.  They
#       sit on alternating hexagon sides, so the starting camps stay disjoint.
#   2 players -- one camp and its literal opposite, head to head.  The three
#       "+" camps are *not* opposite each other, so a 2-player match cannot
#       just take two of the 3-player seats.
#
# Colours follow the reference board art, where a camp and its opposite share
# one colour; the head-to-head layout breaks that tie so the two sides stay
# tellable apart.
SEAT_LAYOUTS: dict[int, tuple[tuple[Camp, str], ...]] = {
    2: ((Camp.Z_POS, RED), (Camp.Z_NEG, BLUE)),
    3: ((Camp.Z_POS, RED), (Camp.Y_POS, YELLOW), (Camp.X_POS, BLUE)),
}

SUPPORTED_PLAYER_COUNTS: tuple[int, ...] = tuple(sorted(SEAT_LAYOUTS))

MIN_PLAYERS = SUPPORTED_PLAYER_COUNTS[0]
MAX_PLAYERS = SUPPORTED_PLAYER_COUNTS[-1]


def seat_ids_for(count: int) -> tuple[int, ...]:
    """The seat ids in play for ``count`` players, in board order."""
    if count not in SEAT_LAYOUTS:
        raise ValueError(f"unsupported player count: {count}")
    return tuple(range(1, count + 1))


def build_players(
    count: int = MAX_PLAYERS,
    *,
    order: tuple[int, ...] | None = None,
    ai_seats: tuple[int, ...] = (),
) -> tuple[PlayerSpec, ...]:
    """Assemble the seats for a match.

    ``order`` is the turn order as a permutation of the seat ids; the returned
    tuple is *in* turn order, because that is the sequence
    :func:`next_player_id` walks.  ``ai_seats`` lists the seats the agent
    drives.  Seat ids stay tied to the board position, so reordering turns does
    not change which camp a player sits in or what colour they are.
    """
    seat_ids = seat_ids_for(count)
    layout = SEAT_LAYOUTS[count]

    if order is None:
        order = seat_ids
    else:
        order = tuple(order)
        if sorted(order) != sorted(seat_ids):
            raise ValueError(f"turn order {order} is not a permutation of seats {seat_ids}")

    specs: list[PlayerSpec] = []
    for seat_id in order:
        camp, color = layout[seat_id - 1]
        specs.append(
            PlayerSpec(
                id=seat_id,
                name=f"Player {seat_id}",
                kind=PlayerKind.AI if seat_id in ai_seats else PlayerKind.HUMAN,
                camp=camp,
                target_camp=camp.opposite,
                color=color,
            )
        )
    return tuple(specs)


DEFAULT_PLAYERS: tuple[PlayerSpec, ...] = build_players(3, ai_seats=(3,))


@dataclass(frozen=True, slots=True)
class GameState:
    """A complete authoritative snapshot.

    ``occupancy`` is a tuple of length 73 holding ``EMPTY`` or a player id, so
    a state is cheap to copy, hashable-in-spirit and safe to keep in history.

    ``finish_order`` is the podium as it fills up: player ids in the order they
    got all ten pieces home.  A match keeps running until every place but the
    last is settled, so a 3-player game plays on past first place to decide
    second and third.
    """

    occupancy: tuple[int, ...]
    current_player_id: int
    turn_number: int
    status: GameStatus = GameStatus.IN_PROGRESS
    finish_order: tuple[int, ...] = ()

    @property
    def winner_id(self) -> int | None:
        """The player in first place, once there is one."""
        return self.finish_order[0] if self.finish_order else None

    def place_of(self, player_id: int) -> int | None:
        """1-based finishing place, or ``None`` while the player is still playing."""
        if player_id in self.finish_order:
            return self.finish_order.index(player_id) + 1
        return None

    def has_placed(self, player_id: int) -> bool:
        return player_id in self.finish_order

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

    def placed(self, player_id: int) -> "GameState":
        """Record ``player_id`` as taking the next place on the podium."""
        if player_id in self.finish_order:
            return self
        return replace(self, finish_order=self.finish_order + (player_id,))

    def finished(self) -> "GameState":
        return replace(self, status=GameStatus.FINISHED)


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


def next_player_id(
    players: tuple[PlayerSpec, ...],
    current_id: int,
    *,
    skip: tuple[int, ...] = (),
) -> int:
    """The next seat to act, skipping any player already on the podium.

    Seat order in ``players`` *is* the turn order.  ``skip`` normally comes from
    :attr:`GameState.finish_order`: a player who is home stops taking turns
    while the rest play on for the remaining places.  If everyone is skipped the
    current player is returned unchanged, which only happens once the match is
    already over.
    """
    ids = [p.id for p in players]
    start = ids.index(current_id)
    for offset in range(1, len(ids) + 1):
        candidate = ids[(start + offset) % len(ids)]
        if candidate not in skip:
            return candidate
    return current_id


def player_by_id(players: tuple[PlayerSpec, ...], player_id: int) -> PlayerSpec:
    for spec in players:
        if spec.id == player_id:
            return spec
    raise KeyError(f"unknown player id {player_id}")
