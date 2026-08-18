"""Geometry-derived player-relative encoding for Diamond states."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .action_codec import ActionCodec
from ..game.board import Board, Camp
from ..game.coordinates import Cube
from ..game.move import Move
from ..game.state import EMPTY, GameState, PlayerSpec, player_by_id


@dataclass(frozen=True, slots=True)
class PositionMapping:
    physical_to_canonical: tuple[int, ...]
    canonical_to_physical: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EncodedState:
    """Framework-neutral node features and the player identity they use."""

    node_features: tuple[tuple[float, ...], ...]
    canonical_player_ids: tuple[int, ...]
    physical_to_canonical: tuple[int, ...]
    canonical_to_physical: tuple[int, ...]

    @property
    def feature_count(self) -> int:
        return len(self.node_features[0]) if self.node_features else 0


class CanonicalEncoder:
    """Orient the acting player's home camp at canonical ``z+``.

    Occupancy channels follow the original match turn order, rotated to
    ``self, next[, previous]``. Finished players remain in those same channels
    even after the authoritative turn rotation starts skipping them.
    """

    def __init__(self, board: Board, codec: ActionCodec) -> None:
        if len(board) != codec.board_size:
            raise ValueError("board and action codec sizes differ")
        self.board = board
        self.codec = codec

    @lru_cache(maxsize=6)
    def position_mapping(self, home_camp: Camp) -> PositionMapping:
        physical_to_canonical = tuple(
            self.board.id_of(self._canonical_cube(position.cube, home_camp))
            for position in self.board.positions
        )
        inverse = [0] * len(self.board)
        for physical, canonical in enumerate(physical_to_canonical):
            inverse[canonical] = physical
        return PositionMapping(physical_to_canonical, tuple(inverse))

    @staticmethod
    def _canonical_cube(cube: Cube, home_camp: Camp) -> Cube:
        axis, sign = home_camp.value
        if axis == "x":
            values = (cube.y, cube.z, cube.x)
        elif axis == "y":
            values = (cube.z, cube.x, cube.y)
        else:
            values = (cube.x, cube.y, cube.z)
        factor = 1 if sign == "+" else -1
        return Cube(*(factor * value for value in values))

    @staticmethod
    def canonical_player_ids(
        players: tuple[PlayerSpec, ...], current_player_id: int
    ) -> tuple[int, ...]:
        ids = tuple(player.id for player in players)
        if current_player_id not in ids:
            raise ValueError(f"current player {current_player_id} is not in the match")
        start = ids.index(current_player_id)
        return tuple(ids[(start + offset) % len(ids)] for offset in range(len(ids)))

    def encode(self, state: GameState, players: tuple[PlayerSpec, ...]) -> EncodedState:
        canonical_players = self.canonical_player_ids(players, state.current_player_id)
        home_camp = player_by_id(players, state.current_player_id).camp
        mapping = self.position_mapping(home_camp)
        channel_by_player = {
            player_id: channel for channel, player_id in enumerate(canonical_players)
        }
        finished = tuple(
            1.0 if player_id in state.finish_order else 0.0
            for player_id in canonical_players
        )
        rows: list[tuple[float, ...] | None] = [None] * len(self.board)
        for physical_id, occupant in enumerate(state.occupancy):
            occupancy = [0.0] * len(canonical_players)
            if occupant != EMPTY:
                try:
                    occupancy[channel_by_player[occupant]] = 1.0
                except KeyError as exc:
                    raise ValueError(f"occupancy contains unknown player id {occupant}") from exc
            canonical_id = mapping.physical_to_canonical[physical_id]
            rows[canonical_id] = tuple(occupancy) + finished
        return EncodedState(
            node_features=tuple(row for row in rows if row is not None),
            canonical_player_ids=canonical_players,
            physical_to_canonical=mapping.physical_to_canonical,
            canonical_to_physical=mapping.canonical_to_physical,
        )

    def to_canonical_action(
        self,
        physical_action_id: int,
        players: tuple[PlayerSpec, ...],
        current_player_id: int,
    ) -> int:
        source, destination = self.codec.decode(physical_action_id)
        mapping = self.position_mapping(player_by_id(players, current_player_id).camp)
        return self.codec.encode(
            mapping.physical_to_canonical[source],
            mapping.physical_to_canonical[destination],
        )

    def to_physical_action(
        self,
        canonical_action_id: int,
        players: tuple[PlayerSpec, ...],
        current_player_id: int,
    ) -> int:
        source, destination = self.codec.decode(canonical_action_id)
        mapping = self.position_mapping(player_by_id(players, current_player_id).camp)
        return self.codec.encode(
            mapping.canonical_to_physical[source],
            mapping.canonical_to_physical[destination],
        )

    def legal_action_mask(
        self,
        physical_moves: tuple[Move, ...],
        players: tuple[PlayerSpec, ...],
        current_player_id: int,
    ) -> tuple[bool, ...]:
        mask = [False] * self.codec.action_size
        for move in physical_moves:
            physical = self.codec.encode(move.source, move.destination)
            mask[self.to_canonical_action(physical, players, current_player_id)] = True
        return tuple(mask)


__all__ = ["CanonicalEncoder", "EncodedState", "PositionMapping"]
