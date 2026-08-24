"""Player-relative encoding for Diamond states, on the core's geometry.

The rotation that puts the acting seat's home camp at canonical ``z+`` used to
be computed here from cube coordinates. The C++ core generates those tables
(``native/src/topology_gen.cpp``); this reads them. What stays here is the
encoding itself -- which occupancy channel a seat gets, and that finished seats
keep theirs -- because that is the network's input contract rather than geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ..contract.camps import Camp
from ..contract.state import EMPTY, GameState, PlayerSpec, player_by_id
from .action_codec import ActionCodec


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

    def __init__(self, codec: ActionCodec) -> None:
        self.codec = codec

    @lru_cache(maxsize=6)  # noqa: B019 - one encoder per codec, six camps, process lifetime
    def position_mapping(self, home_camp: Camp) -> PositionMapping:
        from .native.topology import canonical_to_physical, physical_to_canonical

        forward = physical_to_canonical(home_camp)
        if len(forward) != self.codec.board_size:
            raise ValueError("the core's board and this action codec differ in size")
        return PositionMapping(forward, canonical_to_physical(home_camp))

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
        rows: list[tuple[float, ...] | None] = [None] * self.codec.board_size
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
        physical_action_ids: tuple[int, ...],
        players: tuple[PlayerSpec, ...],
        current_player_id: int,
    ) -> tuple[bool, ...]:
        mask = [False] * self.codec.action_size
        for physical in physical_action_ids:
            mask[self.to_canonical_action(physical, players, current_player_id)] = True
        return tuple(mask)


__all__ = ["CanonicalEncoder", "EncodedState", "PositionMapping"]
