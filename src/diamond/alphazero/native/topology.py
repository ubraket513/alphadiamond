"""The board tables, read from the core that generates them.

This module used to *derive* the tables from ``diamond.contract.board`` and hand
them to the extension at import. The direction is reversed:
``native/src/topology_gen.cpp`` performs that construction now, the extension
configures itself, and this is where Python reads the result.

Everything geometric in the trainer comes through here -- the encoder's
canonical rotation, the bootstrap prior's distances, the opening's camp
positions -- so there is exactly one answer to "where are the holes", and it is
the core's.

The tables are small and fixed (73 holes), so they are read once and cached.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ...contract.camps import CAMP_INDEX, CAMP_ORDER, NUM_DIRECTIONS, PLAYABLE_HOLES

__all__ = [
    "CAMP_INDEX",
    "CAMP_ORDER",
    "camp_positions",
    "canonical_to_physical",
    "neighbour_table",
    "pairwise_distances",
    "physical_to_canonical",
    "player_table",
    "topology_tables",
]


@lru_cache(maxsize=1)
def topology_tables() -> dict[str, Any]:
    """Every fixed table, as the core generated it."""
    from . import require_native

    tables = require_native().export_tables()
    if tables["board_size"] != PLAYABLE_HOLES:
        raise ValueError(
            f"the core reports a {tables['board_size']}-hole board; this package "
            f"describes {PLAYABLE_HOLES}"
        )
    if tables["directions"] != NUM_DIRECTIONS:
        raise ValueError("the core reports a different number of lattice directions")
    return {
        "board_size": tables["board_size"],
        "directions": tables["directions"],
        "neighbour": tuple(tuple(row) for row in tables["neighbour"]),
        "camp_positions": tuple(tuple(row) for row in tables["camp_positions"]),
        "pairwise_distance": tuple(tuple(row) for row in tables["pairwise_distance"]),
        "physical_to_canonical": tuple(tuple(row) for row in tables["physical_to_canonical"]),
        "canonical_to_physical": tuple(tuple(row) for row in tables["canonical_to_physical"]),
    }


def neighbour_table() -> tuple[tuple[int, ...], ...]:
    """``neighbour[position][direction]``; ``-1`` runs off the board."""
    return topology_tables()["neighbour"]


def camp_positions(camp) -> tuple[int, ...]:
    """The ten holes of one camp triangle, in position-id order."""
    return topology_tables()["camp_positions"][CAMP_INDEX[camp]]


def pairwise_distances() -> tuple[tuple[int, ...], ...]:
    """All-pairs neighbour-graph distance, the bootstrap prior's metric."""
    return topology_tables()["pairwise_distance"]


def physical_to_canonical(camp) -> tuple[int, ...]:
    """Position ids as seen by a seat whose home camp is ``camp``."""
    return topology_tables()["physical_to_canonical"][CAMP_INDEX[camp]]


def canonical_to_physical(camp) -> tuple[int, ...]:
    """The inverse rotation, back to board coordinates."""
    return topology_tables()["canonical_to_physical"][CAMP_INDEX[camp]]


def player_table(players: tuple[Any, ...]) -> tuple[tuple[int, int, int], ...]:
    """``(id, camp index, target camp index)`` per seat, in turn order."""
    return tuple(
        (int(spec.id), CAMP_INDEX[spec.camp], CAMP_INDEX[spec.target_camp])
        for spec in players
    )
