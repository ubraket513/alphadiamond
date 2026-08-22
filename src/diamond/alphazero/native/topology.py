"""Authoritative board tables, exported from Python for the native extension.

Risk 1 in ``docs/native_selfplay_phase0.md`` is that a native port becomes a
second authority on Diamond's rules.  Topology is the part that is pure data,
so it is *generated* here from :mod:`diamond.game.board` and handed to the
extension at import time rather than transcribed into C++.  Nothing in
``native/`` hard-codes a neighbour, a camp membership or a canonical rotation.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ...game.board import CAMP_SIZE, PLAYABLE_HOLES, Board, Camp, standard_board
from ...game.coordinates import NUM_DIRECTIONS
from ..action_codec import ActionCodec, ActionSpaceSpec
from ..bootstrap.heuristic import pairwise_distance_table
from ..encoder import CanonicalEncoder

CAMP_ORDER: tuple[Camp, ...] = tuple(Camp)
"""Camp enum order, frozen as the integer camp index the extension uses."""

CAMP_INDEX: dict[Camp, int] = {camp: index for index, camp in enumerate(CAMP_ORDER)}


@lru_cache(maxsize=1)
def topology_tables(board: Board | None = None) -> dict[str, Any]:
    """Every fixed table the native backend needs, derived from ``board``."""
    board = board or standard_board()
    if len(board) != PLAYABLE_HOLES:
        raise ValueError(f"native tables assume {PLAYABLE_HOLES} holes, got {len(board)}")

    encoder = CanonicalEncoder(board, ActionCodec(ActionSpaceSpec.diamond73()))

    neighbour = tuple(
        tuple(-1 if nid is None else nid for nid in board.neighbours(pid))
        for pid in range(len(board))
    )
    camp_positions = tuple(board.camp_positions(camp) for camp in CAMP_ORDER)
    for camp, positions in zip(CAMP_ORDER, camp_positions):
        if len(positions) != CAMP_SIZE:
            raise AssertionError(f"camp {camp} has {len(positions)} holes, expected {CAMP_SIZE}")

    mappings = tuple(encoder.position_mapping(camp) for camp in CAMP_ORDER)

    return {
        "board_size": len(board),
        "directions": NUM_DIRECTIONS,
        "neighbour": neighbour,
        "camp_positions": camp_positions,
        "pairwise_distance": pairwise_distance_table(board),
        "physical_to_canonical": tuple(m.physical_to_canonical for m in mappings),
        "canonical_to_physical": tuple(m.canonical_to_physical for m in mappings),
    }


def player_table(players: tuple[Any, ...]) -> tuple[tuple[int, int, int], ...]:
    """``(id, camp index, target camp index)`` per seat, in turn order."""
    return tuple(
        (int(spec.id), CAMP_INDEX[spec.camp], CAMP_INDEX[spec.target_camp])
        for spec in players
    )


__all__ = ["CAMP_INDEX", "CAMP_ORDER", "player_table", "topology_tables"]
