"""The extension arrives with its geometry, and Python reads it back.

The board used to be generated in Python and handed to the extension at import.
It is generated in C++ now (``native/src/topology_gen.cpp``), and this is the
Python half of that boundary: the module configures itself, and the tables it
exposes are the shape the trainer expects.

Whether the tables are *right* is not asked here -- that is `topology_test`,
which compares them with the frozen corpus in CTest, without an interpreter.
"""

from __future__ import annotations

import pytest

from diamond.alphazero.native import native_module
from diamond.alphazero.native.topology import (
    camp_positions,
    canonical_to_physical,
    neighbour_table,
    pairwise_distances,
    physical_to_canonical,
    topology_tables,
)
from diamond.contract.camps import CAMP_ORDER, CAMP_SIZE, NUM_DIRECTIONS, PLAYABLE_HOLES

pytestmark = pytest.mark.skipif(
    native_module() is None, reason="the native extension is not built"
)


def test_the_extension_configures_itself_without_being_handed_tables() -> None:
    module = native_module()
    assert module is not None
    assert module.is_configured(), (
        "the extension imported unconfigured: it no longer derives its own "
        "geometry, and every caller is one step from a topology error"
    )


def test_the_tables_have_the_shape_the_trainer_indexes_them_by() -> None:
    tables = topology_tables()
    assert tables["board_size"] == PLAYABLE_HOLES
    assert tables["directions"] == NUM_DIRECTIONS
    assert len(neighbour_table()) == PLAYABLE_HOLES
    assert all(len(row) == NUM_DIRECTIONS for row in neighbour_table())
    assert len(pairwise_distances()) == PLAYABLE_HOLES
    for camp in CAMP_ORDER:
        assert len(camp_positions(camp)) == CAMP_SIZE
        assert len(physical_to_canonical(camp)) == PLAYABLE_HOLES


def test_the_canonical_rotation_is_a_bijection_in_both_directions() -> None:
    """The one property a corrupt table copy could satisfy while being wrong."""
    for camp in CAMP_ORDER:
        forward = physical_to_canonical(camp)
        inverse = canonical_to_physical(camp)
        assert sorted(forward) == list(range(PLAYABLE_HOLES)), f"{camp} is not a bijection"
        for physical in range(PLAYABLE_HOLES):
            assert inverse[forward[physical]] == physical, f"{camp} loses position {physical}"


def test_a_camp_lookup_is_keyed_by_the_enum_the_seats_carry() -> None:
    """Camp order is the table index; a reordered enum silently reinterprets it."""
    assert [camp.value for camp in CAMP_ORDER] == ["x+", "y+", "z+", "x-", "y-", "z-"]
