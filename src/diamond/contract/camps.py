"""The six star points, as vocabulary rather than geometry.

A camp names a seat's home and its target: ``PlayerSpec`` carries one of each,
the encoder orients a position by the acting seat's home camp, and the native
tables are indexed by this enum's order. None of that needs to know where the
holes are, which is why the enum outlives the Python board that used to define
it alongside their coordinates.

The order is the contract: ``x+ y+ z+ x- y- z-`` is the camp index the C++ core
generates its tables in (``native/src/topology_gen.cpp``), and reordering it
silently reinterprets every table indexed by camp.
"""

from __future__ import annotations

from enum import Enum


class Camp(Enum):
    """The six star points, named after the axis that peaks there."""

    X_POS = "x+"
    Y_POS = "y+"
    Z_POS = "z+"
    X_NEG = "x-"
    Y_NEG = "y-"
    Z_NEG = "z-"

    @property
    def opposite(self) -> Camp:
        """The camp across the board: same axis, opposite sign.

        This is exactly the "move to the camp across the board" relation the
        rules need, which is why a seat's target is never stored separately from
        its home.
        """
        axis, sign = self.value[0], self.value[1]
        return Camp(axis + ("-" if sign == "+" else "+"))


CAMP_ORDER: tuple[Camp, ...] = tuple(Camp)
"""Camp enum order, frozen as the integer camp index the native tables use."""

CAMP_INDEX: dict[Camp, int] = {camp: index for index, camp in enumerate(CAMP_ORDER)}

CAMP_SIZE = 10
"""Holes in a camp triangle: the star point plus the hexagon side it stands on."""

PLAYABLE_HOLES = 73

NUM_DIRECTIONS = 6
"""The six lattice directions. The order lives in the core, which walks them."""


__all__ = [
    "CAMP_INDEX",
    "CAMP_ORDER",
    "CAMP_SIZE",
    "NUM_DIRECTIONS",
    "PLAYABLE_HOLES",
    "Camp",
]
