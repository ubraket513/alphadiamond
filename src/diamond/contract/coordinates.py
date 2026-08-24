"""Logical lattice coordinates for the Diamond star board.

The board is a triangular lattice, so we use cube coordinates ``(x, y, z)``
constrained by ``x + y + z == 0``.  Cube coordinates make the six lattice
directions symmetric and make the six star camps expressible as simple
inequalities on a single axis (see :mod:`diamond.game.board`).

Nothing in this module knows about pixels.  ``unit_xy`` returns an abstract
unit-space position that the renderer is free to scale; game logic never uses
it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SQRT3_OVER_2 = math.sqrt(3.0) / 2.0


@dataclass(frozen=True, slots=True, order=True)
class Cube:
    """A lattice point in cube coordinates with ``x + y + z == 0``."""

    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        if self.x + self.y + self.z != 0:
            raise ValueError(f"cube coordinates must sum to 0: {self!r}")

    def __add__(self, other: Cube) -> Cube:
        return Cube(self.x + other.x, self.y + other.y, self.z + other.z)

    def scaled(self, factor: int) -> Cube:
        return Cube(self.x * factor, self.y * factor, self.z * factor)

    def unit_xy(self) -> tuple[float, float]:
        """Project onto an unscaled 2D plane (screen y grows downwards).

        One lattice edge has length 1.0.  The renderer applies its own scale
        and centring; this is only a shape, not a pixel position.
        """
        q, r = self.x, self.z
        return (q + r / 2.0, r * SQRT3_OVER_2)


# The six lattice directions, in a fixed order.  This order is the tie-breaker
# that makes jump-path search deterministic, so do not reorder it casually.
DIRECTIONS: tuple[Cube, ...] = (
    Cube(1, -1, 0),
    Cube(1, 0, -1),
    Cube(0, 1, -1),
    Cube(-1, 1, 0),
    Cube(-1, 0, 1),
    Cube(0, -1, 1),
)

NUM_DIRECTIONS = len(DIRECTIONS)
