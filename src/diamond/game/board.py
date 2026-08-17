"""Static board topology: the 73 playable holes, their neighbours and camps.

Geometry, derived rather than hard-coded
----------------------------------------
With ``x + y + z == 0`` the Diamond 73-hole star is the union of two large
triangles that overlap in a hexagon::

    triangle "up"    : x >= -3 and y >= -3 and z >= -3      -> 55 holes
    triangle "down"  : x <=  3 and y <=  3 and z <=  3      -> 55 holes
    overlap (hexagon): -3 <= x, y, z <= 3                   -> 37 holes
    star             : union                                -> 73 holes

The hexagon is 7 holes across, so each of its six sides is 4 holes long.

Camps: what makes Diamond different
-----------------------------------
In traditional Chinese Checkers a camp is only the star point sticking out past
the hexagon.  In Diamond a camp is the 10-hole triangle formed by that point
*plus the hexagon side it sits on* -- the triangle's 4-hole base edge and the
hexagon's 4-hole side are the same row of holes::

    camp "x+" : x >= 3, clipped to triangle "up"   -> rows 4+3+2+1 = 10 holes

Consequences that the rules and the renderer both have to live with:

* The three starting camps ``x+, y+, z+`` are the corners of triangle "up".
  They occupy three *alternating* hexagon sides, so they are mutually disjoint
  and the 30 starting pieces fit.  Likewise the three targets ``x-, y-, z-``.
* A "+" camp and a "-" camp still meet at a single hexagon corner hole, so
  every target camp starts with two of its ten holes occupied by opponents.
  Those clear as soon as the opponents move out; camps are deliberately *not*
  disjoint and nothing may assume they are.

Opposite camps are the same axis with the opposite sign, which is exactly the
"move to the camp across the board" relation the rules need.

Position IDs are assigned by sorting holes by ``(z, x)``, i.e. rendering rows
top-to-bottom then left-to-right.  The ordering is deterministic, so IDs are
stable across runs and safe to persist in saved games.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from .coordinates import DIRECTIONS, NUM_DIRECTIONS, Cube

HEX_RADIUS = 3
"""Half-width of the central hexagon: 7 holes across, 4 holes to a side."""

CAMP_THRESHOLD = HEX_RADIUS
"""A hole reaches a camp when one axis hits +/- this value.

Equal to :data:`HEX_RADIUS`, not one past it: in Diamond the camp triangle
includes the hexagon side it stands on.
"""

CAMP_SIZE = 10
PLAYABLE_HOLES = 73


class Camp(Enum):
    """The six star points, named after the axis that peaks there."""

    X_POS = "x+"
    Y_POS = "y+"
    Z_POS = "z+"
    X_NEG = "x-"
    Y_NEG = "y-"
    Z_NEG = "z-"

    @property
    def opposite(self) -> "Camp":
        axis, sign = self.value[0], self.value[1]
        return Camp(axis + ("-" if sign == "+" else "+"))


@dataclass(frozen=True, slots=True)
class BoardPosition:
    id: int
    cube: Cube
    camps: tuple[Camp, ...]
    """Every camp this hole belongs to.

    Usually zero or one, but the six hexagon corner holes sit in two camps at
    once -- see the module docstring.
    """

    @property
    def q(self) -> int:
        return self.cube.x

    @property
    def r(self) -> int:
        return self.cube.z

    def unit_xy(self) -> tuple[float, float]:
        return self.cube.unit_xy()


def _camps_of(cube: Cube) -> tuple[Camp, ...]:
    """Every camp triangle containing ``cube``.

    A "+" camp is the corresponding corner of triangle "up" and is therefore
    clipped to it; a "-" camp likewise to triangle "down".  Without that clip
    the far tip of a third star point would satisfy two camp inequalities at
    once and two starting camps would overlap.
    """
    found: list[Camp] = []
    for axis, value in (("x", cube.x), ("y", cube.y), ("z", cube.z)):
        if value >= CAMP_THRESHOLD and _in_triangle_up(cube):
            found.append(Camp(axis + "+"))
        elif value <= -CAMP_THRESHOLD and _in_triangle_down(cube):
            found.append(Camp(axis + "-"))
    return tuple(found)


def _in_triangle_up(cube: Cube) -> bool:
    return cube.x >= -HEX_RADIUS and cube.y >= -HEX_RADIUS and cube.z >= -HEX_RADIUS


def _in_triangle_down(cube: Cube) -> bool:
    return cube.x <= HEX_RADIUS and cube.y <= HEX_RADIUS and cube.z <= HEX_RADIUS


def _generate_cubes() -> list[Cube]:
    limit = HEX_RADIUS + CAMP_SIZE  # generous bound; the predicates do the work
    cubes: list[Cube] = []
    for x in range(-limit, limit + 1):
        for z in range(-limit, limit + 1):
            cube = Cube(x, -x - z, z)
            if _in_triangle_up(cube) or _in_triangle_down(cube):
                cubes.append(cube)
    cubes.sort(key=lambda c: (c.z, c.x))
    return cubes


class Board:
    """Immutable board topology.  Shared by every :class:`GameState`."""

    __slots__ = ("_positions", "_index_by_cube", "_neighbours", "_camp_positions", "_edges")

    def __init__(self) -> None:
        cubes = _generate_cubes()
        if len(cubes) != PLAYABLE_HOLES:
            raise AssertionError(f"expected {PLAYABLE_HOLES} holes, generated {len(cubes)}")

        self._positions: tuple[BoardPosition, ...] = tuple(
            BoardPosition(id=i, cube=cube, camps=_camps_of(cube)) for i, cube in enumerate(cubes)
        )
        self._index_by_cube: dict[Cube, int] = {p.cube: p.id for p in self._positions}

        # neighbours[pid][direction] -> position id or None (off board)
        self._neighbours: tuple[tuple[int | None, ...], ...] = tuple(
            tuple(self._index_by_cube.get(p.cube + d) for d in DIRECTIONS)
            for p in self._positions
        )

        camps: dict[Camp, list[int]] = {camp: [] for camp in Camp}
        for p in self._positions:
            for camp in p.camps:
                camps[camp].append(p.id)
        self._camp_positions: dict[Camp, tuple[int, ...]] = {
            camp: tuple(ids) for camp, ids in camps.items()
        }

        edges: list[tuple[int, int]] = []
        for pid, row in enumerate(self._neighbours):
            for nid in row:
                if nid is not None and pid < nid:
                    edges.append((pid, nid))
        self._edges: tuple[tuple[int, int], ...] = tuple(edges)

    # -- lookups ---------------------------------------------------------
    @property
    def positions(self) -> tuple[BoardPosition, ...]:
        return self._positions

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """Every neighbour pair exactly once; used to draw the lattice."""
        return self._edges

    def __len__(self) -> int:
        return len(self._positions)

    def position(self, position_id: int) -> BoardPosition:
        return self._positions[position_id]

    def id_of(self, cube: Cube) -> int:
        return self._index_by_cube[cube]

    def neighbours(self, position_id: int) -> tuple[int | None, ...]:
        return self._neighbours[position_id]

    def neighbour(self, position_id: int, direction: int) -> int | None:
        return self._neighbours[position_id][direction]

    def camp_positions(self, camp: Camp) -> tuple[int, ...]:
        return self._camp_positions[camp]

    def camp_corners(self, camp: Camp) -> tuple[int, int, int]:
        """The three extreme holes of a camp triangle, for drawing its fill."""
        ids = self._camp_positions[camp]
        pts = {pid: self._positions[pid].unit_xy() for pid in ids}
        centre = (
            sum(p[0] for p in pts.values()) / len(pts),
            sum(p[1] for p in pts.values()) / len(pts),
        )

        def dist2(pid: int) -> float:
            x, y = pts[pid]
            return (x - centre[0]) ** 2 + (y - centre[1]) ** 2

        far = sorted(ids, key=lambda pid: (-dist2(pid), pid))[:3]
        return (far[0], far[1], far[2])

    def unit_bounds(self) -> tuple[float, float, float, float]:
        xs = [p.unit_xy()[0] for p in self._positions]
        ys = [p.unit_xy()[1] for p in self._positions]
        return (min(xs), min(ys), max(xs), max(ys))


@lru_cache(maxsize=1)
def standard_board() -> Board:
    """The single shared standard board instance."""
    return Board()


__all__ = [
    "Board",
    "BoardPosition",
    "Camp",
    "CAMP_SIZE",
    "CAMP_THRESHOLD",
    "HEX_RADIUS",
    "NUM_DIRECTIONS",
    "PLAYABLE_HOLES",
    "standard_board",
]
