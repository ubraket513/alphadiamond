from __future__ import annotations

import pytest

from diamond.contract.board import CAMP_SIZE, HEX_RADIUS, PLAYABLE_HOLES, Camp
from diamond.contract.coordinates import DIRECTIONS, Cube


def test_board_has_exactly_73_playable_holes(board):
    assert len(board) == PLAYABLE_HOLES
    assert len(board.positions) == PLAYABLE_HOLES


def test_position_ids_are_dense_and_stable(board):
    assert [p.id for p in board.positions] == list(range(PLAYABLE_HOLES))
    # IDs come from sorting by (z, x); rebuilding must give the same order.
    from diamond.contract.board import Board

    assert [p.cube for p in Board().positions] == [p.cube for p in board.positions]


def test_every_camp_has_ten_holes(board):
    for camp in Camp:
        assert len(board.camp_positions(camp)) == CAMP_SIZE


def test_camp_base_edge_is_a_hexagon_side(board):
    """The Diamond rule: a camp triangle's 4-hole base *is* a hexagon side."""
    for camp in Camp:
        ids = board.camp_positions(camp)
        axis, sign = camp.value[0], camp.value[1]
        on_hexagon_side = [
            pid
            for pid in ids
            if abs(getattr(board.position(pid).cube, axis)) == HEX_RADIUS
        ]
        assert len(on_hexagon_side) == HEX_RADIUS + 1 == 4
        # ...and those holes lie inside the hexagon, not out on the star point.
        for pid in on_hexagon_side:
            cube = board.position(pid).cube
            assert max(abs(cube.x), abs(cube.y), abs(cube.z)) == HEX_RADIUS
            assert (getattr(cube, axis) > 0) == (sign == "+")


def test_camp_rows_shrink_from_four_to_one(board):
    """Ten holes arranged 4 + 3 + 2 + 1 along the camp's axis."""
    for camp in Camp:
        axis = camp.value[0]
        rows: dict[int, int] = {}
        for pid in board.camp_positions(camp):
            value = abs(getattr(board.position(pid).cube, axis))
            rows[value] = rows.get(value, 0) + 1
        assert sorted(rows.values(), reverse=True) == [4, 3, 2, 1]


def test_starting_camps_are_mutually_disjoint(board):
    """The three starting camps sit on alternating hexagon sides, so the 30
    opening pieces all fit."""
    for group in ((Camp.X_POS, Camp.Y_POS, Camp.Z_POS), (Camp.X_NEG, Camp.Y_NEG, Camp.Z_NEG)):
        ids = [pid for camp in group for pid in board.camp_positions(camp)]
        assert len(ids) == len(set(ids)) == 3 * CAMP_SIZE


def test_each_hexagon_corner_is_shared_by_one_plus_and_one_minus_camp(board):
    """Camps are deliberately not globally disjoint; nothing may assume it."""
    shared = [p for p in board.positions if len(p.camps) > 1]
    assert len(shared) == 6
    for position in shared:
        signs = {camp.value[1] for camp in position.camps}
        assert len(position.camps) == 2
        assert signs == {"+", "-"}


def test_central_hexagon_has_37_holes(board):
    inside = [
        p
        for p in board.positions
        if max(abs(p.cube.x), abs(p.cube.y), abs(p.cube.z)) <= HEX_RADIUS
    ]
    assert len(inside) == 37


def test_opposite_camps_are_mutual(board):
    for camp in Camp:
        assert camp.opposite.opposite is camp
        assert camp.opposite is not camp


def test_neighbour_relation_is_symmetric(board):
    for pid in range(len(board)):
        for direction, neighbour in enumerate(board.neighbours(pid)):
            if neighbour is None:
                continue
            opposite = (direction + 3) % len(DIRECTIONS)
            assert board.neighbour(neighbour, opposite) == pid


def test_neighbours_match_cube_arithmetic(board):
    for position in board.positions:
        for direction, neighbour in enumerate(board.neighbours(position.id)):
            expected = position.cube + DIRECTIONS[direction]
            if neighbour is None:
                with pytest.raises(KeyError):
                    board.id_of(expected)
            else:
                assert board.position(neighbour).cube == expected


def test_cube_coordinates_always_sum_to_zero(board):
    for position in board.positions:
        assert position.cube.x + position.cube.y + position.cube.z == 0


def test_invalid_cube_is_rejected():
    with pytest.raises(ValueError):
        Cube(1, 1, 1)


def test_camp_corners_are_camp_members(board):
    for camp in Camp:
        corners = board.camp_corners(camp)
        assert len(set(corners)) == 3
        assert set(corners) <= set(board.camp_positions(camp))


def test_edges_are_unique_neighbour_pairs(board):
    edges = board.edges
    assert len(edges) == len(set(edges))
    for a, b in edges:
        assert a < b
        assert b in [n for n in board.neighbours(a) if n is not None]
