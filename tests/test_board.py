from __future__ import annotations

import pytest

from chinese_checkers.game.board import CAMP_SIZE, PLAYABLE_HOLES, Camp
from chinese_checkers.game.coordinates import DIRECTIONS, Cube


def test_board_has_exactly_121_playable_holes(board):
    assert len(board) == PLAYABLE_HOLES
    assert len(board.positions) == PLAYABLE_HOLES


def test_position_ids_are_dense_and_stable(board):
    assert [p.id for p in board.positions] == list(range(PLAYABLE_HOLES))
    # IDs come from sorting by (z, x); rebuilding must give the same order.
    from chinese_checkers.game.board import Board

    assert [p.cube for p in Board().positions] == [p.cube for p in board.positions]


def test_every_camp_has_ten_holes(board):
    for camp in Camp:
        assert len(board.camp_positions(camp)) == CAMP_SIZE


def test_camps_are_disjoint_and_cover_sixty_holes(board):
    ids = [pid for camp in Camp for pid in board.camp_positions(camp)]
    assert len(ids) == 60
    assert len(set(ids)) == 60


def test_central_hexagon_has_61_holes(board):
    assert sum(1 for p in board.positions if p.camp is None) == 61


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
