from __future__ import annotations

from conftest import make_state

from diamond.game.coordinates import DIRECTIONS, Cube
from diamond.game.move import MoveKind
from diamond.game.rules import find_legal_move, legal_moves, moves_from
from diamond.game.state import initial_state


def cid(board, x, y, z):
    return board.id_of(Cube(x, y, z))


def test_lone_piece_has_six_single_steps(board):
    origin = cid(board, 0, 0, 0)
    state = make_state(board, {origin: 1})
    moves = moves_from(board, state, origin)
    assert len(moves) == 6
    assert all(m.kind is MoveKind.STEP for m in moves.values())
    assert all(m.path == (origin, m.destination) for m in moves.values())


def test_adjacent_occupied_hole_is_not_a_step_destination(board):
    origin = cid(board, 0, 0, 0)
    blocker = cid(board, 1, -1, 0)
    landing = cid(board, 2, -2, 0)
    state = make_state(board, {origin: 1, blocker: 2})
    moves = moves_from(board, state, origin)
    assert blocker not in moves
    assert landing in moves  # blocked adjacent becomes a jump instead


def test_single_jump_over_any_colour(board):
    origin = cid(board, 0, 0, 0)
    for jumped_owner in (1, 2, 3):
        over = cid(board, 1, -1, 0)
        landing = cid(board, 2, -2, 0)
        state = make_state(board, {origin: 1, over: jumped_owner})
        move = find_legal_move(board, state, origin, landing)
        assert move is not None
        assert move.kind is MoveKind.JUMP
        assert move.path == (origin, landing)


def test_jump_blocked_when_landing_is_occupied(board):
    origin = cid(board, 0, 0, 0)
    over = cid(board, 1, -1, 0)
    landing = cid(board, 2, -2, 0)
    state = make_state(board, {origin: 1, over: 2, landing: 3})
    assert find_legal_move(board, state, origin, landing) is None


def test_chained_jump_is_one_move(board):
    origin = cid(board, -2, 2, 0)
    over1 = cid(board, -1, 1, 0)
    mid = cid(board, 0, 0, 0)
    over2 = cid(board, 1, -1, 0)
    landing = cid(board, 2, -2, 0)
    state = make_state(board, {origin: 1, over1: 2, over2: 2})
    move = find_legal_move(board, state, origin, landing)
    assert move is not None
    assert move.kind is MoveKind.JUMP
    assert move.path == (origin, mid, landing)
    assert move.hop_count == 2
    assert move.is_multi_hop


def test_chain_may_change_direction_between_hops(board):
    origin = cid(board, 0, 0, 0)
    over1 = cid(board, 1, -1, 0)
    mid = cid(board, 2, -2, 0)
    # second hop uses a different lattice direction, (0, 1, -1)
    over2 = cid(board, 2, -1, -1)
    landing = cid(board, 2, 0, -2)
    state = make_state(board, {origin: 1, over1: 3, over2: 3})
    move = find_legal_move(board, state, origin, landing)
    assert move is not None
    assert move.path == (origin, mid, landing)


def test_every_jump_hop_is_collinear_and_two_steps(board):
    """Source, jumped-over hole and landing must share one lattice direction."""
    state = initial_state()
    checked = 0
    for move in legal_moves(board, state, 1):
        if move.kind is not MoveKind.JUMP:
            continue
        for a, b in zip(move.path, move.path[1:]):
            ca, cb = board.position(a).cube, board.position(b).cube
            delta = Cube(cb.x - ca.x, cb.y - ca.y, cb.z - ca.z)
            matches = [d for d in DIRECTIONS if d.scaled(2) == delta]
            assert len(matches) == 1, f"hop {a}->{b} is not a straight double step"
            over = board.id_of(ca + matches[0])
            assert state.occupant(over) != 0, "the jumped-over hole must be occupied"
            assert over != move.source, "the vacated source is not a jumpable obstacle"
            checked += 1
    assert checked > 0


def test_paths_never_revisit_a_hole(board):
    state = initial_state()
    for player_id in (1, 2, 3):
        for move in legal_moves(board, state, player_id):
            assert len(set(move.path)) == len(move.path)
            assert move.destination != move.source


def test_a_jump_chain_never_lands_back_on_its_source(board):
    """A ring of blockers offers a way back; the visited set must forbid it."""
    origin = cid(board, 0, 0, 0)
    ring = {cid(board, *d): 2 for d in ((1, -1, 0), (-1, 1, 0), (0, 1, -1), (0, -1, 1))}
    state = make_state(board, {origin: 1, **ring})
    moves = moves_from(board, state, origin)
    assert origin not in moves
    for move in moves.values():
        assert move.path.count(origin) == 1


def test_step_and_jump_destinations_are_disjoint(board):
    """Jump hops move by an even offset, single steps by an odd one, so a hole
    is never reachable both ways and the canonical path is unambiguous."""
    state = initial_state()
    for player_id in (1, 2, 3):
        by_source: dict[int, dict[int, MoveKind]] = {}
        for move in legal_moves(board, state, player_id):
            kinds = by_source.setdefault(move.source, {})
            assert move.destination not in kinds
            kinds[move.destination] = move.kind
        for source, kinds in by_source.items():
            src = board.position(source).cube
            for destination, kind in kinds.items():
                dst = board.position(destination).cube
                even = (dst.x - src.x) % 2 == 0 and (dst.y - src.y) % 2 == 0
                assert even == (kind is MoveKind.JUMP)


def test_move_generation_is_deterministic(board):
    state = initial_state()
    first = legal_moves(board, state, 1)
    for _ in range(5):
        assert legal_moves(board, state, 1) == first


def test_only_the_named_players_pieces_move(board):
    state = initial_state()
    for player_id in (1, 2, 3):
        for move in legal_moves(board, state, player_id):
            assert move.player_id == player_id
            assert state.occupant(move.source) == player_id


def test_moves_from_empty_or_foreign_hole_is_empty(board):
    state = initial_state()
    empty_hole = cid(board, 0, 0, 0)
    assert state.is_empty(empty_hole)
    assert moves_from(board, state, empty_hole) == {}
    foreign = state.positions_of(2)[0]
    assert moves_from(board, state, foreign, player_id=1) == {}
