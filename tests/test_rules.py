from __future__ import annotations

import pytest
from conftest import make_state

from chinese_checkers.game.board import CAMP_SIZE
from chinese_checkers.game.move import Move, MoveKind
from chinese_checkers.game.rules import (
    IllegalMoveError,
    find_winner,
    has_finished,
    legal_moves,
    validate_move,
)
from chinese_checkers.game.state import DEFAULT_PLAYERS, EMPTY, initial_state


def test_initial_state_gives_each_player_ten_pieces(board):
    state = initial_state()
    for spec in DEFAULT_PLAYERS:
        assert len(state.positions_of(spec.id)) == CAMP_SIZE


def test_initial_state_has_thirty_pieces(board):
    state = initial_state()
    assert sum(1 for v in state.occupancy if v != EMPTY) == 30
    assert len(state.occupancy) == len(board)


def test_initial_pieces_sit_exactly_in_their_home_camp(board):
    state = initial_state()
    for spec in DEFAULT_PLAYERS:
        assert set(state.positions_of(spec.id)) == set(board.camp_positions(spec.camp))


def test_target_camps_are_empty_at_the_start(board):
    state = initial_state()
    for spec in DEFAULT_PLAYERS:
        for pid in board.camp_positions(spec.target_camp):
            assert state.is_empty(pid)


def test_active_camps_are_distinct_and_targets_are_opposites():
    camps = {spec.camp for spec in DEFAULT_PLAYERS}
    assert len(camps) == len(DEFAULT_PLAYERS)
    for spec in DEFAULT_PLAYERS:
        assert spec.target_camp is spec.camp.opposite


def test_validate_rejects_a_move_by_the_wrong_player(board):
    state = initial_state()
    move = legal_moves(board, state, 2)[0]
    with pytest.raises(IllegalMoveError):
        validate_move(board, state, move)  # state says it is player 1's turn


def test_validate_rejects_a_fabricated_destination(board):
    state = initial_state()
    source = state.positions_of(1)[0]
    empty = next(i for i, v in enumerate(state.occupancy) if v == EMPTY)
    bogus = Move(1, source, empty, (source, empty), MoveKind.STEP)
    with pytest.raises(IllegalMoveError):
        validate_move(board, state, bogus)


def test_validate_rejects_a_non_canonical_path(board):
    state = initial_state()
    move = legal_moves(board, state, 1)[0]
    tampered = Move(
        move.player_id, move.source, move.destination, (move.source, 999, move.destination)
    )
    with pytest.raises(IllegalMoveError):
        validate_move(board, state, tampered)


def test_validate_accepts_every_generated_move(board):
    state = initial_state()
    for move in legal_moves(board, state, 1):
        validate_move(board, state, move)


def test_win_requires_all_ten_pieces_in_the_target_camp(board):
    spec = DEFAULT_PLAYERS[0]
    target = board.camp_positions(spec.target_camp)

    almost = make_state(board, {pid: spec.id for pid in target[:-1]})
    assert not has_finished(board, almost, spec)
    assert find_winner(board, almost, DEFAULT_PLAYERS) is None

    complete = make_state(board, {pid: spec.id for pid in target})
    assert has_finished(board, complete, spec)
    assert find_winner(board, complete, DEFAULT_PLAYERS) == spec.id


def test_a_target_camp_filled_by_the_wrong_player_is_not_a_win(board):
    spec = DEFAULT_PLAYERS[0]
    other = DEFAULT_PLAYERS[1]
    target = board.camp_positions(spec.target_camp)
    state = make_state(board, {pid: other.id for pid in target})
    assert not has_finished(board, state, spec)


def test_no_winner_in_the_initial_position(board):
    assert find_winner(board, initial_state(), DEFAULT_PLAYERS) is None
