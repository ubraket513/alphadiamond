from __future__ import annotations

import pytest

from conftest import make_state
from diamond.contract.board import CAMP_SIZE
from diamond.contract.move import Move, MoveKind
from diamond.contract.state import DEFAULT_PLAYERS, EMPTY, initial_state
from diamond.game.rules import (
    IllegalMoveError,
    find_winner,
    has_finished,
    legal_moves,
    validate_move,
)


def test_initial_state_gives_each_player_ten_pieces(board):
    state = initial_state()
    for spec in DEFAULT_PLAYERS:
        assert len(state.positions_of(spec.id)) == CAMP_SIZE == 10


def test_initial_state_has_thirty_pieces(board):
    state = initial_state()
    assert sum(1 for v in state.occupancy if v != EMPTY) == 30
    assert len(state.occupancy) == len(board)


def test_initial_pieces_sit_exactly_in_their_home_camp(board):
    state = initial_state()
    for spec in DEFAULT_PLAYERS:
        assert set(state.positions_of(spec.id)) == set(board.camp_positions(spec.camp))


def test_no_player_starts_with_a_piece_already_home(board):
    state = initial_state()
    for spec in DEFAULT_PLAYERS:
        for pid in board.camp_positions(spec.target_camp):
            assert state.occupant(pid) != spec.id


def test_each_target_camp_starts_with_two_holes_held_by_opponents(board):
    """A "+" camp and a "-" camp share a hexagon corner, so every target camp
    opens with exactly two holes blocked.  They clear as opponents move out."""
    state = initial_state()
    for spec in DEFAULT_PLAYERS:
        blocked = [pid for pid in board.camp_positions(spec.target_camp) if not state.is_empty(pid)]
        assert len(blocked) == 2


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


# -- match setup: seat count and turn order ---------------------------------


def test_two_player_match_reuses_two_of_the_three_seats(board):
    """A 2-player match leaves the yellow seat empty rather than moving anyone,
    so the board geometry is identical whatever the seat count."""
    from diamond.contract.state import build_players

    two = build_players(2)
    three = build_players(3)
    assert len(two) == 2

    three_by_camp = {spec.camp for spec in three}
    assert {spec.camp for spec in two} <= three_by_camp

    # ...and the two seats are the 120-degree-apart pair, not opposites.
    assert two[0].camp is not two[1].target_camp
    assert two[1].camp is not two[0].target_camp


def test_two_player_targets_are_empty_of_the_owner_at_the_start(board):
    """Both aim at camps nobody starts in, so neither has to evict the other."""
    from diamond.contract.state import build_players, initial_state

    players = build_players(2)
    state = initial_state(players, board)
    for spec in players:
        for pid in board.camp_positions(spec.target_camp):
            assert state.occupant(pid) != spec.id


def test_turn_order_is_the_seat_list_order(board):
    from diamond.contract.state import build_players
    from diamond.game.rules import next_player_id

    players = build_players(3, order=(3, 1, 2))
    assert [p.id for p in players] == [3, 1, 2]
    assert next_player_id(players, 3) == 1
    assert next_player_id(players, 1) == 2
    assert next_player_id(players, 2) == 3


def test_first_player_to_act_follows_the_chosen_order(board):
    from diamond.contract.state import build_players, initial_state

    for order in ((1, 2, 3), (2, 3, 1), (3, 1, 2)):
        players = build_players(3, order=order)
        assert initial_state(players, board).current_player_id == order[0]


def test_next_player_skips_players_already_home(board):
    from diamond.contract.state import build_players
    from diamond.game.rules import next_player_id

    players = build_players(3)
    assert next_player_id(players, 1, skip=(2,)) == 3
    assert next_player_id(players, 3, skip=(1,)) == 2
    # Everyone skipped: caller gets its own id back rather than looping forever.
    assert next_player_id(players, 1, skip=(1, 2, 3)) == 1


def test_every_seat_count_places_ten_pieces_per_player(board):
    from diamond.contract.state import EMPTY, SUPPORTED_PLAYER_COUNTS, build_players, initial_state

    for count in SUPPORTED_PLAYER_COUNTS:
        players = build_players(count)
        state = initial_state(players, board)
        assert sum(1 for v in state.occupancy if v != EMPTY) == 10 * count
        for spec in players:
            assert len(state.positions_of(spec.id)) == 10


def test_build_players_rejects_a_bad_seat_count_or_order(board):
    from diamond.contract.state import build_players

    with pytest.raises(ValueError):
        build_players(4)
    with pytest.raises(ValueError):
        build_players(3, order=(1, 1, 2))
    with pytest.raises(ValueError):
        build_players(2, order=(1, 3))  # seat 3 is not in a 2-player match


def test_ai_seats_become_ai_players(board):
    from diamond.contract.state import PlayerKind, build_players

    players = build_players(3, ai_seats=(2,))
    kinds = {p.id: p.kind for p in players}
    assert kinds[2] is PlayerKind.AI
    assert kinds[1] is PlayerKind.HUMAN
    assert kinds[3] is PlayerKind.HUMAN
