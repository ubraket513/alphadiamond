from __future__ import annotations

import json

import pytest

from chinese_checkers.game.rules import IllegalMoveError
from chinese_checkers.game.session import SCHEMA_VERSION, GameSession
from chinese_checkers.game.state import initial_state


def play(session: GameSession, count: int) -> None:
    for _ in range(count):
        session.commit(session.legal_moves()[0])


def test_turn_order_cycles_p1_p2_p3(board):
    session = GameSession()
    seen = []
    for _ in range(9):
        seen.append(session.state.current_player_id)
        session.commit(session.legal_moves()[0])
    assert seen == [1, 2, 3, 1, 2, 3, 1, 2, 3]


def test_turn_number_increments_once_per_committed_move(board):
    session = GameSession()
    assert session.state.turn_number == 1
    play(session, 5)
    assert session.state.turn_number == 6
    assert len(session.history) == 5


def test_history_records_the_committed_move(board):
    session = GameSession()
    move = session.legal_moves()[0]
    record = session.commit(move)
    assert record.move == move
    assert record.player_id == 1
    assert record.turn_number == 1
    assert record.timestamp
    assert record.state_after is session.state


def test_undo_restores_the_exact_previous_state(board):
    session = GameSession()
    before = session.state
    move = session.legal_moves()[0]
    session.commit(move)
    assert session.state != before
    session.undo()
    assert session.state == before
    assert session.history == ()


def test_repeated_undo_walks_back_to_the_initial_position(board):
    session = GameSession()
    snapshots = []
    for _ in range(6):
        snapshots.append(session.state)
        session.commit(session.legal_moves()[0])
    for expected in reversed(snapshots):
        session.undo()
        assert session.state == expected
    assert session.state == initial_state()
    assert not session.can_undo()


def test_undo_on_a_fresh_game_is_a_no_op(board):
    session = GameSession()
    assert session.undo() is None
    assert session.state == initial_state()


def test_commit_rejects_an_illegal_move(board):
    session = GameSession()
    foreign = session.legal_moves(2)[0]
    with pytest.raises(IllegalMoveError):
        session.commit(foreign)
    assert session.state == initial_state()
    assert session.history == ()


def test_state_is_immutable_across_commits(board):
    session = GameSession()
    before = session.state
    occupancy_snapshot = before.occupancy
    session.commit(session.legal_moves()[0])
    assert before.occupancy == occupancy_snapshot  # the old snapshot is untouched


def test_save_and_load_round_trip(board, tmp_path):
    session = GameSession()
    play(session, 7)
    path = session.save(tmp_path / "game.json")

    restored = GameSession()
    restored.load(path)
    assert restored.state == session.state
    assert restored.state.current_player_id == session.state.current_player_id
    assert restored.state.turn_number == session.state.turn_number
    assert [r.move for r in restored.history] == [r.move for r in session.history]


def test_loaded_game_can_still_be_undone(board, tmp_path):
    session = GameSession()
    play(session, 4)
    expected = session.history[-2].state_after
    path = session.save(tmp_path / "game.json")

    restored = GameSession()
    restored.load(path)
    restored.undo()
    assert restored.state == expected


def test_save_payload_contains_the_documented_fields(board, tmp_path):
    session = GameSession()
    play(session, 2)
    data = json.loads(session.save(tmp_path / "g.json").read_text())
    for key in (
        "schema_version",
        "occupancy",
        "current_player_id",
        "turn_number",
        "status",
        "players",
        "history",
    ):
        assert key in data
    assert data["schema_version"] == SCHEMA_VERSION


def test_loading_an_unknown_schema_version_fails(board):
    with pytest.raises(ValueError):
        GameSession().load_dict({"schema_version": 999})


def test_loading_a_tampered_board_fails(board, tmp_path):
    session = GameSession()
    play(session, 3)
    data = session.to_dict()
    data["occupancy"][0] = 3  # corrupt the board without touching the history
    with pytest.raises(ValueError):
        GameSession().load_dict(data)


def one_move_from_winning(board, spec):
    """P1 has nine pieces home and the tenth adjacent to the last empty hole."""
    target = board.camp_positions(spec.target_camp)
    last = target[-1]
    entry = next(
        n
        for n in board.neighbours(last)
        if n is not None and n not in target
    )
    pieces = {pid: spec.id for pid in target[:-1]}
    pieces[entry] = spec.id
    return pieces, entry, last


def test_finishing_the_target_camp_ends_the_game(board):
    from chinese_checkers.game.state import DEFAULT_PLAYERS, GameStatus

    from conftest import make_state

    spec = DEFAULT_PLAYERS[0]
    pieces, entry, last = one_move_from_winning(board, spec)

    setup = make_state(board, pieces, current_player_id=spec.id, turn=40)
    session = GameSession(initial=setup)
    assert not session.is_over

    move = session.moves_from(entry)[last]
    session.commit(move)

    assert session.is_over
    assert session.state.status is GameStatus.FINISHED
    assert session.state.winner_id == spec.id


def test_no_further_moves_are_accepted_after_the_game_ends(board):
    from chinese_checkers.game.state import DEFAULT_PLAYERS

    from conftest import make_state

    spec = DEFAULT_PLAYERS[0]
    pieces, entry, last = one_move_from_winning(board, spec)
    setup = make_state(board, pieces, current_player_id=spec.id, turn=40)
    session = GameSession(initial=setup)
    winning_move = session.moves_from(entry)[last]
    session.commit(winning_move)

    # `is_over` is checked before legality, so any move is refused outright
    with pytest.raises(RuntimeError):
        session.commit(winning_move)


def test_reset_returns_to_the_initial_position(board):
    session = GameSession()
    play(session, 5)
    session.reset()
    assert session.state == initial_state()
    assert session.history == ()
