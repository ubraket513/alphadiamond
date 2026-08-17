"""Controller state machine: proposal/confirm/cancel, undo, and stale AI results.

These run on a plain QCoreApplication — no display, no QML.
"""

from __future__ import annotations

import pytest
from conftest import pump

from diamond.agents.base import MoveProposal, MoveRequest
from diamond.agents.random_agent import RandomAgent
from diamond.app.controller import GameController, Phase
from diamond.game.board import PLAYABLE_HOLES
from diamond.game.state import DEFAULT_PLAYERS, PlayerKind


@pytest.fixture
def ctrl(qapp):
    controller = GameController(
        DEFAULT_PLAYERS,
        agents={
            spec.id: RandomAgent(seed=17)
            for spec in DEFAULT_PLAYERS
            if spec.kind is PlayerKind.AI
        },
        thinking_delay_ms=0,
        animate=False,
    )
    yield controller
    controller.shutdown()


def first_legal(controller):
    return controller.session.legal_moves()[0]


def play_human_turn(controller):
    move = first_legal(controller)
    controller.selectPosition(move.source)
    controller.selectPosition(move.destination)
    controller.confirmProposal()
    return move


# -- selection -------------------------------------------------------------


def test_starts_waiting_for_player_one(ctrl):
    assert ctrl.phase == str(Phase.WAITING_FOR_HUMAN)
    assert ctrl.currentPlayerId == 1
    assert ctrl.turnNumber == 1
    assert not ctrl.hasProposal


def test_selecting_own_piece_publishes_legal_destinations(ctrl):
    move = first_legal(ctrl)
    ctrl.selectPosition(move.source)
    assert ctrl.selectedPosition == move.source
    assert ctrl.canCancel
    assert not ctrl.hasProposal


def test_selecting_an_opponent_piece_is_refused(ctrl):
    foreign = ctrl.session.state.positions_of(2)[0]
    errors = []
    ctrl.errorRaised.connect(errors.append)
    ctrl.selectPosition(foreign)
    assert errors
    assert ctrl.selectedPosition == -1


def test_illegal_destination_keeps_the_selection(ctrl):
    move = first_legal(ctrl)
    ctrl.selectPosition(move.source)
    illegal = next(
        pid
        for pid in range(PLAYABLE_HOLES)
        if ctrl.session.state.is_empty(pid) and pid not in ctrl.session.moves_from(move.source)
    )
    errors = []
    ctrl.errorRaised.connect(errors.append)
    ctrl.selectPosition(illegal)
    assert errors
    assert ctrl.selectedPosition == move.source
    assert not ctrl.hasProposal


# -- proposal / confirm / cancel -------------------------------------------


def test_proposing_does_not_change_authoritative_state(ctrl):
    before = ctrl.session.state
    move = first_legal(ctrl)
    ctrl.selectPosition(move.source)
    ctrl.selectPosition(move.destination)
    assert ctrl.phase == str(Phase.HUMAN_MOVE_PROPOSED)
    assert ctrl.hasProposal
    assert ctrl.canConfirm
    assert ctrl.session.state == before  # nothing committed yet


def test_cancel_leaves_the_state_untouched(ctrl):
    before = ctrl.session.state
    move = first_legal(ctrl)
    ctrl.selectPosition(move.source)
    ctrl.selectPosition(move.destination)
    ctrl.cancelProposal()
    assert ctrl.session.state == before
    assert not ctrl.hasProposal
    assert ctrl.phase == str(Phase.WAITING_FOR_HUMAN)


def test_confirm_commits_and_advances_the_turn(ctrl):
    before = ctrl.session.state
    move = play_human_turn(ctrl)
    assert ctrl.session.state != before
    assert ctrl.session.state.occupant(move.destination) == 1
    assert ctrl.session.state.is_empty(move.source)
    assert ctrl.currentPlayerId == 2
    assert ctrl.turnNumber == 2
    assert not ctrl.hasProposal


def test_board_input_is_locked_while_a_proposal_is_pending(ctrl):
    move = first_legal(ctrl)
    ctrl.selectPosition(move.source)
    ctrl.selectPosition(move.destination)
    other = next(p for p in ctrl.session.state.positions_of(1) if p != move.source)
    errors = []
    ctrl.errorRaised.connect(errors.append)
    ctrl.selectPosition(other)
    assert errors
    assert ctrl.phase == str(Phase.HUMAN_MOVE_PROPOSED)
    assert ctrl.proposalSummary == move.short_text()


def test_confirm_without_a_proposal_does_nothing(ctrl):
    before = ctrl.session.state
    ctrl.confirmProposal()
    assert ctrl.session.state == before


# -- AI turn ---------------------------------------------------------------


def reach_ai_turn(qapp, controller):
    play_human_turn(controller)  # P1
    play_human_turn(controller)  # P2
    assert controller.currentPlayerId == 3
    assert pump(qapp, lambda: controller.phase == str(Phase.AI_MOVE_PROPOSED))


def test_ai_proposal_is_not_committed_automatically(qapp, ctrl):
    reach_ai_turn(qapp, ctrl)
    before = ctrl.session.state
    assert ctrl.proposalIsAi
    assert ctrl.aiStatus == "Move proposed"
    assert ctrl.aiAgentName == "RandomAgent"
    assert ctrl.session.state == before
    assert ctrl.currentPlayerId == 3


def test_confirm_ai_move_commits_it(qapp, ctrl):
    reach_ai_turn(qapp, ctrl)
    summary = ctrl.proposalSummary
    ctrl.confirmProposal()
    assert ctrl.currentPlayerId == 1
    assert ctrl.session.history[-1].move.short_text() == summary
    assert ctrl.session.history[-1].metadata["agent"] == "RandomAgent"


def test_think_again_proposes_without_changing_state(qapp, ctrl):
    reach_ai_turn(qapp, ctrl)
    before = ctrl.session.state
    first = ctrl.proposalSummary
    ctrl.thinkAgain()
    assert pump(qapp, lambda: ctrl.phase == str(Phase.AI_MOVE_PROPOSED))
    assert ctrl.session.state == before
    assert ctrl.proposalSummary != first


def test_ai_details_expose_only_reported_metadata(qapp, ctrl):
    reach_ai_turn(qapp, ctrl)
    labels = {entry["label"] for entry in ctrl.aiDetails}
    assert labels == {"Agent", "Seed", "Legal moves"}


def test_ai_details_are_empty_on_a_human_turn(ctrl):
    assert ctrl.aiDetails == []


# -- undo / new game -------------------------------------------------------


def test_undo_reverts_the_last_committed_move(ctrl):
    before = ctrl.session.state
    play_human_turn(ctrl)
    ctrl.undoLastMove()
    assert ctrl.session.state == before
    assert ctrl.currentPlayerId == 1
    assert not ctrl.canUndo


def test_new_game_resets_everything(qapp, ctrl):
    play_human_turn(ctrl)
    label = ctrl.gameLabel
    ctrl.newGame()
    assert ctrl.turnNumber == 1
    assert ctrl.currentPlayerId == 1
    assert not ctrl.canUndo
    assert ctrl.gameLabel != label


# -- stale asynchronous results -------------------------------------------


def test_new_game_discards_an_in_flight_ai_result(qapp):
    controller = GameController(
        DEFAULT_PLAYERS,
        agents={3: RandomAgent(seed=5)},
        thinking_delay_ms=250,
        animate=False,
    )
    try:
        play_human_turn(controller)
        play_human_turn(controller)
        assert controller.phase == str(Phase.AI_THINKING)

        controller.newGame()  # bumps the generation while the worker is running
        pump(qapp, lambda: False, timeout=0.8)

        assert controller.phase == str(Phase.WAITING_FOR_HUMAN)
        assert controller.currentPlayerId == 1
        assert not controller.hasProposal
        assert controller.session.history == ()
    finally:
        controller.shutdown()


def test_undo_discards_an_in_flight_ai_result(qapp):
    controller = GameController(
        DEFAULT_PLAYERS,
        agents={3: RandomAgent(seed=5)},
        thinking_delay_ms=250,
        animate=False,
    )
    try:
        play_human_turn(controller)
        play_human_turn(controller)
        assert controller.phase == str(Phase.AI_THINKING)
        expected = controller.session.history[-2].state_after

        controller.undoLastMove()
        pump(qapp, lambda: False, timeout=0.8)

        assert controller.session.state == expected
        assert controller.currentPlayerId == 2
        assert not controller.hasProposal
    finally:
        controller.shutdown()


# -- game over -------------------------------------------------------------


def _one_move_from_home(board, spec):
    target = board.camp_positions(spec.target_camp)
    last = target[-1]
    entry = next(n for n in board.neighbours(last) if n is not None and n not in target)
    pieces = {pid: spec.id for pid in target[:-1]}
    pieces[entry] = spec.id
    return pieces, entry, last


def test_filling_the_target_camp_ends_a_two_player_match(qapp, board):
    from conftest import make_state

    from diamond.game.state import build_players

    players = build_players(2, ai_seats=(2,))
    spec = players[0]
    pieces, entry, last = _one_move_from_home(board, spec)

    controller = GameController(
        players,
        agents={2: RandomAgent(seed=1)},
        thinking_delay_ms=0,
        animate=False,
        initial_state=make_state(board, pieces, current_player_id=spec.id, turn=40),
    )
    try:
        finished = []
        controller.gameFinished.connect(finished.append)

        controller.selectPosition(entry)
        controller.selectPosition(last)
        controller.confirmProposal()

        assert controller.phase == str(Phase.GAME_OVER)
        assert controller.isGameOver
        assert controller.winnerId == spec.id
        assert controller.winnerName == spec.name
        assert finished == [spec.id]
        assert [row["playerId"] for row in controller.standings] == [spec.id, players[1].id]

        # the board is locked once the match is over
        errors = []
        controller.errorRaised.connect(errors.append)
        controller.selectPosition(last)
        assert errors
    finally:
        controller.shutdown()


def test_three_player_match_continues_after_first_place(qapp, board):
    """First place is announced, the seat drops out, the match stays live."""
    from conftest import make_state

    spec = DEFAULT_PLAYERS[0]
    others = [p for p in DEFAULT_PLAYERS if p.id != spec.id]
    pieces, entry, last = _one_move_from_home(board, spec)
    target = set(board.camp_positions(spec.target_camp))
    for other in others:
        for pid in board.camp_positions(other.camp):
            if pid not in target:
                pieces.setdefault(pid, other.id)

    controller = GameController(
        DEFAULT_PLAYERS,
        agents={3: RandomAgent(seed=1)},
        thinking_delay_ms=0,
        animate=False,
        initial_state=make_state(board, pieces, current_player_id=spec.id, turn=40),
    )
    try:
        over = []
        placed = []
        controller.gameFinished.connect(over.append)
        controller.playerFinished.connect(lambda pid, place: placed.append((pid, place)))

        controller.selectPosition(entry)
        controller.selectPosition(last)
        controller.confirmProposal()

        assert placed == [(spec.id, 1)]
        assert over == []
        assert not controller.isGameOver
        assert controller.winnerId == spec.id
        assert controller.currentPlayerId != spec.id
        assert [row["playerId"] for row in controller.standings] == [spec.id]
        assert "2nd" in controller.resultSummary
    finally:
        controller.shutdown()


def test_start_match_reconfigures_seats_and_turn_order(qapp, board):
    controller = GameController(
        DEFAULT_PLAYERS, agents={3: RandomAgent(seed=1)}, thinking_delay_ms=0, animate=False
    )
    try:
        assert controller.startMatch([2, 1], [])
        assert controller.playerCount == 2
        assert list(controller.turnOrder) == [2, 1]
        assert controller.currentPlayerId == 2
        assert controller.playerModel.rowCount() == 2
        assert controller.pieceModel.rowCount() == 20

        assert controller.startMatch([3, 1, 2], [3])
        assert list(controller.turnOrder) == [3, 1, 2]
        assert controller.currentPlayerId == 3
        assert controller.isCurrentPlayerAi
        assert controller.pieceModel.rowCount() == 30
    finally:
        controller.shutdown()


def test_start_match_rejects_an_invalid_setup(qapp, board):
    controller = GameController(
        DEFAULT_PLAYERS, agents={3: RandomAgent(seed=1)}, thinking_delay_ms=0, animate=False
    )
    try:
        errors = []
        controller.errorRaised.connect(errors.append)
        assert not controller.startMatch([1, 1, 2], [])
        assert errors
        # the running match is untouched by a rejected setup
        assert controller.playerCount == 3
        assert list(controller.turnOrder) == [1, 2, 3]
    finally:
        controller.shutdown()


def test_an_illegal_agent_proposal_is_rejected_by_the_engine(qapp):
    class BadAgent:
        name = "BadAgent"

        def choose_move(self, request: MoveRequest):
            return MoveProposal(3, 0, 1, (0, 1))  # not player 3's piece

    controller = GameController(
        DEFAULT_PLAYERS, agents={3: BadAgent()}, thinking_delay_ms=0, animate=False
    )
    try:
        errors = []
        controller.errorRaised.connect(errors.append)
        play_human_turn(controller)
        play_human_turn(controller)
        assert pump(qapp, lambda: bool(errors))
        assert not controller.hasProposal
        assert controller.session.state.current_player_id == 3
    finally:
        controller.shutdown()


def test_player_model_reports_the_finishing_place(qapp, board):
    """Regression: the commit path once refreshed the model without the podium,
    blanking every player's place until the next unrelated sync."""
    from conftest import make_state

    spec = DEFAULT_PLAYERS[0]
    others = [p for p in DEFAULT_PLAYERS if p.id != spec.id]
    pieces, entry, last = _one_move_from_home(board, spec)
    target = set(board.camp_positions(spec.target_camp))
    for other in others:
        for pid in board.camp_positions(other.camp):
            if pid not in target:
                pieces.setdefault(pid, other.id)

    controller = GameController(
        DEFAULT_PLAYERS,
        agents={3: RandomAgent(seed=1)},
        thinking_delay_ms=0,
        animate=False,
        initial_state=make_state(board, pieces, current_player_id=spec.id, turn=40),
    )
    try:
        controller.selectPosition(entry)
        controller.selectPosition(last)
        controller.confirmProposal()

        model = controller.playerModel
        roles = {bytes(v).decode(): k for k, v in model.roleNames().items()}
        rows = [
            {
                name: model.data(model.index(r, 0), role)
                for name, role in roles.items()
            }
            for r in range(model.rowCount())
        ]
        by_id = {row["playerId"]: row for row in rows}

        assert by_id[spec.id]["place"] == 1
        assert by_id[spec.id]["placeLabel"] == "1st"
        assert by_id[spec.id]["homeCount"] == 10
        for other in others:
            assert by_id[other.id]["place"] == 0
            assert by_id[other.id]["placeLabel"] == ""
    finally:
        controller.shutdown()


def test_player_model_rows_follow_the_chosen_turn_order(qapp, board):
    controller = GameController(
        DEFAULT_PLAYERS, agents={3: RandomAgent(seed=1)}, thinking_delay_ms=0, animate=False
    )
    try:
        assert controller.startMatch([3, 1, 2], [1])
        model = controller.playerModel
        roles = {bytes(v).decode(): k for k, v in model.roleNames().items()}
        seats = [
            model.data(model.index(r, 0), roles["playerId"]) for r in range(model.rowCount())
        ]
        turn_indexes = [
            model.data(model.index(r, 0), roles["turnIndex"]) for r in range(model.rowCount())
        ]
        assert seats == [3, 1, 2]
        assert turn_indexes == [1, 2, 3]
    finally:
        controller.shutdown()
