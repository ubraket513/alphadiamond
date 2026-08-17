"""End-to-end controller loop, headless: P1 -> P2 -> P3(agent) -> P1."""

from __future__ import annotations

import pytest
from conftest import pump

from diamond.agents.random_agent import RandomAgent
from diamond.game.board import CAMP_SIZE, PLAYABLE_HOLES
from diamond.app.controller import GameController, Phase
from diamond.game.state import DEFAULT_PLAYERS, PlayerKind


@pytest.fixture
def ctrl(qapp):
    controller = GameController(
        DEFAULT_PLAYERS,
        agents={
            spec.id: RandomAgent(seed=2024)
            for spec in DEFAULT_PLAYERS
            if spec.kind is PlayerKind.AI
        },
        thinking_delay_ms=0,
        animate=False,
    )
    yield controller
    controller.shutdown()


def human_turn(controller):
    move = controller.session.legal_moves()[0]
    controller.selectPosition(move.source)
    controller.selectPosition(move.destination)
    assert controller.phase == str(Phase.HUMAN_MOVE_PROPOSED)
    controller.confirmProposal()
    return move


def test_full_three_player_round_trip(qapp, ctrl):
    assert ctrl.currentPlayerId == 1

    p1 = human_turn(ctrl)
    assert ctrl.currentPlayerId == 2

    p2 = human_turn(ctrl)
    assert ctrl.currentPlayerId == 3

    assert pump(qapp, lambda: ctrl.phase == str(Phase.AI_MOVE_PROPOSED))
    ai_summary = ctrl.proposalSummary
    assert len(ctrl.session.history) == 2  # the agent has not committed anything

    ctrl.confirmProposal()
    assert ctrl.currentPlayerId == 1
    assert ctrl.phase == str(Phase.WAITING_FOR_HUMAN)

    history = ctrl.session.history
    assert len(history) == 3
    assert [r.player_id for r in history] == [1, 2, 3]
    assert history[0].move == p1
    assert history[1].move == p2
    assert history[2].move.short_text() == ai_summary
    assert ctrl.turnNumber == 4


def test_models_stay_in_sync_with_the_engine(qapp, ctrl):
    from PySide6.QtCore import Qt

    def role(model, name):
        for key, value in model.roleNames().items():
            if bytes(value).decode() == name:
                return key
        raise KeyError(name)

    board_model = ctrl.boardModel
    piece_model = ctrl.pieceModel
    history_model = ctrl.historyModel

    assert board_model.rowCount() == PLAYABLE_HOLES
    assert piece_model.rowCount() == CAMP_SIZE * 3
    assert history_model.rowCount() == 0

    human_turn(ctrl)
    human_turn(ctrl)
    assert pump(qapp, lambda: ctrl.phase == str(Phase.AI_MOVE_PROPOSED))
    ctrl.confirmProposal()

    assert history_model.rowCount() == 3
    assert piece_model.rowCount() == CAMP_SIZE * 3

    occupant_role = role(board_model, "occupant")
    for pid in range(PLAYABLE_HOLES):
        shown = board_model.data(board_model.index(pid, 0), occupant_role)
        assert shown == ctrl.session.state.occupant(pid)

    position_role = role(piece_model, "positionId")
    occupied = sorted(
        piece_model.data(piece_model.index(row, 0), position_role)
        for row in range(piece_model.rowCount())
    )
    assert occupied == sorted(
        pid for pid in range(PLAYABLE_HOLES) if not ctrl.session.state.is_empty(pid)
    )


def test_save_and_load_through_the_controller(qapp, ctrl, tmp_path):
    human_turn(ctrl)
    human_turn(ctrl)
    assert pump(qapp, lambda: ctrl.phase == str(Phase.AI_MOVE_PROPOSED))
    ctrl.confirmProposal()

    target = tmp_path / "match.json"
    assert ctrl.saveGame(str(target))
    assert target.exists()

    expected_state = ctrl.session.state
    expected_moves = [r.move for r in ctrl.session.history]

    ctrl.newGame()
    assert ctrl.session.history == ()

    assert ctrl.loadGame(target.as_uri())
    assert ctrl.session.state == expected_state
    assert [r.move for r in ctrl.session.history] == expected_moves
    assert ctrl.historyModel.rowCount() == 3
    assert ctrl.currentPlayerId == expected_state.current_player_id


def test_a_long_random_match_never_produces_an_illegal_state(qapp, ctrl):
    """Drive many turns and check the invariants hold the whole way."""
    for _ in range(30):
        if ctrl.phase == str(Phase.GAME_OVER):
            break
        if ctrl.isCurrentPlayerAi:
            assert pump(qapp, lambda: ctrl.phase == str(Phase.AI_MOVE_PROPOSED))
            ctrl.confirmProposal()
        else:
            human_turn(ctrl)

        state = ctrl.session.state
        assert sum(1 for v in state.occupancy if v != 0) == CAMP_SIZE * 3
        for player_id in (1, 2, 3):
            assert len(state.positions_of(player_id)) == CAMP_SIZE
