"""The GUI-facing agent plays the Soo/Min models and says so."""

from __future__ import annotations

import pytest

from diamond.agents.alphazero_agent import AlphaZeroAgent
from diamond.agents.base import MoveProposal, MoveRequest
from diamond.alphazero.config import (
    BOOTSTRAP_PRIOR_NONE,
    CANONICAL_TARGET_VACANCY_DISTANCE_V2,
)
from diamond.alphazero.evaluator.dummy import DummyEvaluator
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter
from diamond.alphazero.identity import MIN_MODEL_NAME, SOO_MODEL_NAME
from diamond.contract.board import standard_board
from diamond.contract.state import build_players, initial_state
from diamond.game.rules import legal_moves
from diamond.game.session import GameSession


@pytest.fixture(scope="module")
def board():
    return standard_board()


def request_for(board, players, state=None) -> MoveRequest:
    state = initial_state(players, board) if state is None else state
    return MoveRequest(board=board, state=state, legal_moves=legal_moves(board, state))


def neutral(player_count: int):
    return 0.0 if player_count == 2 else (0.0, 0.0, 0.0)


# -- naming: the models are Soo and Min, never "AlphaZero" -------------------


@pytest.mark.parametrize(
    ("player_count", "model"), [(2, SOO_MODEL_NAME), (3, MIN_MODEL_NAME)]
)
def test_seat_count_selects_the_model(board, player_count: int, model: str) -> None:
    agent = AlphaZeroAgent(build_players(player_count), simulations=8)
    assert agent.model_name == model


@pytest.mark.parametrize(
    ("player_count", "model"), [(2, SOO_MODEL_NAME), (3, MIN_MODEL_NAME)]
)
def test_untrained_agent_is_named_for_its_model_and_flagged_bootstrap(
    board, player_count: int, model: str
) -> None:
    agent = AlphaZeroAgent(build_players(player_count), simulations=8)
    assert agent.name == f"{model} (bootstrap)"


@pytest.mark.parametrize(
    ("player_count", "model"), [(2, SOO_MODEL_NAME), (3, MIN_MODEL_NAME)]
)
def test_agent_with_an_evaluator_drops_the_bootstrap_suffix(
    board, player_count: int, model: str
) -> None:
    agent = AlphaZeroAgent(
        build_players(player_count), evaluator=DummyEvaluator(neutral(player_count))
    )
    assert agent.name == model


def test_the_display_name_never_says_alphazero(board) -> None:
    """AlphaZero is the method; Soo and Min are the models the operator sees."""
    for player_count in (2, 3):
        agent = AlphaZeroAgent(build_players(player_count), simulations=8)
        assert "AlphaZero" not in agent.name


# -- proposals ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("player_count", "model"), [(2, SOO_MODEL_NAME), (3, MIN_MODEL_NAME)]
)
def test_proposal_reports_the_model_in_metadata(
    board, player_count: int, model: str
) -> None:
    players = build_players(player_count)
    agent = AlphaZeroAgent(players, simulations=8)
    proposal = agent.choose_move(request_for(board, players))
    assert proposal is not None
    assert proposal.metadata["model"] == model
    assert proposal.metadata["agent"] == agent.name
    assert proposal.metadata["bootstrap_prior"] == CANONICAL_TARGET_VACANCY_DISTANCE_V2


@pytest.mark.parametrize("player_count", [2, 3])
def test_proposal_is_a_legal_move(board, player_count: int) -> None:
    players = build_players(player_count)
    request = request_for(board, players)
    proposal = AlphaZeroAgent(players, simulations=8).choose_move(request)
    assert isinstance(proposal, MoveProposal)
    legal = {(m.source, m.destination) for m in request.legal_moves}
    assert (proposal.source, proposal.destination) in legal


def test_no_legal_moves_yields_no_proposal(board) -> None:
    players = build_players(2)
    state = initial_state(players, board)
    empty = MoveRequest(board=board, state=state, legal_moves=())
    assert AlphaZeroAgent(players, simulations=8).choose_move(empty) is None


def test_same_seed_gives_the_same_proposal(board) -> None:
    players = build_players(2)
    request = request_for(board, players)
    first = AlphaZeroAgent(players, simulations=16, seed=3).choose_move(request)
    second = AlphaZeroAgent(players, simulations=16, seed=3).choose_move(request)
    assert (first.source, first.destination) == (second.source, second.destination)


def test_think_again_avoids_the_previous_suggestion(board) -> None:
    players = build_players(2)
    request = request_for(board, players)
    agent = AlphaZeroAgent(players, simulations=32)
    first = agent.choose_move(request)
    again = agent.choose_move(
        MoveRequest(
            board=board,
            state=request.state,
            legal_moves=request.legal_moves,
            avoid=(first.to_move(),),
        )
    )
    assert (again.source, again.destination) != (first.source, first.destination)


# -- configuration -----------------------------------------------------------


def test_bootstrap_prior_can_be_disabled(board) -> None:
    players = build_players(2)
    agent = AlphaZeroAgent(players, simulations=8, bootstrap_prior=BOOTSTRAP_PRIOR_NONE)
    proposal = agent.choose_move(request_for(board, players))
    assert proposal.metadata["bootstrap_prior"] == BOOTSTRAP_PRIOR_NONE


@pytest.mark.parametrize("player_count", [1, 4])
def test_unsupported_seat_counts_are_rejected(player_count: int) -> None:
    with pytest.raises(ValueError):
        AlphaZeroAgent(tuple(build_players(2)[:1]) * player_count)


def test_non_positive_simulations_are_rejected() -> None:
    with pytest.raises(ValueError):
        AlphaZeroAgent(build_players(2), simulations=0)


def test_the_agent_finishes_a_two_player_game(board) -> None:
    """The default settings must actually terminate, not just look busy."""
    players = build_players(2)
    adapter = AlphaZeroGameAdapter(players)
    agent = AlphaZeroAgent(players)
    state = initial_state(players, board)
    moves = 0
    while moves < 400 and not adapter.is_terminal(state):
        proposal = agent.choose_move(request_for(board, players, state))
        if proposal is None:
            break
        session = GameSession(players, board=board, initial=state)
        session.commit(proposal.to_move())
        state = session.state
        moves += 1
    assert adapter.is_terminal(state), f"unfinished after {moves} moves"
