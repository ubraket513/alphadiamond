from __future__ import annotations

from diamond.agents.base import MoveProposal, MoveRequest
from diamond.agents.random_agent import RandomAgent
from diamond.contract.state import initial_state
from diamond.game.rules import legal_moves, validate_move
from diamond.game.session import GameSession


def request_for(board, state, avoid=(), seed=None):
    return MoveRequest(board, state, legal_moves(board, state), avoid=avoid, seed=seed)


def test_agent_returns_a_legal_move_for_the_current_player(board):
    state = initial_state()
    agent = RandomAgent(seed=1)
    proposal = agent.choose_move(request_for(board, state))
    assert proposal is not None
    assert proposal.player_id == state.current_player_id
    assert state.occupant(proposal.source) == state.current_player_id
    validate_move(board, state, proposal.to_move())


def test_agent_never_moves_another_players_piece(board):
    session = GameSession()
    agent = RandomAgent(seed=7)
    for _ in range(30):
        state = session.state
        proposal = agent.choose_move(request_for(session.board, state))
        assert proposal is not None
        assert state.occupant(proposal.source) == state.current_player_id
        session.commit(proposal.to_move())


def test_same_seed_reproduces_the_same_game(board):
    def play(seed: int) -> list[tuple[int, int]]:
        session = GameSession()
        agent = RandomAgent(seed=seed)
        moves = []
        for _ in range(25):
            proposal = agent.choose_move(request_for(session.board, session.state))
            assert proposal is not None
            moves.append((proposal.source, proposal.destination))
            session.commit(proposal.to_move())
        return moves

    assert play(42) == play(42)
    assert play(42) != play(43)


def test_agent_does_not_touch_the_global_rng(board):
    import random

    random.seed(0)
    expected = random.random()

    random.seed(0)
    RandomAgent(seed=99).choose_move(request_for(board, initial_state()))
    assert random.random() == expected


def test_per_call_seed_is_deterministic(board):
    state = initial_state()
    agent = RandomAgent(seed=None)
    first = agent.choose_move(request_for(board, state, seed=5))
    second = agent.choose_move(request_for(board, state, seed=5))
    assert (first.source, first.destination) == (second.source, second.destination)


def test_avoid_steers_the_agent_to_a_different_move(board):
    state = initial_state()
    agent = RandomAgent(seed=3)
    first = agent.choose_move(request_for(board, state))
    for _ in range(10):
        again = agent.choose_move(request_for(board, state, avoid=(first.to_move(),)))
        assert (again.source, again.destination) != (first.source, first.destination)


def test_avoid_falls_back_when_only_one_move_exists(board):
    state = initial_state()
    only = legal_moves(board, state)[0]
    agent = RandomAgent(seed=3)
    proposal = agent.choose_move(
        MoveRequest(board, state, legal_moves=(only,), avoid=(only,))
    )
    assert proposal is not None
    assert (proposal.source, proposal.destination) == (only.source, only.destination)


def test_agent_returns_none_without_legal_moves(board):
    state = initial_state()
    assert RandomAgent(seed=1).choose_move(MoveRequest(board, state, legal_moves=())) is None


def test_metadata_reports_only_real_facts(board):
    state = initial_state()
    proposal = RandomAgent(seed=11).choose_move(request_for(board, state))
    assert proposal.metadata["agent"] == "RandomAgent"
    assert proposal.metadata["seed"] == 11
    assert proposal.metadata["legal_move_count"] == len(legal_moves(board, state))
    assert "value" not in proposal.metadata  # no fabricated evaluation


def test_proposal_survives_without_metadata(board):
    proposal = MoveProposal(1, 10, 12, (10, 12))
    assert proposal.metadata == {}
    assert proposal.to_move().path == (10, 12)
