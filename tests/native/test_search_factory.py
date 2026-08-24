"""Callers get the C++ core when it can play, and the Python search when it cannot.

The selector is the single place that decides, so this is where the decision is
worth pinning: an "authoritative" core that quietly never gets used would look
exactly like one that does.
"""

from __future__ import annotations

import pytest

from diamond.agents.alphazero_agent import AlphaZeroAgent
from diamond.agents.base import MoveRequest
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.native.search import NativeSearch2P
from diamond.alphazero.search_factory import NativeSearchUnavailable, two_player_search
from diamond.game.board import standard_board
from diamond.game.rules import legal_moves
from diamond.game.state import build_players, initial_state


class _Evaluator:
    def evaluate(self, requests):  # pragma: no cover - never called here
        raise AssertionError("the factory must not evaluate anything")


class _Config:
    simulations = 4
    c_puct = 1.5
    dirichlet_alpha = 0.3
    dirichlet_epsilon = 0.0
    seed = 0


def _adapter(player_count: int = 2) -> DiamondSearchAdapter:
    return DiamondSearchAdapter(AlphaZeroGameAdapter(build_players(player_count)))


def test_the_real_game_goes_to_the_native_core() -> None:
    search = two_player_search()(_adapter(), _Evaluator(), _Config())
    assert isinstance(search, NativeSearch2P)


def test_a_three_seat_game_is_refused_by_the_two_seat_search() -> None:
    """There is no Python fallback to receive it: it is an unsupported input,
    not a slower path (decision 1 in docs/architecture/decisions.md)."""
    assert not NativeSearch2P.can_drive(_adapter(3))
    with pytest.raises(NativeSearchUnavailable, match="two-seat"):
        two_player_search()(_adapter(3), _Evaluator(), _Config())


def test_a_deadline_no_longer_forces_the_python_search() -> None:
    """`SearchSession::set_budget` is why: the bound survives the crossing.

    Covered in full by tests/native/test_native_deadline.py; here it is only the
    selector's half of the contract.
    """
    from diamond.alphazero.deadline import Deadline

    search = two_player_search()(_adapter(), _Evaluator(), _Config(), deadline=Deadline.start(5.0))
    assert isinstance(search, NativeSearch2P)


def test_an_option_the_native_search_does_not_understand_routes_to_python() -> None:
    import pytest

    with pytest.raises(TypeError, match="some_future_option"):
        two_player_search()(_adapter(), _Evaluator(), _Config(), some_future_option=1)


def test_a_reduced_board_falls_back() -> None:
    board = standard_board()
    assert len(board) == 73, "this test's premise is that 73 is the native size"

    class Smaller:
        players = build_players(2)
        board = tuple(range(19))

    assert not NativeSearch2P.can_drive(Smaller())


def test_the_agent_proposes_a_move_through_the_native_search(monkeypatch) -> None:
    """End to end: the agent's two-seat path must actually reach C++."""
    built: list[str] = []
    original = NativeSearch2P.__init__

    def spy(self, game, evaluator, config):
        built.append(type(self).__name__)
        original(self, game, evaluator, config)

    monkeypatch.setattr(NativeSearch2P, "__init__", spy)

    players = build_players(2)
    board = standard_board()
    state = initial_state(players, board)
    agent = AlphaZeroAgent(players, simulations=2, seed=7)
    proposal = agent.choose_move(
        MoveRequest(board=board, state=state, legal_moves=legal_moves(board, state))
    )

    assert proposal is not None
    assert built == ["NativeSearch2P"], "the agent did not use the native search"


def test_a_three_seat_agent_stays_on_python(monkeypatch) -> None:
    built: list[str] = []
    original = NativeSearch2P.__init__

    def spy(self, game, evaluator, config):  # pragma: no cover - must not run
        built.append(type(self).__name__)
        original(self, game, evaluator, config)

    monkeypatch.setattr(NativeSearch2P, "__init__", spy)

    players = build_players(3)
    board = standard_board()
    state = initial_state(players, board)
    agent = AlphaZeroAgent(players, simulations=2, seed=7)
    assert agent.choose_move(
        MoveRequest(board=board, state=state, legal_moves=legal_moves(board, state))
    ) is not None
    assert built == []
