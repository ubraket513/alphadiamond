"""A wall-clock budget stops the native search, and still returns a usable move.

`MCTS2P` and `MCTS3P` take a `Deadline` and check it once per simulation, with
one exception that matters: the first simulation always runs, so the visit
distribution is never empty even when the budget is already spent. A native
search without that rule cannot serve a caller that bounds a game by time --
which is why the selector used to refuse those callers outright.
"""

from __future__ import annotations

import time

import pytest

from diamond.alphazero.config import MCTSConfig
from diamond.alphazero.deadline import Deadline
from diamond.alphazero.evaluator.base import EvalResult
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.native.search import NativeSearch2P
from diamond.alphazero.search_factory import two_player_search
from diamond.contract.state import build_players

SIMULATIONS = 20_000
"""Far more than a small budget can finish, so the cut is unambiguous."""


class Cheap:
    def evaluate(self, requests):
        results = []
        for request in requests:
            count = len(request.legal_action_ids)
            results.append(
                EvalResult(
                    priors={action: 1.0 / count for action in request.legal_action_ids},
                    value=0.25,
                )
            )
        return tuple(results)


@pytest.fixture(scope="module")
def adapter() -> DiamondSearchAdapter:
    return DiamondSearchAdapter(AlphaZeroGameAdapter(build_players(2)))


def _config(simulations: int = SIMULATIONS) -> MCTSConfig:
    return MCTSConfig(simulations=simulations, c_puct=1.5, dirichlet_epsilon=0.0, seed=0)


def test_a_spent_budget_still_returns_a_move(adapter: DiamondSearchAdapter) -> None:
    """The rule the Python search has: never come back empty-handed."""
    state = adapter.initial_state()
    expired = Deadline(started_at=time.monotonic() - 10.0, budget_s=0.001)
    assert expired.expired

    result = NativeSearch2P(adapter, Cheap(), _config(), deadline=expired).run(state)
    assert result.selected_action in result.visit_counts
    assert sum(result.visit_counts.values()) >= 1


def test_the_budget_actually_cuts_the_search_short(adapter: DiamondSearchAdapter) -> None:
    state = adapter.initial_state()
    started = time.perf_counter()
    bounded = NativeSearch2P(
        adapter, Cheap(), _config(), deadline=Deadline.start(0.05)
    ).run(state)
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0, "the budget did not stop the search"
    unbounded_visits = 0
    for action, visits in bounded.visit_counts.items():
        assert visits >= 0, action
        unbounded_visits += visits
    assert unbounded_visits < SIMULATIONS, "the search ran its full budget of simulations"


def test_no_budget_means_no_bound(adapter: DiamondSearchAdapter) -> None:
    """A `None` deadline must not be mistaken for an already-spent one."""
    state = adapter.initial_state()
    result = NativeSearch2P(adapter, Cheap(), _config(64), deadline=None).run(state)
    assert sum(result.visit_counts.values()) == 64


def test_the_selector_now_accepts_a_deadline(adapter: DiamondSearchAdapter) -> None:
    """This is the point of the exercise: `deadline` no longer forces Python."""
    search = two_player_search()(adapter, Cheap(), _config(8), deadline=Deadline.start(5.0))
    assert isinstance(search, NativeSearch2P)

    # An option the native search does not understand routes to Python, where
    # it fails on its own terms rather than being silently dropped here.
    with pytest.raises(TypeError, match="some_future_option"):
        two_player_search()(adapter, Cheap(), _config(8), some_future_option=1)
