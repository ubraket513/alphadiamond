from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from diamond.alphazero.arena import MinArena, SooArena, _balanced_matchups
from diamond.alphazero.config import ArenaConfig, MCTSConfig
from diamond.alphazero.evaluator.base import EvalRequest
from diamond.alphazero.evaluator.dummy import DummyEvaluator


@dataclass(frozen=True)
class State:
    terminal: bool
    player_id: int


class _FirstLegalSearch:
    """Plays the first legal action, and reports it as the whole distribution.

    These tests exercise the arena's bookkeeping, not a search. Before the
    Python search was retired they got one by accident: a fake game fell back to
    it. Stating the stub is better than reintroducing an engine to satisfy a
    test that never wanted one.
    """

    def __init__(self, game, evaluator, config, **kwargs) -> None:
        self.game = game
        self.config = config

    def run(self, state, temperature: float = 0.0):
        action = self.game.legal_action_ids(state)[0]
        return SimpleNamespace(
            selected_action=action,
            visit_counts={action: 1},
            policy={action: 1.0},
            q_values={action: 0.0},
        )


class OneMoveArenaGame:
    def __init__(self, order: tuple[int, ...], final_order: tuple[int, ...]) -> None:
        self.order = order
        self._final_order = final_order

    def initial_state(self):
        return State(False, self.order[0])

    def current_player_id(self, state):
        return state.player_id

    def legal_action_ids(self, state):
        return () if state.terminal else (5,)

    def apply_action(self, state, action_id):
        assert action_id == 5
        return State(True, self.order[1])

    def is_terminal(self, state):
        return state.terminal

    def evaluation_request(self, state):
        start = self.order.index(state.player_id)
        canonical = tuple(
            self.order[(start + offset) % len(self.order)]
            for offset in range(len(self.order))
        )
        return EvalRequest(((0.0,),), self.legal_action_ids(state), canonical)

    def terminal_scalar_value(self, state, player_id):
        return 1.0 if player_id == self._final_order[0] else -1.0

    def terminal_vector_value(self, state):
        return dict(zip(self._final_order, (1.0, 0.0, -1.0)))

    def final_order(self, state):
        assert state.terminal
        return self._final_order


def test_soo_arena_balances_candidate_seat_and_turn_order() -> None:
    observed_orders: list[tuple[int, ...]] = []

    def factory(order):
        observed_orders.append(order)
        return OneMoveArenaGame(order, (1, 2))

    result = SooArena(
        candidate=DummyEvaluator(0.0),
        baseline=DummyEvaluator(0.0),
        mcts_config=MCTSConfig(simulations=2, dirichlet_epsilon=0.25),
        arena_config=ArenaConfig(games=4, seed=8),
        search_factory=_FirstLegalSearch,
    ).run(factory)

    assert observed_orders == [(1, 2), (1, 2), (2, 1), (2, 1)]
    assert result.wins == 2
    assert result.losses == 2
    assert result.aborted_games == 0
    assert result.win_rate == pytest.approx(0.5)


def test_min_arena_rotates_candidate_and_all_turn_orders() -> None:
    observed_orders: list[tuple[int, ...]] = []

    def factory(order):
        observed_orders.append(order)
        return OneMoveArenaGame(order, (1, 2, 3))

    result = MinArena(
        candidate=DummyEvaluator((0.0, 0.0, 0.0)),
        baseline=DummyEvaluator((0.0, 0.0, 0.0)),
        mcts_config=MCTSConfig(simulations=2, dirichlet_epsilon=0.25),
        arena_config=ArenaConfig(games=18, seed=4),
        search_factory=_FirstLegalSearch,
    ).run(factory)

    assert observed_orders == [
        order
        for order in itertools.permutations((1, 2, 3))
        for _candidate in (1, 2, 3)
    ]
    assert result.first_places == 6
    assert result.second_places == 6
    assert result.third_places == 6
    assert result.aborted_games == 0
    assert result.mean_utility == pytest.approx(0.0)


@pytest.mark.parametrize("player_ids", [(1, 2), (1, 2, 3)])
def test_balanced_matchups_cross_every_seat_with_every_turn_order(player_ids) -> None:
    matchups = _balanced_matchups(player_ids)

    expected_orders = set(itertools.permutations(player_ids))
    assert {order for order, _candidate in matchups} == expected_orders
    assert Counter(candidate for _order, candidate in matchups) == {
        player_id: len(expected_orders) for player_id in player_ids
    }
    assert Counter(order.index(candidate) for order, candidate in matchups) == {
        position: len(matchups) // len(player_ids)
        for position in range(len(player_ids))
    }


def test_arenas_reject_partial_balance_cycles() -> None:
    with pytest.raises(ValueError, match="multiple of 4"):
        SooArena(
            candidate=DummyEvaluator(0.0),
            baseline=DummyEvaluator(0.0),
            mcts_config=MCTSConfig(),
            arena_config=ArenaConfig(games=3),
            search_factory=_FirstLegalSearch,
        )

    with pytest.raises(ValueError, match="multiple of 18"):
        MinArena(
            candidate=DummyEvaluator((0.0, 0.0, 0.0)),
            baseline=DummyEvaluator((0.0, 0.0, 0.0)),
            mcts_config=MCTSConfig(),
            arena_config=ArenaConfig(games=5),
            search_factory=_FirstLegalSearch,
        )
