from __future__ import annotations

from dataclasses import dataclass

import pytest

from diamond.alphazero.config import MCTSConfig
from diamond.alphazero.evaluator.base import EvalRequest, EvalResult
from diamond.alphazero.evaluator.dummy import DummyEvaluator
from diamond.alphazero.mcts.search_3p import MCTS3P


@dataclass(frozen=True)
class State:
    name: str
    player: int


class Toy3PGame:
    def __init__(self, acting_player: int = 1):
        self.acting_player = acting_player
        self.children = {
            ("root", 10): State("a_first", 2),
            ("root", 20): State("b_first", 3),
            ("root", 30): State("c_first", 1),
        }
        self.utilities = {
            "a_first": {1: 1.0, 2: 0.0, 3: -1.0},
            "b_first": {1: -1.0, 2: 1.0, 3: 0.0},
            "c_first": {1: 0.0, 2: -1.0, 3: 1.0},
        }

    def current_player_id(self, state):
        return state.player

    def legal_action_ids(self, state):
        return tuple(action for (name, action) in self.children if name == state.name)

    def apply_action(self, state, action):
        return self.children[(state.name, action)]

    def is_terminal(self, state):
        return state.name != "root"

    def terminal_vector_value(self, state):
        return self.utilities[state.name]

    def evaluation_request(self, state):
        order = (state.player, state.player % 3 + 1, (state.player + 1) % 3 + 1)
        return EvalRequest(((0.0,),), self.legal_action_ids(state), order)


@pytest.mark.parametrize("acting_player,expected_action", [(1, 10), (2, 20), (3, 30)])
def test_three_player_selection_uses_acting_players_own_component(
    acting_player: int, expected_action: int
) -> None:
    game = Toy3PGame(acting_player)
    result = MCTS3P(
        game,
        DummyEvaluator((0.0, 0.0, 0.0)),
        MCTSConfig(simulations=90, c_puct=1.0, dirichlet_epsilon=0.0, seed=2),
    ).run(State("root", acting_player), temperature=0.0)

    assert result.selected_action == expected_action
    assert sum(result.visit_counts.values()) == 90
    assert result.q_values[expected_action][acting_player] > 0


def test_three_player_backup_preserves_global_identity_without_negation() -> None:
    result = MCTS3P(
        Toy3PGame(),
        DummyEvaluator((0.0, 0.0, 0.0)),
        MCTSConfig(simulations=60, c_puct=1.0, dirichlet_epsilon=0.0, seed=3),
    ).run(State("root", 1), temperature=0.0)

    first = result.q_values[10]
    assert first[1] == pytest.approx(1.0)
    assert first[2] == pytest.approx(0.0)
    assert first[3] == pytest.approx(-1.0)


class CountingGame3P(Toy3PGame):
    """Counts authoritative generations so a duplicate one is visible."""

    def __init__(self, acting_player: int = 1) -> None:
        super().__init__(acting_player)
        self.legal_calls = 0
        self.request_calls = 0

    def legal_action_ids(self, state):
        self.legal_calls += 1
        return super().legal_action_ids(state)

    def evaluation_request(self, state):
        self.request_calls += 1
        return super().evaluation_request(state)


class MismatchedPriorEvaluator3P:
    """Answers with an action the request never offered."""

    def evaluate(self, requests):
        return tuple(
            EvalResult(priors={-1: 1.0}, value=(0.0, 0.0, 0.0)) for _ in requests
        )


def test_expansion_does_not_regenerate_the_authoritative_legal_set() -> None:
    """Same invariant as MCTS2P: the request's legal set is checked, not re-derived."""
    game = CountingGame3P()

    MCTS3P(
        game,
        DummyEvaluator((0.0, 0.0, 0.0)),
        MCTSConfig(simulations=12, c_puct=1.0, dirichlet_epsilon=0.0, seed=4),
    ).run(State("root", 1), temperature=0.0)

    assert game.request_calls > 0
    assert game.legal_calls == game.request_calls


def test_expansion_still_rejects_priors_that_do_not_match_the_request() -> None:
    with pytest.raises(ValueError, match="evaluator priors must match"):
        MCTS3P(
            Toy3PGame(),
            MismatchedPriorEvaluator3P(),
            MCTSConfig(simulations=4, c_puct=1.0, dirichlet_epsilon=0.0, seed=4),
        ).run(State("root", 1), temperature=0.0)
