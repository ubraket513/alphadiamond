from __future__ import annotations

from dataclasses import dataclass

import pytest

from diamond.alphazero.config import MCTSConfig
from diamond.alphazero.evaluator.base import EvalRequest, EvalResult
from diamond.alphazero.evaluator.dummy import DummyEvaluator
from diamond.alphazero.mcts.search_2p import MCTS2P


@dataclass(frozen=True)
class State:
    name: str
    player: int


class Toy2PGame:
    def __init__(self):
        self.children = {
            ("root", 10): State("a_wins", 2),
            ("root", 20): State("b_wins", 2),
        }

    def current_player_id(self, state):
        return state.player

    def legal_action_ids(self, state):
        return tuple(action for (name, action) in self.children if name == state.name)

    def apply_action(self, state, action):
        return self.children[(state.name, action)]

    def is_terminal(self, state):
        return state.name.endswith("wins")

    def terminal_scalar_value(self, state, player_id):
        winner = 1 if state.name == "a_wins" else 2
        return 1.0 if player_id == winner else -1.0

    def evaluation_request(self, state):
        return EvalRequest(((0.0,),), self.legal_action_ids(state), (state.player, 3 - state.player))


def test_two_player_backup_flips_child_perspective_sign() -> None:
    game = Toy2PGame()
    search = MCTS2P(
        game,
        DummyEvaluator(0.0),
        MCTSConfig(simulations=48, c_puct=1.0, dirichlet_epsilon=0.0, seed=4),
    )

    result = search.run(State("root", 1), temperature=0.0)

    assert result.selected_action == 10
    assert sum(result.visit_counts.values()) == 48
    assert result.visit_counts[10] > result.visit_counts[20]
    assert result.q_values[10] > 0
    assert result.q_values[20] < 0
    assert sum(result.policy.values()) == pytest.approx(1.0)


def test_two_player_search_is_reproducible_with_fixed_seed() -> None:
    config = MCTSConfig(simulations=20, c_puct=1.0, dirichlet_epsilon=0.25, seed=9)
    first = MCTS2P(Toy2PGame(), DummyEvaluator(0.0), config).run(State("root", 1))
    second = MCTS2P(Toy2PGame(), DummyEvaluator(0.0), config).run(State("root", 1))
    assert first == second



class FakeClock:
    """Drives deadline expiry without spending real time."""

    def __init__(self, start: float = 500.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_search_without_a_deadline_runs_every_simulation() -> None:
    """The unlimited path must stay byte-for-byte the previous behaviour."""
    search = MCTS2P(
        Toy2PGame(),
        DummyEvaluator(0.0),
        MCTSConfig(simulations=32, c_puct=1.0, dirichlet_epsilon=0.0, seed=4),
    )

    result = search.run(State("root", 1), temperature=0.0)

    assert sum(result.visit_counts.values()) == 32


def test_an_expired_deadline_truncates_the_simulation_loop() -> None:
    from diamond.alphazero.deadline import Deadline

    clock = FakeClock()
    deadline = Deadline.start(10.0, clock=clock)
    search = MCTS2P(
        Toy2PGame(),
        DummyEvaluator(0.0),
        MCTSConfig(simulations=64, c_puct=1.0, dirichlet_epsilon=0.0, seed=4),
        deadline=deadline,
    )
    clock.advance(11.0)  # already past the budget when the search starts

    result = search.run(State("root", 1), temperature=0.0)

    # Truncated, but still a usable distribution: the root is always expanded.
    assert sum(result.visit_counts.values()) < 64
    assert result.selected_action in (10, 20)
    assert sum(result.policy.values()) == pytest.approx(1.0)


def test_a_deadline_that_expires_midway_stops_between_simulations() -> None:
    from diamond.alphazero.deadline import Deadline

    class CountingClock(FakeClock):
        def __init__(self) -> None:
            super().__init__()
            self.reads = 0

        def __call__(self) -> float:
            self.reads += 1
            # Expire after a handful of simulation-loop checks.
            if self.reads > 6:
                self.now = 1000.0
            return self.now

    clock = CountingClock()
    search = MCTS2P(
        Toy2PGame(),
        DummyEvaluator(0.0),
        MCTSConfig(simulations=64, c_puct=1.0, dirichlet_epsilon=0.0, seed=4),
        deadline=Deadline.start(10.0, clock=clock),
    )

    result = search.run(State("root", 1), temperature=0.0)

    total = sum(result.visit_counts.values())
    assert 0 < total < 64


class CountingGame(Toy2PGame):
    """Counts authoritative generations so a duplicate one is visible."""

    def __init__(self) -> None:
        super().__init__()
        self.legal_calls = 0
        self.request_calls = 0

    def legal_action_ids(self, state):
        self.legal_calls += 1
        return super().legal_action_ids(state)

    def evaluation_request(self, state):
        self.request_calls += 1
        return super().evaluation_request(state)


class MismatchedPriorEvaluator:
    """Answers with an action the request never offered."""

    def evaluate(self, requests):
        return tuple(
            EvalResult(priors={action: 1.0 for action in (-1,)}, value=0.0)
            for _ in requests
        )


def test_expansion_does_not_regenerate_the_authoritative_legal_set() -> None:
    """Legal moves are generated to build the request, and not a second time.

    ``_expand`` used to validate the evaluator's priors against a fresh
    ``legal_action_ids`` call on the same state, so every expansion ran the
    authoritative generator twice. On the real Diamond board that second run is
    a full chained-jump BFS, measured at ~18% of worker-side search CPU.
    """
    game = CountingGame()

    MCTS2P(
        game,
        DummyEvaluator(0.0),
        MCTSConfig(simulations=16, c_puct=1.0, dirichlet_epsilon=0.0, seed=4),
    ).run(State("root", 1), temperature=0.0)

    assert game.request_calls > 0
    assert game.legal_calls == game.request_calls


def test_expansion_still_rejects_priors_that_do_not_match_the_request() -> None:
    """The invariant is preserved: priors must cover exactly the actions asked."""
    with pytest.raises(ValueError, match="evaluator priors must match"):
        MCTS2P(
            Toy2PGame(),
            MismatchedPriorEvaluator(),
            MCTSConfig(simulations=4, c_puct=1.0, dirichlet_epsilon=0.0, seed=4),
        ).run(State("root", 1), temperature=0.0)
