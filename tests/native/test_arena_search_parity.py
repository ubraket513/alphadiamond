"""The native search answers a Python evaluator exactly as the Python one does.

This is what lets the arena move to the C++ core. The arena is not self-play:
two different networks alternate moves inside one game, so the search has to
suspend on every node and ask Python for that node's answer. The engine
underneath must not change the game that gets played.

Compared per position: the selected action, the visit distribution, the
expansion order, and how many times the evaluator was asked. Visit counts alone
would not be enough -- two searches can reach the same counts through different
traversals -- so the evaluator call count and the root action order are checked
too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from diamond.alphazero.config import MCTSConfig
from diamond.alphazero.evaluator.base import EvalRequest, EvalResult
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.mcts.search_2p import MCTS2P
from diamond.alphazero.native.search import NativeSearch2P
from diamond.contract.state import GameState, GameStatus, build_players

FIXTURE = Path(__file__).parent / "fixtures" / "positions.jsonl"
SIMULATIONS = (1, 2, 16, 64)
POSITION_STRIDE = 211
"""Sampled: each case runs two full searches, one of them across the boundary."""


class RecordingEvaluator:
    """A pure function of the request, so both engines see identical answers.

    Deliberately not a constant: a constant makes every PUCT key tie, and a
    divergent traversal could then still produce matching visit counts.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[tuple[int, ...]] = []

    def evaluate(self, requests: tuple[EvalRequest, ...]) -> tuple[EvalResult, ...]:
        results = []
        for request in requests:
            self.calls += 1
            self.requests.append(request.legal_action_ids)
            digest = 1469598103934665603
            for row in request.node_features:
                for value in row:
                    digest = ((digest ^ (1 if value == 1.0 else 0)) * 1099511628211) & (2**64 - 1)
            for action in request.legal_action_ids:
                digest = ((digest ^ (action & 0xFF)) * 1099511628211) & (2**64 - 1)
            weights = [
                (((digest >> (index % 40)) & 0xFF) + 1)
                for index in range(len(request.legal_action_ids))
            ]
            total = sum(weights)
            results.append(
                EvalResult(
                    priors={
                        action: weight / total
                        for action, weight in zip(request.legal_action_ids, weights)
                    },
                    value=(digest % 2001) / 1000.0 - 1.0,
                )
            )
        return tuple(results)


def _corpus() -> list[dict]:
    if not FIXTURE.exists():  # pragma: no cover - regenerate with tools/
        pytest.skip(f"missing corpus: {FIXTURE}")
    records = [
        json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line
    ]
    # MCTS2P is two-player only, and a terminal position cannot be searched.
    return [r for r in records if r["player_count"] == 2 and r["status"] != "finished"]


SAMPLE = _corpus()[::POSITION_STRIDE]


def _state(record: dict) -> GameState:
    return GameState(
        occupancy=tuple(record["occupancy"]),
        current_player_id=record["current_player_id"],
        turn_number=record["turn_number"],
        status=GameStatus(record["status"]),
        finish_order=tuple(record["finish_order"]),
    )


@pytest.fixture(scope="module")
def adapter() -> DiamondSearchAdapter:
    return DiamondSearchAdapter(AlphaZeroGameAdapter(build_players(2)))


def _cases():
    for record in SAMPLE:
        for simulations in SIMULATIONS:
            yield record, simulations


def test_sample_is_populated() -> None:
    assert len(SAMPLE) >= 4


def test_native_search_plays_the_same_move(adapter: DiamondSearchAdapter) -> None:
    for record, simulations in _cases():
        where = f"{record['tag']} sims={simulations}"
        state = _state(record)
        config = MCTSConfig(
            simulations=simulations, c_puct=1.5, dirichlet_epsilon=0.0, seed=0
        )

        python_evaluator = RecordingEvaluator()
        expected = MCTS2P(adapter, python_evaluator, config).run(state, temperature=0.0)

        native_evaluator = RecordingEvaluator()
        actual = NativeSearch2P(adapter, native_evaluator, config).run(state, temperature=0.0)

        assert actual.selected_action == expected.selected_action, where
        assert actual.visit_counts == expected.visit_counts, where
        # Expansion order is observable and must match, not just the counts.
        assert list(actual.visit_counts) == list(expected.visit_counts), where
        assert native_evaluator.calls == python_evaluator.calls, where
        assert native_evaluator.requests == python_evaluator.requests, where


FLOAT32_TOLERANCE = 1e-6
"""The callback ABI carries features, priors and values as float32 -- the same
shape the self-play pool uses, and the precision the network itself computes in.
The Python search is the outlier: it widens those float32 numbers to double and
keeps them there. So the two agree to float32, not bit for bit, and the test
above is what matters -- the *decisions* (selected action, visit counts,
evaluator request sequence) are still compared exactly."""


def test_policies_and_values_agree(adapter: DiamondSearchAdapter) -> None:
    for record, simulations in _cases():
        where = f"{record['tag']} sims={simulations}"
        state = _state(record)
        config = MCTSConfig(
            simulations=simulations, c_puct=1.5, dirichlet_epsilon=0.0, seed=0
        )
        expected = MCTS2P(adapter, RecordingEvaluator(), config).run(state, temperature=0.0)
        actual = NativeSearch2P(adapter, RecordingEvaluator(), config).run(state, temperature=0.0)

        for action, value in actual.q_values.items():
            assert value == pytest.approx(
                expected.q_values[action], abs=FLOAT32_TOLERANCE
            ), f"{where} {action}"
        for action, value in actual.policy.items():
            assert value == pytest.approx(
                expected.policy[action], abs=FLOAT32_TOLERANCE
            ), f"{where} {action}"
