"""Min's native search plays the same game as the Python one.

The three-player search differs from Soo's in what a value *is*: one component
per seat, backed through every ancestor unchanged rather than negated once per
edge. Two things can go wrong there and neither shows up as a crash — the
components can be assigned to the wrong seats (the encoder rotates them per
node, so the root's order is not every node's order), and a node can maximise
somebody else's component. Both would still produce a plausible-looking search.

So the comparison is per position: the selected action, the visit distribution,
the evaluator request sequence, and every q *vector* by seat.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from diamond.alphazero.config import MCTSConfig
from diamond.alphazero.evaluator.base import EvalRequest, EvalResult
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.mcts.search_3p import MCTS3P
from diamond.alphazero.native.search import NativeSearch3P
from diamond.game.state import GameState, GameStatus, build_players

FIXTURE = Path(__file__).parent / "fixtures" / "positions.jsonl"
SIMULATIONS = (1, 2, 16, 64)
POSITION_STRIDE = 211

FLOAT32_TOLERANCE = 1e-6
"""The callback ABI carries values as float32, as the network computes them."""


class RecordingEvaluator:
    """A pure function of the request, and asymmetric between seats.

    Symmetric values would hide exactly the bug this test exists to catch: a
    search that maximises the wrong seat's component still looks right when
    every component is the same number.
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
                    value=tuple(((digest >> (8 * seat)) % 2001) / 1000.0 - 1.0 for seat in range(3)),
                )
            )
        return tuple(results)


def _corpus() -> list[dict]:
    if not FIXTURE.exists():  # pragma: no cover - regenerate with tools/
        pytest.skip(f"missing corpus: {FIXTURE}")
    records = [
        json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line
    ]
    return [r for r in records if r["player_count"] == 3 and r["status"] != "finished"]


def _sample() -> list[dict]:
    """Strided, plus every position with a seat already placed.

    Only 7 of the corpus's 601 searchable 3P positions have a finished seat,
    and they are all at the tail -- a plain stride misses them completely, and
    with them the placement vector this search exists to back up.
    """
    corpus = _corpus()
    chosen = corpus[::POSITION_STRIDE]
    placed = [record for record in corpus if record["finish_order"]]
    seen = {id(record) for record in chosen}
    return chosen + [record for record in placed if id(record) not in seen]


SAMPLE = _sample()


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
    return DiamondSearchAdapter(AlphaZeroGameAdapter(build_players(3)))


def _cases():
    for record in SAMPLE:
        for simulations in SIMULATIONS:
            yield record, simulations


def test_sample_is_populated() -> None:
    assert len(SAMPLE) >= 3
    assert any(record["finish_order"] for record in SAMPLE), (
        "no position with a seat already placed: the placement vector would go unexercised"
    )


def test_native_3p_search_plays_the_same_move(adapter: DiamondSearchAdapter) -> None:
    for record, simulations in _cases():
        where = f"{record['tag']} sims={simulations}"
        state = _state(record)
        config = MCTSConfig(simulations=simulations, c_puct=1.5, dirichlet_epsilon=0.0, seed=0)

        python_evaluator = RecordingEvaluator()
        expected = MCTS3P(adapter, python_evaluator, config).run(state, temperature=0.0)

        native_evaluator = RecordingEvaluator()
        actual = NativeSearch3P(adapter, native_evaluator, config).run(state, temperature=0.0)

        assert actual.selected_action == expected.selected_action, where
        assert actual.visit_counts == expected.visit_counts, where
        assert list(actual.visit_counts) == list(expected.visit_counts), where
        assert native_evaluator.calls == python_evaluator.calls, where
        assert native_evaluator.requests == python_evaluator.requests, where


def test_every_seat_gets_its_own_value(adapter: DiamondSearchAdapter) -> None:
    """The q vectors must agree seat by seat, not just in aggregate."""
    for record, simulations in _cases():
        where = f"{record['tag']} sims={simulations}"
        state = _state(record)
        config = MCTSConfig(simulations=simulations, c_puct=1.5, dirichlet_epsilon=0.0, seed=0)
        expected = MCTS3P(adapter, RecordingEvaluator(), config).run(state, temperature=0.0)
        actual = NativeSearch3P(adapter, RecordingEvaluator(), config).run(state, temperature=0.0)

        for action, vector in actual.q_values.items():
            assert set(vector) == set(expected.q_values[action]), f"{where} {action}: seats differ"
            for seat, value in vector.items():
                assert value == pytest.approx(
                    expected.q_values[action][seat], abs=FLOAT32_TOLERANCE
                ), f"{where} action {action} seat {seat}"


def test_the_search_declines_a_two_seat_game() -> None:
    two_seat = DiamondSearchAdapter(AlphaZeroGameAdapter(build_players(2)))
    assert not NativeSearch3P.can_drive(two_seat)
    with pytest.raises(ValueError, match="three-seat"):
        NativeSearch3P(two_seat, RecordingEvaluator(), MCTSConfig(simulations=1))
