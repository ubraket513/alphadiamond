"""Gate A: native rules, encoding and prior equal the Python oracle exactly.

Section 7 of ``docs/native_selfplay_phase0.md``.  For every corpus position the
two implementations must agree on:

* the player to act, terminal status and finishing order
* legal canonical action ids **as an ordered sequence**, not a set
* the state resulting from every legal action
* canonical player ids and node features (exact float equality)
* the physical <-> canonical position and action mappings
* the vacancy bootstrap prior, within 1e-12

Ordering is not cosmetic: ``add_dirichlet_noise`` draws one ``gammavariate``
per entry of ``priors.items()``, so a native backend that produced the same set
in a different order would land the noise on different actions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from diamond.alphazero.action_codec import ActionCodec, ActionSpaceSpec
from diamond.alphazero.bootstrap.heuristic import (
    CanonicalTargetVacancyDistancePrior,
    pairwise_distance_table,
)
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.native import native_game, require_native
from diamond.game.board import Camp, standard_board
from diamond.game.state import GameState, GameStatus, build_players

FIXTURE = Path(__file__).parent / "fixtures" / "positions.jsonl"
PRIOR_TOLERANCE = 1e-12


def _load_corpus() -> list[dict]:
    if not FIXTURE.exists():  # pragma: no cover - regenerate with tools/
        pytest.skip(f"missing corpus: {FIXTURE}; run tools/build_native_corpus.py")
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line]


CORPUS = _load_corpus()


class _Oracle:
    """The Python side of one seat count, built once per player count."""

    def __init__(self, player_count: int) -> None:
        self.players = build_players(player_count)
        self.game = AlphaZeroGameAdapter(self.players)
        self.search = DiamondSearchAdapter(self.game)
        self.codec = ActionCodec(ActionSpaceSpec.diamond73())
        self.prior = CanonicalTargetVacancyDistancePrior()
        self.pairwise = pairwise_distance_table(self.game.board)
        self.target = frozenset(standard_board().camp_positions(Camp.Z_NEG))
        self.native = native_game(self.players)

    def priors(self, state: GameState) -> list[float]:
        request = self.search.evaluation_request(state)
        values = self.prior.priors(
            request.legal_action_ids,
            self.codec,
            self.target,
            self.pairwise,
            request.node_features,
        )
        return [values[action] for action in request.legal_action_ids]


ORACLES: dict[int, _Oracle] = {}


def _oracle(player_count: int) -> _Oracle:
    if player_count not in ORACLES:
        ORACLES[player_count] = _Oracle(player_count)
    return ORACLES[player_count]


def _python_state(record: dict) -> GameState:
    return GameState(
        occupancy=tuple(record["occupancy"]),
        current_player_id=record["current_player_id"],
        turn_number=record["turn_number"],
        status=GameStatus(record["status"]),
        finish_order=tuple(record["finish_order"]),
    )


def _native_state(record: dict, module) -> object:
    return module.State(
        occupancy=record["occupancy"],
        current_player=record["current_player_id"],
        turn_number=record["turn_number"],
        status=1 if record["status"] == "finished" else 0,
        finish_order=record["finish_order"],
    )


def _pairs():
    module = require_native()
    for record in CORPUS:
        yield record, _oracle(record["player_count"]), _python_state(record), _native_state(
            record, module
        )


def _describe(record: dict) -> str:
    return f"{record['tag']} ({record['player_count']}P)"


def test_corpus_is_populated() -> None:
    assert len(CORPUS) >= 500
    assert any(r["status"] == "finished" for r in CORPUS), "corpus has no terminal position"
    assert any(r["finish_order"] for r in CORPUS), "corpus has no partially-placed position"
    assert {r["player_count"] for r in CORPUS} == {2, 3}


def test_terminal_status_and_player_to_act() -> None:
    for record, oracle, python_state, native_state in _pairs():
        where = _describe(record)
        assert native_state.current_player_id == python_state.current_player_id, where
        assert native_state.status == (1 if python_state.status is GameStatus.FINISHED else 0), where
        assert list(native_state.finish_order) == list(python_state.finish_order), where
        assert oracle.native.is_terminal(native_state) == oracle.game.is_terminal(
            python_state
        ), where
        assert oracle.native.search_current_player_id(
            native_state
        ) == oracle.search.current_player_id(python_state), where


def test_legal_actions_match_as_ordered_sequences() -> None:
    for record, oracle, python_state, native_state in _pairs():
        where = _describe(record)
        assert list(oracle.native.legal_action_ids(native_state)) == list(
            oracle.game.legal_action_ids(python_state)
        ), where
        assert list(oracle.native.canonical_legal_action_ids(native_state)) == list(
            oracle.search.legal_action_ids(python_state)
        ), where


def test_every_legal_action_produces_the_same_successor() -> None:
    for record, oracle, python_state, native_state in _pairs():
        if oracle.game.is_terminal(python_state):
            continue
        where = _describe(record)
        for action in oracle.game.legal_action_ids(python_state):
            expected = oracle.game.apply_action(python_state, action)
            actual = oracle.native.apply_action(native_state, action)
            assert list(actual.occupancy) == list(expected.occupancy), f"{where} action {action}"
            assert actual.current_player_id == expected.current_player_id, f"{where} {action}"
            assert actual.turn_number == expected.turn_number, f"{where} {action}"
            assert actual.status == (
                1 if expected.status is GameStatus.FINISHED else 0
            ), f"{where} {action}"
            assert list(actual.finish_order) == list(expected.finish_order), f"{where} {action}"


def test_canonical_action_round_trip() -> None:
    for record, oracle, python_state, native_state in _pairs():
        where = _describe(record)
        for action in oracle.game.legal_action_ids(python_state):
            canonical = oracle.native.to_canonical_action(action, native_state)
            assert canonical == oracle.game.encoder.to_canonical_action(
                action, oracle.players, python_state.current_player_id
            ), f"{where} {action}"
            assert oracle.native.to_physical_action(canonical, native_state) == action, where


def test_canonical_actions_apply_identically() -> None:
    for record, oracle, python_state, native_state in _pairs():
        if oracle.game.is_terminal(python_state):
            continue
        where = _describe(record)
        for canonical in oracle.search.legal_action_ids(python_state):
            expected = oracle.search.apply_action(python_state, canonical)
            actual = oracle.native.apply_canonical_action(native_state, canonical)
            assert list(actual.occupancy) == list(expected.occupancy), f"{where} {canonical}"
            assert actual.current_player_id == expected.current_player_id, f"{where} {canonical}"


def test_node_features_are_bit_identical() -> None:
    for record, oracle, python_state, native_state in _pairs():
        where = _describe(record)
        rows, canonical_ids = oracle.native.encode(native_state)
        encoded = oracle.game.encoder.encode(python_state, oracle.players)
        assert list(canonical_ids) == list(encoded.canonical_player_ids), where
        assert len(rows) == len(encoded.node_features), where
        # Features are exactly 0.0 or 1.0, so exact equality is the right test.
        for canonical_id, (actual, expected) in enumerate(zip(rows, encoded.node_features)):
            assert list(actual) == list(expected), f"{where} row {canonical_id}"


def test_vacancy_prior_matches_within_tolerance() -> None:
    for record, oracle, python_state, native_state in _pairs():
        if oracle.game.is_terminal(python_state):
            continue
        where = _describe(record)
        actual = list(oracle.native.vacancy_prior(native_state))
        expected = oracle.priors(python_state)
        assert len(actual) == len(expected), where
        for index, (got, want) in enumerate(zip(actual, expected)):
            assert abs(got - want) <= PRIOR_TOLERANCE, f"{where} action {index}: {got} != {want}"
