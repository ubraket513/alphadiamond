"""Gate B: deterministic native MCTS equals the Python oracle exactly.

Scope is deliberately narrow (``docs/native_selfplay_phase0.md`` section 7):
single-threaded, single-game, one deterministic evaluator, ``epsilon = 0`` and
``temperature = 0``.  No batcher, no thread pool, no Python callback, no RNG.

The evaluator request sequence is compared, not just the root statistics.  Two
searches can arrive at the same visit counts through different traversals, so
matching root visits is necessary but nowhere near sufficient; the request
sequence pins which leaves were evaluated, in which order.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from diamond.alphazero.config import MCTSConfig
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.mcts.search_2p import MCTS2P
from diamond.alphazero.native import native_game, require_native
from diamond.game.state import GameState, GameStatus, build_players

from .reference_evaluator import ReferenceEvaluator, request_hash

FIXTURE = Path(__file__).parent / "fixtures" / "positions.jsonl"
Q_TOLERANCE = 1e-6
"""Section 7's stated bound.  In practice the two agree bit for bit."""

SIMULATION_COUNTS = (1, 2, 8, 33, 64)
"""Deliberately includes 1 and 2: the first simulation and the first re-descent
are where an off-by-one in the backup or the parent-visit aggregate shows up."""

POSITION_STRIDE = 47
"""Sample the corpus rather than searching all of it; a full search per position
costs ~200x what a Gate A comparison does."""


def _load() -> list[dict]:
    if not FIXTURE.exists():  # pragma: no cover - regenerate with tools/
        pytest.skip(f"missing corpus: {FIXTURE}; run tools/build_native_corpus.py")
    records = [
        json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line
    ]
    # MCTS2P is two-player only, and a terminal state cannot be searched.
    return [r for r in records if r["player_count"] == 2 and r["status"] != "finished"]


CORPUS = _load()
SAMPLE = CORPUS[::POSITION_STRIDE]


def _python_state(record: dict) -> GameState:
    return GameState(
        occupancy=tuple(record["occupancy"]),
        current_player_id=record["current_player_id"],
        turn_number=record["turn_number"],
        status=GameStatus(record["status"]),
        finish_order=tuple(record["finish_order"]),
    )


def _native_state(record: dict):
    return require_native().State(
        occupancy=record["occupancy"],
        current_player=record["current_player_id"],
        turn_number=record["turn_number"],
        status=0,
        finish_order=record["finish_order"],
    )


class _Harness:
    def __init__(self) -> None:
        self.players = build_players(2)
        self.game = AlphaZeroGameAdapter(self.players)
        self.search = DiamondSearchAdapter(self.game)
        self.native = native_game(self.players)
        self.module = require_native()

    def python_search(self, state: GameState, simulations: int, evaluator_name: str = "hash"):
        evaluator = ReferenceEvaluator(uniform=evaluator_name == "uniform")
        config = MCTSConfig(
            simulations=simulations, c_puct=1.5, dirichlet_epsilon=0.0, seed=0
        )
        result = MCTS2P(self.search, evaluator, config).run(state, temperature=0.0)
        return result, evaluator

    def native_search(self, record: dict, simulations: int, evaluator_name: str = "hash"):
        config = self.module.MCTSConfig(
            simulations=simulations, c_puct=1.5, dirichlet_epsilon=0.0, seed=0
        )
        return self.native.search(
            _native_state(record),
            config,
            temperature=0.0,
            trace=True,
            evaluator=evaluator_name,
        )


_HARNESS: _Harness | None = None


def _harness() -> _Harness:
    """Built on first use, never at import.

    Module-level construction would touch the extension during collection, which
    fails outright on a host without it instead of skipping -- the optional
    backend must stay optional all the way through test collection.
    """
    global _HARNESS
    if _HARNESS is None:
        _HARNESS = _Harness()
    return _HARNESS


def _cases():
    for record in SAMPLE:
        for simulations in SIMULATION_COUNTS:
            yield record, simulations


def test_sample_is_representative() -> None:
    assert len(SAMPLE) >= 15
    assert any(r["tag"].startswith("packing") for r in SAMPLE), "no endgame position sampled"
    assert any(r["tag"].startswith("walk") or r["tag"] == "opening" for r in SAMPLE)


def test_reference_evaluator_is_bit_identical_across_backends() -> None:
    """Gate B compares two searches; this pins the one thing they share.

    If the Python and native evaluators disagreed by a single ulp, every search
    comparison below would fail for a reason that has nothing to do with search.
    """
    for record in CORPUS:
        state = _python_state(record)
        request = _harness().search.evaluation_request(state)
        expected = ReferenceEvaluator().evaluate((request,))[0]

        legal, priors, value, hash_value = _harness().native.reference_evaluate(
            _native_state(record)
        )
        where = record["tag"]
        assert hash_value == request_hash(request), where
        assert list(legal) == list(request.legal_action_ids), where
        assert value == expected.value, where  # bit-exact, not approximate
        assert list(priors) == [expected.priors[a] for a in request.legal_action_ids], where


def test_root_statistics_match() -> None:
    for record, simulations in _cases():
        where = f"{record['tag']} sims={simulations}"
        expected, _ = _harness().python_search(_python_state(record), simulations)
        actual = _harness().native_search(record, simulations)

        # Root action order is the expansion order, and it is observable: it is
        # the order Dirichlet noise components will be assigned in.
        assert list(actual["root_actions"]) == list(expected.visit_counts), where
        assert list(actual["visit_counts"]) == [
            expected.visit_counts[a] for a in expected.visit_counts
        ], where
        assert actual["selected_action"] == expected.selected_action, where

        for action, q_value in zip(actual["root_actions"], actual["q_values"]):
            assert abs(q_value - expected.q_values[action]) <= Q_TOLERANCE, f"{where} {action}"
        for action, policy in zip(actual["root_actions"], actual["policy"]):
            assert abs(policy - expected.policy[action]) <= Q_TOLERANCE, f"{where} {action}"


def test_q_values_are_bit_identical() -> None:
    """Stronger than section 7's 1e-6, and worth asserting while it holds.

    The PUCT key is a double comparison, so a single-ulp drift in q is not a
    rounding curiosity -- it can flip a selection and change the whole descent.
    """
    for record, simulations in _cases():
        where = f"{record['tag']} sims={simulations}"
        expected, _ = _harness().python_search(_python_state(record), simulations)
        actual = _harness().native_search(record, simulations)
        for action, q_value in zip(actual["root_actions"], actual["q_values"]):
            assert q_value == expected.q_values[action], f"{where} action {action}"


def test_evaluator_request_sequence_matches() -> None:
    for record, simulations in _cases():
        where = f"{record['tag']} sims={simulations}"
        _, evaluator = _harness().python_search(_python_state(record), simulations)
        actual = _harness().native_search(record, simulations)

        assert actual["evaluator_calls"] == evaluator.calls, where
        native_trace = [(h, tuple(actions)) for h, actions in actual["trace"]]
        assert len(native_trace) == len(evaluator.trace), where
        # Ordered, element-wise: same leaves, evaluated in the same order.
        assert native_trace == evaluator.trace, where


def test_expanded_legal_action_sequences_match() -> None:
    """Per-node legal action ordering, at every expansion, not just the root."""
    for record, simulations in _cases():
        where = f"{record['tag']} sims={simulations}"
        _, evaluator = _harness().python_search(_python_state(record), simulations)
        actual = _harness().native_search(record, simulations)
        for index, (native, expected) in enumerate(zip(actual["trace"], evaluator.trace)):
            assert list(native[1]) == list(expected[1]), f"{where} expansion {index}"


def test_simulation_count_is_exact() -> None:
    for record, simulations in _cases():
        actual = _harness().native_search(record, simulations)
        assert actual["simulations_run"] == simulations, f"{record['tag']} sims={simulations}"


def test_deep_trees_match() -> None:
    """The shallow-tree regime is not enough on its own.

    At 64 simulations against ~54 root actions most simulations expand a fresh
    root child (section 0.3), so the descent rarely goes past depth 2 and a
    broken parent-visit aggregate or a mis-signed backup can stay invisible.
    These cases force a deep tree: the fewest-branching positions in the sample,
    searched hard.
    """
    narrow = sorted(
        SAMPLE,
        key=lambda record: len(_harness().search.legal_action_ids(_python_state(record))),
    )[:3]
    for record in narrow:
        where = f"{record['tag']} deep"
        expected, evaluator = _harness().python_search(_python_state(record), 400)
        actual = _harness().native_search(record, 400)

        assert list(actual["root_actions"]) == list(expected.visit_counts), where
        assert list(actual["visit_counts"]) == list(expected.visit_counts.values()), where
        assert actual["selected_action"] == expected.selected_action, where
        for action, q_value in zip(actual["root_actions"], actual["q_values"]):
            assert q_value == expected.q_values[action], f"{where} action {action}"
        native_trace = [(h, tuple(actions)) for h, actions in actual["trace"]]
        assert native_trace == evaluator.trace, where
        # The tree must actually be deep, or this test proves nothing.
        assert max(actual["visit_counts"]) > 2 * len(actual["root_actions"]) // 3, where


def test_tied_priors_exercise_the_puct_tie_break() -> None:
    """The case distinct priors can never reach.

    With a uniform prior every unvisited edge of a node has an exactly equal
    PUCT key, so selection is decided purely by ``min`` over the action id.  The
    production vacancy prior does this constantly -- every action with the same
    integer progress score gets the same probability -- so an untested tie-break
    is a live risk, not a theoretical one.
    """
    for record in SAMPLE:
        for simulations in (1, 8, 64):
            where = f"{record['tag']} sims={simulations} uniform"
            expected, evaluator = _harness().python_search(
                _python_state(record), simulations, "uniform"
            )
            actual = _harness().native_search(record, simulations, "uniform")

            assert list(actual["root_actions"]) == list(expected.visit_counts), where
            assert list(actual["visit_counts"]) == list(expected.visit_counts.values()), where
            assert actual["selected_action"] == expected.selected_action, where
            for action, q_value in zip(actual["root_actions"], actual["q_values"]):
                assert q_value == expected.q_values[action], f"{where} action {action}"
            native_trace = [(h, tuple(actions)) for h, actions in actual["trace"]]
            assert native_trace == evaluator.trace, where


def _terminal_leaf_simulations(simulations: int, evaluator_calls: int) -> int:
    """How many simulations ended on a terminal node rather than an expansion.

    Every simulation either expands exactly one leaf or bottoms out on a
    terminal node, and the root costs one expansion before the loop, so this
    falls straight out of the evaluator call count -- no extra instrumentation,
    and it is derived identically on both sides.
    """
    return simulations - (evaluator_calls - 1)


def test_terminal_leaves_are_reached_and_valued_identically() -> None:
    """The terminal-value path, covered on purpose rather than by luck.

    ``terminal_scalar_value`` is the one place the 2P search reads a result
    instead of an evaluator, and it depends on the search-adapter's terminal
    perspective rule (``finish_order[1]``, not ``state.current_player_id``).
    Most corpus positions never reach a terminal leaf in 64 simulations, so
    without this test the path rides entirely on test_deep_trees_match.
    """
    reached = 0
    for record in CORPUS:
        actual = _harness().native_search(record, 64)
        terminal_leaves = _terminal_leaf_simulations(64, actual["evaluator_calls"])
        if not terminal_leaves:
            continue
        reached += 1
        where = f"{record['tag']} terminal-leaves={terminal_leaves}"
        expected, evaluator = _harness().python_search(_python_state(record), 64)
        assert _terminal_leaf_simulations(64, evaluator.calls) == terminal_leaves, where
        assert list(actual["root_actions"]) == list(expected.visit_counts), where
        assert list(actual["visit_counts"]) == list(expected.visit_counts.values()), where
        assert actual["selected_action"] == expected.selected_action, where
        for action, q_value in zip(actual["root_actions"], actual["q_values"]):
            assert q_value == expected.q_values[action], f"{where} action {action}"
        native_trace = [(h, tuple(actions)) for h, actions in actual["trace"]]
        assert native_trace == evaluator.trace, where

    # Guard the guard: if the corpus ever stops reaching terminal leaves this
    # test would silently become a no-op.
    assert reached >= 10, f"only {reached} corpus positions reached a terminal leaf"


def test_stochastic_search_is_refused_rather_than_approximated() -> None:
    """The RNG policy of section 9 is not implemented, so it must not be faked."""
    record = SAMPLE[0]
    module = _harness().module
    with pytest.raises(Exception, match="deterministic-only"):
        _harness().native.search(
            _native_state(record),
            module.MCTSConfig(simulations=8, dirichlet_epsilon=0.25),
            temperature=0.0,
        )
    with pytest.raises(Exception, match="deterministic-only"):
        _harness().native.search(
            _native_state(record), module.MCTSConfig(simulations=8), temperature=1.0
        )
