"""The adapter's rules are the C++ core's, and must equal the Python engine's.

``AlphaZeroGameAdapter`` used to apply moves through ``legal_moves``,
``find_legal_move`` and ``GameSession.commit``. It now asks the native ``Game``.
That is the last behavioural dependent of ``diamond.game``, so the swap is the
one place where a silent divergence would rewrite the game itself: a wrong
successor does not crash, it trains.

This is a bridge gate, not a parity gate over an engine's internals. It runs the
whole fixture corpus -- every position, every legal action of each -- through
both implementations and requires the same answers:

* the same legal action ids, as a set and in the codec's encoding;
* the same successor: occupancy, seat to move, turn number, status, podium.

It retires with the bridge, when ``diamond.game`` is deleted (Phase B).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from diamond.alphazero.game_adapter import AlphaZeroGameAdapter
from diamond.game.rules import find_legal_move, legal_moves
from diamond.game.session import GameSession
from diamond.game.state import GameState, GameStatus, build_players

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "positions.jsonl"


def _positions() -> list[GameState]:
    rows = []
    for line in FIXTURES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(
            (
                row["player_count"],
                row["tag"],
                GameState(
                    occupancy=tuple(row["occupancy"]),
                    current_player_id=row["current_player_id"],
                    turn_number=row["turn_number"],
                    status=(
                        GameStatus.FINISHED
                        if row["status"] == "finished"
                        else GameStatus.IN_PROGRESS
                    ),
                    finish_order=tuple(row["finish_order"]),
                ),
            )
        )
    return rows


def _python_legal_action_ids(adapter: AlphaZeroGameAdapter, state: GameState) -> tuple[int, ...]:
    return tuple(
        adapter.codec.encode(move.source, move.destination)
        for move in legal_moves(adapter.board, state)
    )


def _python_successor(
    adapter: AlphaZeroGameAdapter, state: GameState, action_id: int
) -> GameState:
    source, destination = adapter.codec.decode(action_id)
    move = find_legal_move(
        adapter.board, state, source, destination, player_id=state.current_player_id
    )
    assert move is not None, f"the Python engine calls {action_id} illegal"
    session = GameSession(adapter.players, board=adapter.board, initial=state)
    session.commit(move)
    return session.state


def _fields(state: GameState) -> tuple:
    return (
        state.occupancy,
        state.current_player_id,
        state.turn_number,
        state.status,
        state.finish_order,
    )


@pytest.fixture(scope="module")
def corpus() -> list:
    rows = _positions()
    assert len(rows) > 1000, f"the corpus shrank to {len(rows)} positions"
    return rows


def test_the_two_engines_offer_the_same_legal_actions(corpus: list) -> None:
    for player_count, tag, state in corpus:
        adapter = AlphaZeroGameAdapter(build_players(player_count))
        native = adapter.legal_action_ids(state)
        python = _python_legal_action_ids(adapter, state)
        assert sorted(native) == sorted(python), f"legal actions differ at {tag}"


def test_every_legal_action_reaches_the_same_successor(corpus: list) -> None:
    compared = 0
    for player_count, tag, state in corpus:
        if state.status is GameStatus.FINISHED:
            # The Python session refuses to commit into a finished game; there
            # is no successor for either engine to disagree about.
            continue
        adapter = AlphaZeroGameAdapter(build_players(player_count))
        for action_id in adapter.legal_action_ids(state):
            assert _fields(adapter.apply_action(state, action_id)) == _fields(
                _python_successor(adapter, state, action_id)
            ), f"successor differs at {tag} for action {action_id}"
            compared += 1
    assert compared > 10_000, f"only {compared} successors compared"
