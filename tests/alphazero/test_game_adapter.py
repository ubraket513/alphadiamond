from __future__ import annotations

import pytest

from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.game.rules import IllegalMoveError
from diamond.game.state import build_players


@pytest.mark.parametrize("player_count", [2, 3])
def test_adapter_uses_authoritative_legal_moves_and_transitions(player_count: int) -> None:
    adapter = AlphaZeroGameAdapter(build_players(player_count))
    state = adapter.initial_state()
    legal_moves = adapter.legal_moves(state)
    legal_actions = adapter.legal_action_ids(state)

    assert len(legal_actions) == len(legal_moves)
    assert legal_actions == tuple(
        adapter.codec.encode(move.source, move.destination) for move in legal_moves
    )

    selected = legal_actions[0]
    resolved = adapter.resolve_action(state, selected)
    next_state = adapter.apply_action(state, selected)

    assert resolved == legal_moves[0]
    assert next_state.occupant(resolved.source) == 0
    assert next_state.occupant(resolved.destination) == resolved.player_id
    assert next_state.turn_number == state.turn_number + 1


def test_adapter_rejects_representable_but_illegal_action() -> None:
    adapter = AlphaZeroGameAdapter(build_players(2))
    state = adapter.initial_state()

    with pytest.raises(IllegalMoveError):
        adapter.resolve_action(state, adapter.codec.encode(0, 0))


@pytest.mark.parametrize("player_count", [2, 3])
def test_search_adapter_uses_canonical_actions_without_changing_semantics(
    player_count: int,
) -> None:
    game = AlphaZeroGameAdapter(build_players(player_count, order=tuple(range(player_count, 0, -1))))
    search = DiamondSearchAdapter(game)
    state = game.initial_state()
    request = search.evaluation_request(state)

    assert request.legal_action_ids == search.legal_action_ids(state)
    assert request.canonical_player_ids[0] == state.current_player_id
    next_state = search.apply_action(state, request.legal_action_ids[0])
    assert next_state.turn_number == 2
