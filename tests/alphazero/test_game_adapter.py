from __future__ import annotations

import pytest

from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.native.topology import camp_positions, neighbour_table
from diamond.contract.camps import PLAYABLE_HOLES
from diamond.contract.move import IllegalMoveError
from diamond.contract.state import EMPTY, GameState, build_players, initial_state


@pytest.mark.parametrize("player_count", [2, 3])
def test_adapter_uses_authoritative_legal_moves_and_transitions(player_count: int) -> None:
    adapter = AlphaZeroGameAdapter(build_players(player_count))
    state = adapter.initial_state()
    legal_actions = adapter.legal_action_ids(state)

    assert legal_actions
    # Action ids are the codec's, whichever engine produced them.
    for action in legal_actions:
        source, destination = adapter.codec.decode(action)
        assert state.occupant(source) == state.current_player_id
        assert state.occupant(destination) == EMPTY

    selected = legal_actions[0]
    source, destination = adapter.codec.decode(selected)
    next_state = adapter.apply_action(state, selected)

    assert next_state.occupant(source) == EMPTY
    assert next_state.occupant(destination) == state.current_player_id
    assert next_state.turn_number == state.turn_number + 1
    assert next_state.current_player_id != state.current_player_id


def test_adapter_rejects_representable_but_illegal_action() -> None:
    adapter = AlphaZeroGameAdapter(build_players(2))
    state = adapter.initial_state()

    with pytest.raises(IllegalMoveError):
        adapter.apply_action(state, adapter.codec.encode(0, 0))


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


def test_adapter_can_start_from_an_authoritative_setup_state() -> None:
    players = build_players(2, order=(2, 1))
    setup = initial_state(players)
    game = AlphaZeroGameAdapter(players, initial=setup)

    assert game.initial_state() is setup


def test_soo_terminal_search_perspective_advances_to_the_loser() -> None:
    players = build_players(2)
    winner = players[0]
    target = camp_positions(winner.target_camp)
    destination = target[-1]
    entry = next(
        neighbour
        for neighbour in neighbour_table()[destination]
        if neighbour >= 0 and neighbour not in target
    )
    occupancy = [EMPTY] * PLAYABLE_HOLES
    for position in target[:-1]:
        occupancy[position] = winner.id
    occupancy[entry] = winner.id
    setup = GameState(tuple(occupancy), winner.id, 40)
    game = AlphaZeroGameAdapter(players, initial=setup)
    search = DiamondSearchAdapter(game)
    physical = game.codec.encode(entry, destination)
    canonical = game.encoder.to_canonical_action(physical, players, winner.id)

    terminal = search.apply_action(setup, canonical)

    assert search.is_terminal(terminal)
    assert search.current_player_id(terminal) == players[1].id
    assert search.terminal_scalar_value(terminal, players[1].id) == -1.0
