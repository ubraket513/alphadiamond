from __future__ import annotations

import itertools

import pytest

from diamond.alphazero.action_codec import ActionCodec, ActionSpaceSpec
from diamond.alphazero.encoder import CanonicalEncoder
from diamond.contract.board import Camp, standard_board
from diamond.contract.state import GameState, build_players, initial_state
from diamond.game.rules import legal_moves


@pytest.mark.parametrize("camp", list(Camp))
def test_camp_mapping_is_a_complete_invertible_board_permutation(camp: Camp) -> None:
    encoder = CanonicalEncoder(standard_board(), ActionCodec(ActionSpaceSpec.diamond73()))
    mapping = encoder.position_mapping(camp)

    assert set(mapping.physical_to_canonical) == set(range(73))
    assert set(mapping.canonical_to_physical) == set(range(73))
    for physical in range(73):
        canonical = mapping.physical_to_canonical[physical]
        assert mapping.canonical_to_physical[canonical] == physical


@pytest.mark.parametrize("player_count", [2, 3])
def test_canonical_player_order_follows_match_order_not_player_ids(player_count: int) -> None:
    encoder = CanonicalEncoder(standard_board(), ActionCodec(ActionSpaceSpec.diamond73()))

    for order in itertools.permutations(range(1, player_count + 1)):
        players = build_players(player_count, order=order)
        for index, player in enumerate(players):
            expected = tuple(players[(index + offset) % player_count].id for offset in range(player_count))
            assert encoder.canonical_player_ids(players, player.id) == expected


@pytest.mark.parametrize("player_count", [2, 3])
def test_every_legal_action_round_trips_through_canonical_space(player_count: int) -> None:
    board = standard_board()
    codec = ActionCodec(ActionSpaceSpec.diamond73())
    encoder = CanonicalEncoder(board, codec)

    for order in itertools.permutations(range(1, player_count + 1)):
        players = build_players(player_count, order=order)
        state = initial_state(players, board)
        for move in legal_moves(board, state):
            physical = codec.encode(move.source, move.destination)
            canonical = encoder.to_canonical_action(physical, players, state.current_player_id)
            assert encoder.to_physical_action(canonical, players, state.current_player_id) == physical


def test_encoded_features_use_self_next_previous_and_finished_channels() -> None:
    board = standard_board()
    players = build_players(3, order=(3, 1, 2))
    state = initial_state(players, board)
    state = GameState(
        occupancy=state.occupancy,
        current_player_id=1,
        turn_number=8,
        finish_order=(3,),
    )
    encoder = CanonicalEncoder(board, ActionCodec(ActionSpaceSpec.diamond73()))

    encoded = encoder.encode(state, players)

    assert encoded.canonical_player_ids == (1, 2, 3)
    assert encoded.feature_count == 6
    assert len(encoded.node_features) == 73
    assert all(len(row) == 6 for row in encoded.node_features)
    assert all(row[3:] == (0.0, 0.0, 1.0) for row in encoded.node_features)
    assert sum(row[0] for row in encoded.node_features) == 10
    assert sum(row[1] for row in encoded.node_features) == 10
    assert sum(row[2] for row in encoded.node_features) == 10


def test_legal_mask_contains_only_canonicalized_authoritative_actions() -> None:
    board = standard_board()
    players = build_players(2, order=(2, 1))
    state = initial_state(players, board)
    codec = ActionCodec(ActionSpaceSpec.diamond73())
    encoder = CanonicalEncoder(board, codec)
    physical_moves = legal_moves(board, state)

    mask = encoder.legal_action_mask(physical_moves, players, state.current_player_id)
    expected = {
        encoder.to_canonical_action(
            codec.encode(move.source, move.destination), players, state.current_player_id
        )
        for move in physical_moves
    }

    assert len(mask) == 5329
    assert {action_id for action_id, allowed in enumerate(mask) if allowed} == expected
