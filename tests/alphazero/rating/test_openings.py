from __future__ import annotations

from dataclasses import replace

import pytest

from diamond.alphazero.game_adapter import AlphaZeroGameAdapter
from diamond.alphazero.rating.openings import OpeningSuite
from diamond.contract.state import build_players


def _suite(player_count: int = 2) -> OpeningSuite:
    return OpeningSuite.generate(
        player_count=player_count,
        seed=31,
        opening_count=4,
        max_depth=3,
        version="benchmark-openings-v1",
    )


def test_opening_suite_includes_the_standard_initial_state() -> None:
    suite = _suite()

    opening = suite.openings[0]

    assert opening.action_ids == ()
    assert suite.reconstruct(opening) == AlphaZeroGameAdapter(build_players(2)).initial_state()


def test_opening_suite_generation_is_deterministic_for_a_fixed_seed() -> None:
    first = _suite()
    second = _suite()

    assert first == second
    assert first.suite_hash == second.suite_hash
    assert first.suite_hash.startswith("sha256:")


def test_opening_suite_reconstructs_every_action_through_the_authoritative_adapter() -> None:
    suite = _suite()
    adapter = AlphaZeroGameAdapter(build_players(2))

    for opening in suite.openings:
        state = adapter.initial_state()
        for action_id in opening.action_ids:
            assert action_id in adapter.legal_action_ids(state)
            state = adapter.apply_action(state, action_id)
        assert suite.reconstruct(opening) == state


def test_opening_suite_rejects_an_opening_for_a_different_player_count() -> None:
    two_player = _suite(2)
    three_player = _suite(3)

    with pytest.raises(ValueError, match="player_count"):
        three_player.reconstruct(two_player.openings[0])


def test_opening_suite_rejects_a_ruleset_fingerprint_mismatch() -> None:
    suite = _suite()
    incompatible = replace(suite, ruleset_fingerprint="sha256:other-ruleset")

    with pytest.raises(ValueError, match="ruleset_fingerprint"):
        incompatible.reconstruct(incompatible.openings[0])


def test_opening_suite_hash_binds_the_semantic_opening_identity() -> None:
    two_player = _suite(2)
    three_player = _suite(3)

    assert two_player.suite_hash != three_player.suite_hash
    assert len({opening.opening_id for opening in two_player.openings}) == len(two_player.openings)
