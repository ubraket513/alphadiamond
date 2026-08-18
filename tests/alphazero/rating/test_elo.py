from __future__ import annotations

import pytest

from diamond.alphazero.rating.elo import expected_score, rate_soo_match
from diamond.alphazero.rating.protocol import EloConfig


def test_expected_score_is_half_for_equal_ratings() -> None:
    assert expected_score(1_000.0, 1_000.0, EloConfig()) == 0.5


def test_expected_score_uses_the_configured_logistic_scale() -> None:
    assert expected_score(1_200.0, 1_000.0, EloConfig()) == pytest.approx(
        0.7597469266479578
    )


def test_completed_match_updates_winner_and_loser_from_pre_match_ratings() -> None:
    winner, loser = rate_soo_match(1_200.0, 1_000.0, True, EloConfig())

    assert winner == pytest.approx(1_207.6880983472654)
    assert loser == pytest.approx(992.3119016527346)


def test_aborted_match_leaves_both_ratings_unchanged() -> None:
    assert rate_soo_match(1_200.0, 1_000.0, False, EloConfig()) == (1_200.0, 1_000.0)
