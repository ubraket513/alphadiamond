"""Pure Elo updates for completed or aborted Soo benchmark matches."""

from __future__ import annotations

from .protocol import EloConfig


def expected_score(rating_a: float, rating_b: float, config: EloConfig) -> float:
    """Return A's expected win probability against B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / config.logistic_scale))


def rate_soo_match(
    winner_rating: float,
    loser_rating: float,
    completed: bool,
    config: EloConfig,
) -> tuple[float, float]:
    """Return winner and loser ratings after one Soo match.

    An aborted match leaves both values unchanged.  Completed-match updates use
    only the pair of ratings supplied at match start.
    """
    if not completed:
        return winner_rating, loser_rating

    winner_expected = expected_score(winner_rating, loser_rating, config)
    loser_expected = expected_score(loser_rating, winner_rating, config)
    return (
        winner_rating + config.k_factor * (1.0 - winner_expected),
        loser_rating - config.k_factor * loser_expected,
    )


__all__ = ["expected_score", "rate_soo_match"]
