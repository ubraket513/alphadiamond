"""Official TrueSkill support for three-player Min benchmark events."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import trueskill

from .events import MinRatingEvent
from .protocol import TrueSkillConfig


@dataclass(frozen=True, slots=True)
class MinRating:
    """A persisted TrueSkill value for one immutable Min checkpoint."""

    mu: float
    sigma: float
    exposure: float
    rated_games: int

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.mu, self.sigma, self.exposure)):
            raise ValueError("Min rating values must be finite")
        if self.sigma <= 0.0:
            raise ValueError("Min rating sigma must be positive")
        if not isinstance(self.rated_games, int) or isinstance(self.rated_games, bool):
            raise ValueError("rated_games must be an integer")
        if self.rated_games < 0:
            raise ValueError("rated_games must be non-negative")


def create_environment(config: TrueSkillConfig) -> trueskill.TrueSkill:
    """Create an isolated TrueSkill environment for immutable Min checkpoints."""
    return trueskill.TrueSkill(
        mu=config.mu,
        sigma=config.sigma,
        beta=config.beta,
        tau=config.tau,
        draw_probability=config.draw_probability,
        backend=config.backend,
    )


def initial_min_ratings(
    participant_ids: Sequence[str], config: TrueSkillConfig
) -> dict[str, MinRating]:
    """Create configured priors for distinct Min checkpoint participants."""
    if len(set(participant_ids)) != len(participant_ids):
        raise ValueError("Min ratings require distinct participant IDs")
    environment = create_environment(config)
    rating = environment.create_rating()
    return {
        participant_id: MinRating(
            mu=rating.mu,
            sigma=rating.sigma,
            exposure=environment.expose(rating),
            rated_games=0,
        )
        for participant_id in participant_ids
    }


def rate_min_event(
    ratings: Mapping[str, MinRating], event: MinRatingEvent, config: TrueSkillConfig
) -> dict[str, MinRating]:
    """Apply one completed three-way Min ranking with native TrueSkill ranks."""
    if not isinstance(event, MinRatingEvent):
        raise ValueError("Min rating updates require a MinRatingEvent")
    if not event.completed:
        return dict(ratings)
    if len(set(event.participant_ids)) != 3:
        raise ValueError("Min rating events require three distinct participant IDs")
    missing = set(event.participant_ids) - ratings.keys()
    if missing:
        raise ValueError("Min rating event references unknown participants")

    environment = create_environment(config)
    groups = [
        (
            environment.create_rating(
                mu=ratings[participant_id].mu,
                sigma=ratings[participant_id].sigma,
            ),
        )
        for participant_id in event.participant_ids
    ]
    assert event.final_ranking is not None
    ranks_by_participant = {
        participant_id: rank for rank, participant_id in enumerate(event.final_ranking)
    }
    updated_groups = environment.rate(
        groups,
        ranks=tuple(ranks_by_participant[participant_id] for participant_id in event.participant_ids),
    )
    updated = dict(ratings)
    for participant_id, group in zip(event.participant_ids, updated_groups, strict=True):
        rating = group[0]
        updated[participant_id] = MinRating(
            mu=rating.mu,
            sigma=rating.sigma,
            exposure=environment.expose(rating),
            rated_games=ratings[participant_id].rated_games + 1,
        )
    return updated


__all__ = ["MinRating", "create_environment", "initial_min_ratings", "rate_min_event"]
