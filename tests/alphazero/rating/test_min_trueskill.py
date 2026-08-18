from __future__ import annotations

import math

import pytest
import trueskill

from diamond.alphazero.rating.events import MinRatingEvent, RatingEventError
from diamond.alphazero.rating.min_trueskill import (
    initial_min_ratings,
    rate_min_event,
    create_environment,
)
from diamond.alphazero.rating.protocol import TrueSkillConfig


PARTICIPANTS = ("checkpoint:Min:alpha", "checkpoint:Min:beta", "checkpoint:Min:gamma")


def _event(*, completed: bool = True) -> MinRatingEvent:
    return MinRatingEvent(
        sequence_index=0,
        protocol_id="sha256:min-benchmark",
        participant_ids=PARTICIPANTS,
        seat_assignment=(1, 2, 3),
        turn_order=(1, 2, 3),
        opening_id="opening-0",
        completed=completed,
        final_ranking=PARTICIPANTS if completed else None,
    )


def test_explicit_environment_uses_min_parameters_without_mutating_global_environment() -> None:
    global_environment = trueskill.global_env()
    global_parameters = (
        global_environment.mu,
        global_environment.sigma,
        global_environment.beta,
        global_environment.tau,
        global_environment.draw_probability,
        global_environment.backend,
    )

    environment = create_environment(TrueSkillConfig())

    assert (
        environment.mu,
        environment.sigma,
        environment.beta,
        environment.tau,
        environment.draw_probability,
        environment.backend,
    ) == (25.0, 25.0 / 3.0, 25.0 / 6.0, 0.0, 0.0, None)
    assert (
        global_environment.mu,
        global_environment.sigma,
        global_environment.beta,
        global_environment.tau,
        global_environment.draw_probability,
        global_environment.backend,
    ) == global_parameters


def test_completed_free_for_all_rates_each_participant_from_its_full_ranking() -> None:
    ratings = initial_min_ratings(PARTICIPANTS, TrueSkillConfig())

    updated = rate_min_event(ratings, _event(), TrueSkillConfig())

    assert updated[PARTICIPANTS[0]].mu > ratings[PARTICIPANTS[0]].mu
    assert updated[PARTICIPANTS[2]].mu < ratings[PARTICIPANTS[2]].mu
    assert all(math.isfinite(rating.sigma) for rating in updated.values())
    assert all(rating.sigma < 25.0 / 3.0 for rating in updated.values())
    assert updated[PARTICIPANTS[0]].exposure > updated[PARTICIPANTS[1]].exposure
    assert updated[PARTICIPANTS[1]].exposure > updated[PARTICIPANTS[2]].exposure
    assert all(rating.rated_games == 1 for rating in updated.values())


def test_aborted_min_event_does_not_change_ratings() -> None:
    ratings = initial_min_ratings(PARTICIPANTS, TrueSkillConfig())

    assert rate_min_event(ratings, _event(completed=False), TrueSkillConfig()) == ratings


def test_candidate_champion_champion_cannot_form_a_rated_min_event() -> None:
    with pytest.raises(RatingEventError, match="three distinct"):
        MinRatingEvent(
            sequence_index=0,
            protocol_id="sha256:min-benchmark",
            participant_ids=(PARTICIPANTS[0], PARTICIPANTS[1], PARTICIPANTS[1]),
            seat_assignment=(1, 2, 3),
            turn_order=(1, 2, 3),
            opening_id="opening-0",
            completed=True,
            final_ranking=(PARTICIPANTS[0], PARTICIPANTS[1], PARTICIPANTS[1]),
        )
