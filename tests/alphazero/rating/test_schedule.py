from __future__ import annotations

import itertools
from dataclasses import replace

import pytest

from diamond.alphazero.rating.openings import OpeningSuite
from diamond.alphazero.rating.schedule import (
    MinRatedMatch,
    SooRatedMatch,
    schedule_min_triple,
    schedule_soo_pair,
    validate_min_rated_batch,
    validate_soo_rated_batch,
)


SOO_IDS = ("checkpoint:Soo:alpha", "checkpoint:Soo:beta")
MIN_IDS = (
    "checkpoint:Min:alpha",
    "checkpoint:Min:beta",
    "checkpoint:Min:gamma",
)


def _suite(player_count: int) -> OpeningSuite:
    return OpeningSuite.generate(
        player_count=player_count,
        seed=17,
        opening_count=2,
        max_depth=2,
    )


def test_soo_schedule_has_four_balanced_assignments_per_opening() -> None:
    suite = _suite(2)

    matches = schedule_soo_pair(opening_suite=suite, participant_ids=SOO_IDS)

    assert len(matches) == 4 * len(suite.openings)
    for opening in suite.openings:
        opening_matches = [match for match in matches if match.opening_id == opening.opening_id]
        assert {(match.seat_assignment, match.turn_order) for match in opening_matches} == {
            (seat_assignment, turn_order)
            for seat_assignment in itertools.permutations((1, 2))
            for turn_order in itertools.permutations((1, 2))
        }
        assert all(match.participant_ids == SOO_IDS for match in opening_matches)


def test_min_schedule_has_all_36_seat_and_turn_order_assignments_per_opening() -> None:
    suite = _suite(3)

    matches = schedule_min_triple(opening_suite=suite, participant_ids=MIN_IDS)

    assert len(matches) == 36 * len(suite.openings)
    for opening in suite.openings:
        opening_matches = [match for match in matches if match.opening_id == opening.opening_id]
        assert {(match.seat_assignment, match.turn_order) for match in opening_matches} == {
            (seat_assignment, turn_order)
            for seat_assignment in itertools.permutations((1, 2, 3))
            for turn_order in itertools.permutations((1, 2, 3))
        }
        assert all(match.participant_ids == MIN_IDS for match in opening_matches)


def test_rated_schedules_have_a_deterministic_order_and_unique_match_ids() -> None:
    soo_suite = _suite(2)
    min_suite = _suite(3)

    first_soo = schedule_soo_pair(opening_suite=soo_suite, participant_ids=SOO_IDS)
    second_soo = schedule_soo_pair(opening_suite=soo_suite, participant_ids=SOO_IDS)
    first_min = schedule_min_triple(opening_suite=min_suite, participant_ids=MIN_IDS)
    second_min = schedule_min_triple(opening_suite=min_suite, participant_ids=MIN_IDS)

    assert first_soo == second_soo
    assert first_min == second_min
    assert len({match.match_id for match in first_soo}) == len(first_soo)
    assert len({match.match_id for match in first_min}) == len(first_min)


def test_schedule_rejects_incompatible_suites_and_duplicate_min_artifacts() -> None:
    with pytest.raises(ValueError, match="player_count"):
        schedule_soo_pair(opening_suite=_suite(3), participant_ids=SOO_IDS)
    with pytest.raises(ValueError, match="three distinct"):
        schedule_min_triple(
            opening_suite=_suite(3),
            participant_ids=(MIN_IDS[0], MIN_IDS[1], MIN_IDS[1]),
        )


def test_validators_reject_partial_duplicate_and_incompatible_rated_batches() -> None:
    soo_suite = _suite(2)
    min_suite = _suite(3)
    soo_matches = schedule_soo_pair(opening_suite=soo_suite, participant_ids=SOO_IDS)
    min_matches = schedule_min_triple(opening_suite=min_suite, participant_ids=MIN_IDS)

    with pytest.raises(ValueError, match="complete"):
        validate_soo_rated_batch(soo_matches[:-1], opening_suite=soo_suite)
    with pytest.raises(ValueError, match="duplicate"):
        validate_min_rated_batch(
            (min_matches[0], min_matches[0], *min_matches[2:]),
            opening_suite=min_suite,
        )
    with pytest.raises(ValueError, match="participant_ids"):
        validate_soo_rated_batch(
            (replace(soo_matches[0], participant_ids=(SOO_IDS[1], SOO_IDS[0])), *soo_matches[1:]),
            opening_suite=soo_suite,
        )


def test_match_dtos_expose_task_two_event_dimensions_directly() -> None:
    soo_match = schedule_soo_pair(opening_suite=_suite(2), participant_ids=SOO_IDS)[0]
    min_match = schedule_min_triple(opening_suite=_suite(3), participant_ids=MIN_IDS)[0]

    assert isinstance(soo_match, SooRatedMatch)
    assert isinstance(min_match, MinRatedMatch)
    assert soo_match.participant_ids == SOO_IDS
    assert soo_match.seat_assignment == (1, 2)
    assert soo_match.turn_order == (1, 2)
    assert min_match.participant_ids == MIN_IDS
    assert min_match.seat_assignment == (1, 2, 3)
    assert min_match.turn_order == (1, 2, 3)
