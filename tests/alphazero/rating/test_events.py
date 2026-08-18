from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from diamond.alphazero.rating.events import (
    MinRatingEvent,
    RatingEventError,
    SooRatingEvent,
)


SOO_PARTICIPANTS = ("checkpoint:Soo:alpha", "checkpoint:Soo:beta")
MIN_PARTICIPANTS = (
    "checkpoint:Min:alpha",
    "checkpoint:Min:beta",
    "checkpoint:Min:gamma",
)
PROTOCOL_ID = "sha256:benchmark-protocol"


def _soo_event(**changes: object) -> SooRatingEvent:
    values: dict[str, object] = {
        "sequence_index": 7,
        "protocol_id": PROTOCOL_ID,
        "participant_ids": SOO_PARTICIPANTS,
        "opening_id": "opening-03",
        "completed": True,
        "winner_id": SOO_PARTICIPANTS[0],
        "loser_id": SOO_PARTICIPANTS[1],
    }
    values.update(changes)
    return SooRatingEvent(**values)


def _min_event(**changes: object) -> MinRatingEvent:
    values: dict[str, object] = {
        "sequence_index": 8,
        "protocol_id": PROTOCOL_ID,
        "participant_ids": MIN_PARTICIPANTS,
        "opening_id": "opening-04",
        "completed": True,
        "final_ranking": MIN_PARTICIPANTS,
    }
    values.update(changes)
    return MinRatingEvent(**values)


def test_rating_event_id_is_deterministic_and_binds_semantic_payload() -> None:
    first = _soo_event()
    duplicate = _soo_event()
    next_in_sequence = _soo_event(sequence_index=8)

    assert first.event_id == duplicate.event_id
    assert first.event_id.startswith("sha256:")
    assert first.event_id != next_in_sequence.event_id


def test_completed_soo_event_requires_winner_and_loser_to_be_a_participant_permutation() -> None:
    with pytest.raises(RatingEventError, match="winner_id and loser_id"):
        _soo_event(winner_id=SOO_PARTICIPANTS[0], loser_id=SOO_PARTICIPANTS[0])

    with pytest.raises(RatingEventError, match="winner_id and loser_id"):
        _soo_event(winner_id=SOO_PARTICIPANTS[0], loser_id="checkpoint:Soo:outsider")


def test_completed_min_event_requires_a_strict_full_ranking_permutation() -> None:
    with pytest.raises(RatingEventError, match="final_ranking"):
        _min_event(final_ranking=MIN_PARTICIPANTS[:2])

    with pytest.raises(RatingEventError, match="final_ranking"):
        _min_event(
            final_ranking=(MIN_PARTICIPANTS[0], MIN_PARTICIPANTS[1], MIN_PARTICIPANTS[1])
        )


def test_min_event_requires_three_distinct_participants() -> None:
    with pytest.raises(RatingEventError, match="three distinct"):
        _min_event(
            participant_ids=(MIN_PARTICIPANTS[0], MIN_PARTICIPANTS[1], MIN_PARTICIPANTS[1])
        )


@pytest.mark.parametrize("factory", [_soo_event, _min_event])
def test_aborted_event_has_no_outcome(factory) -> None:
    if factory is _soo_event:
        event = factory(completed=False, winner_id=None, loser_id=None)
        with pytest.raises(RatingEventError, match="aborted"):
            factory(completed=False, winner_id=SOO_PARTICIPANTS[0], loser_id=SOO_PARTICIPANTS[1])
    else:
        event = factory(completed=False, final_ranking=None)
        with pytest.raises(RatingEventError, match="aborted"):
            factory(completed=False, final_ranking=MIN_PARTICIPANTS)

    assert event.completed is False


def test_event_keeps_protocol_identity_and_is_immutable() -> None:
    event = _min_event()

    assert event.protocol_id == PROTOCOL_ID
    with pytest.raises(FrozenInstanceError):
        event.sequence_index = 9
    assert replace(event).event_id == event.event_id
