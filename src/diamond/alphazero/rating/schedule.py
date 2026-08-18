"""Complete deterministic benchmark schedules for rated Soo and Min matches."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field

from ..identity import RULESET_FINGERPRINT, RULESET_VERSION
from .openings import OpeningSuite


def _match_id(match_type: str, payload: dict[str, object]) -> str:
    encoded = json.dumps(
        {"match_type": match_type, **payload},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _validate_participant_ids(value: object, count: int) -> tuple[str, ...]:
    if not isinstance(value, tuple) or len(value) != count:
        raise ValueError(f"participant_ids must contain exactly {count} artifact IDs")
    if any(not isinstance(participant_id, str) or not participant_id for participant_id in value):
        raise ValueError("participant_ids must contain non-empty artifact IDs")
    if len(set(value)) != count:
        word = "two" if count == 2 else "three"
        raise ValueError(f"participant_ids must contain {word} distinct artifact IDs")
    return value


def _validate_permutation(name: str, value: object, count: int) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or len(value) != count
        or any(not isinstance(seat, int) or isinstance(seat, bool) for seat in value)
        or set(value) != set(range(1, count + 1))
    ):
        raise ValueError(f"{name} must be a 1-based physical-seat permutation")
    return value


def _validate_opening_id(opening_id: object) -> str:
    if not isinstance(opening_id, str) or not opening_id:
        raise ValueError("opening_id must be a non-empty string")
    return opening_id


@dataclass(frozen=True, slots=True)
class SooRatedMatch:
    """A 2-player schedule item directly compatible with a Soo rating event."""

    participant_ids: tuple[str, str]
    seat_assignment: tuple[int, int]
    turn_order: tuple[int, int]
    opening_id: str
    match_id: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_participant_ids(self.participant_ids, 2)
        _validate_permutation("seat_assignment", self.seat_assignment, 2)
        _validate_permutation("turn_order", self.turn_order, 2)
        _validate_opening_id(self.opening_id)
        object.__setattr__(
            self,
            "match_id",
            _match_id(
                "soo",
                {
                    "participant_ids": self.participant_ids,
                    "seat_assignment": self.seat_assignment,
                    "turn_order": self.turn_order,
                    "opening_id": self.opening_id,
                },
            ),
        )


@dataclass(frozen=True, slots=True)
class MinRatedMatch:
    """A 3-player schedule item directly compatible with a Min rating event."""

    participant_ids: tuple[str, str, str]
    seat_assignment: tuple[int, int, int]
    turn_order: tuple[int, int, int]
    opening_id: str
    match_id: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_participant_ids(self.participant_ids, 3)
        _validate_permutation("seat_assignment", self.seat_assignment, 3)
        _validate_permutation("turn_order", self.turn_order, 3)
        _validate_opening_id(self.opening_id)
        object.__setattr__(
            self,
            "match_id",
            _match_id(
                "min",
                {
                    "participant_ids": self.participant_ids,
                    "seat_assignment": self.seat_assignment,
                    "turn_order": self.turn_order,
                    "opening_id": self.opening_id,
                },
            ),
        )


def _validate_suite(opening_suite: object, player_count: int) -> OpeningSuite:
    if not isinstance(opening_suite, OpeningSuite):
        raise ValueError("opening_suite must be an OpeningSuite")
    if opening_suite.player_count != player_count:
        raise ValueError(f"opening_suite player_count must be {player_count}")
    if opening_suite.ruleset_version != RULESET_VERSION:
        raise ValueError("opening_suite ruleset_version is incompatible")
    if opening_suite.ruleset_fingerprint != RULESET_FINGERPRINT:
        raise ValueError("opening_suite ruleset_fingerprint is incompatible")
    if any(
        opening.player_count != player_count
        or opening.suite_version != opening_suite.version
        or opening.ruleset_version != opening_suite.ruleset_version
        or opening.ruleset_fingerprint != opening_suite.ruleset_fingerprint
        for opening in opening_suite.openings
    ):
        raise ValueError("opening_suite contains incompatible openings")
    return opening_suite


def schedule_soo_pair(
    *, opening_suite: OpeningSuite, participant_ids: tuple[str, str]
) -> tuple[SooRatedMatch, ...]:
    """Cross both artifact-to-seat assignments and both turn orders per opening."""
    suite = _validate_suite(opening_suite, 2)
    _validate_participant_ids(participant_ids, 2)
    seats = (1, 2)
    return tuple(
        SooRatedMatch(
            participant_ids=participant_ids,
            seat_assignment=seat_assignment,
            turn_order=turn_order,
            opening_id=opening.opening_id,
        )
        for opening in suite.openings
        for seat_assignment in itertools.permutations(seats)
        for turn_order in itertools.permutations(seats)
    )


def schedule_min_triple(
    *, opening_suite: OpeningSuite, participant_ids: tuple[str, str, str]
) -> tuple[MinRatedMatch, ...]:
    """Cross all six artifact-to-seat and six turn-order assignments per opening."""
    suite = _validate_suite(opening_suite, 3)
    _validate_participant_ids(participant_ids, 3)
    seats = (1, 2, 3)
    return tuple(
        MinRatedMatch(
            participant_ids=participant_ids,
            seat_assignment=seat_assignment,
            turn_order=turn_order,
            opening_id=opening.opening_id,
        )
        for opening in suite.openings
        for seat_assignment in itertools.permutations(seats)
        for turn_order in itertools.permutations(seats)
    )


def validate_soo_rated_batch(
    matches: tuple[SooRatedMatch, ...], *, opening_suite: OpeningSuite
) -> None:
    """Reject anything other than a complete, unique Soo balance schedule."""
    suite = _validate_suite(opening_suite, 2)
    if not isinstance(matches, tuple) or not matches:
        raise ValueError("rated Soo batch must be a non-empty tuple")
    if any(not isinstance(match, SooRatedMatch) for match in matches):
        raise ValueError("rated Soo batch must contain SooRatedMatch values")
    participant_ids = matches[0].participant_ids
    if any(match.participant_ids != participant_ids for match in matches):
        raise ValueError("rated Soo batch participant_ids must be consistent")
    expected = schedule_soo_pair(opening_suite=suite, participant_ids=participant_ids)
    if len(matches) != len(expected):
        raise ValueError("rated Soo batch must contain a complete balance cycle")
    if len({match.match_id for match in matches}) != len(matches):
        raise ValueError("rated Soo batch contains duplicate matches")
    if matches != expected:
        raise ValueError("rated Soo batch is incompatible with its complete balance schedule")


def validate_min_rated_batch(
    matches: tuple[MinRatedMatch, ...], *, opening_suite: OpeningSuite
) -> None:
    """Reject anything other than a complete, unique Min balance schedule."""
    suite = _validate_suite(opening_suite, 3)
    if not isinstance(matches, tuple) or not matches:
        raise ValueError("rated Min batch must be a non-empty tuple")
    if any(not isinstance(match, MinRatedMatch) for match in matches):
        raise ValueError("rated Min batch must contain MinRatedMatch values")
    participant_ids = matches[0].participant_ids
    if any(match.participant_ids != participant_ids for match in matches):
        raise ValueError("rated Min batch participant_ids must be consistent")
    expected = schedule_min_triple(opening_suite=suite, participant_ids=participant_ids)
    if len(matches) != len(expected):
        raise ValueError("rated Min batch must contain a complete balance cycle")
    if len({match.match_id for match in matches}) != len(matches):
        raise ValueError("rated Min batch contains duplicate matches")
    if matches != expected:
        raise ValueError("rated Min batch is incompatible with its complete balance schedule")


__all__ = [
    "MinRatedMatch",
    "SooRatedMatch",
    "schedule_min_triple",
    "schedule_soo_pair",
    "validate_min_rated_batch",
    "validate_soo_rated_batch",
]
