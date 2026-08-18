"""Immutable semantic events for rated benchmark matches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


class RatingEventError(ValueError):
    """Raised when a rating event does not describe a valid rated match."""


def _canonical_event_id(event_type: str, payload: dict[str, object]) -> str:
    semantic_payload = {"event_type": event_type, **payload}
    encoded = json.dumps(
        semantic_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _require_non_empty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RatingEventError(f"{name} must be a non-empty string")
    return value


def _validate_sequence_index(sequence_index: object) -> int:
    if (
        not isinstance(sequence_index, int)
        or isinstance(sequence_index, bool)
        or sequence_index < 0
    ):
        raise RatingEventError("sequence_index must be a non-negative integer")
    return sequence_index


def _validate_participant_ids(participant_ids: object, count: int) -> tuple[str, ...]:
    if not isinstance(participant_ids, tuple) or len(participant_ids) != count:
        raise RatingEventError(f"participant_ids must contain exactly {count} participant IDs")
    if any(not isinstance(participant_id, str) or not participant_id for participant_id in participant_ids):
        raise RatingEventError("participant_ids must contain non-empty strings")
    if len(set(participant_ids)) != count:
        word = "two" if count == 2 else "three"
        raise RatingEventError(f"participant_ids must contain exactly {word} distinct participant IDs")
    return participant_ids


def _validate_completed(completed: object) -> bool:
    if not isinstance(completed, bool):
        raise RatingEventError("completed must be a boolean")
    return completed


@dataclass(frozen=True, slots=True)
class SooRatingEvent:
    """A completed or aborted two-player Soo benchmark match."""

    sequence_index: int
    protocol_id: str
    participant_ids: tuple[str, str]
    opening_id: str
    completed: bool
    winner_id: str | None = None
    loser_id: str | None = None
    event_id: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_sequence_index(self.sequence_index)
        _require_non_empty_string("protocol_id", self.protocol_id)
        participants = _validate_participant_ids(self.participant_ids, 2)
        _require_non_empty_string("opening_id", self.opening_id)
        completed = _validate_completed(self.completed)

        if completed:
            if (
                not isinstance(self.winner_id, str)
                or not isinstance(self.loser_id, str)
                or {self.winner_id, self.loser_id} != set(participants)
            ):
                raise RatingEventError(
                    "completed Soo events require winner_id and loser_id to be a participant permutation"
                )
        elif self.winner_id is not None or self.loser_id is not None:
            raise RatingEventError("aborted Soo events must not contain an outcome")

        object.__setattr__(
            self,
            "event_id",
            _canonical_event_id(
                "soo",
                {
                    "sequence_index": self.sequence_index,
                    "protocol_id": self.protocol_id,
                    "participant_ids": self.participant_ids,
                    "opening_id": self.opening_id,
                    "completed": self.completed,
                    "winner_id": self.winner_id,
                    "loser_id": self.loser_id,
                },
            ),
        )


@dataclass(frozen=True, slots=True)
class MinRatingEvent:
    """A completed or aborted three-player Min benchmark match."""

    sequence_index: int
    protocol_id: str
    participant_ids: tuple[str, str, str]
    opening_id: str
    completed: bool
    final_ranking: tuple[str, str, str] | None = None
    event_id: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_sequence_index(self.sequence_index)
        _require_non_empty_string("protocol_id", self.protocol_id)
        participants = _validate_participant_ids(self.participant_ids, 3)
        _require_non_empty_string("opening_id", self.opening_id)
        completed = _validate_completed(self.completed)

        if completed:
            if (
                not isinstance(self.final_ranking, tuple)
                or len(self.final_ranking) != 3
                or any(not isinstance(participant_id, str) for participant_id in self.final_ranking)
                or set(self.final_ranking) != set(participants)
            ):
                raise RatingEventError(
                    "completed Min events require final_ranking to be a full participant permutation"
                )
        elif self.final_ranking is not None:
            raise RatingEventError("aborted Min events must not contain an outcome")

        object.__setattr__(
            self,
            "event_id",
            _canonical_event_id(
                "min",
                {
                    "sequence_index": self.sequence_index,
                    "protocol_id": self.protocol_id,
                    "participant_ids": self.participant_ids,
                    "opening_id": self.opening_id,
                    "completed": self.completed,
                    "final_ranking": self.final_ranking,
                },
            ),
        )


__all__ = ["MinRatingEvent", "RatingEventError", "SooRatingEvent"]
