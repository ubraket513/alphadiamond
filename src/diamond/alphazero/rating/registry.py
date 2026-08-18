"""Replayable Soo Elo registry with atomic JSON persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..config import NetworkConfig
from ..identity import CheckpointCompatibilitySpec, MIN_MODEL_NAME, SOO_MODEL_NAME
from .elo import rate_soo_match
from .events import SooRatingEvent
from .participants import CheckpointParticipant
from .protocol import BenchmarkProtocol, EloConfig

_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class SooLeaderboardEntry:
    """One Soo participant's current replayed rating and rated-game count."""

    participant_id: str
    display_name: str
    rating: float
    games: int


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _protocol_payload(protocol: BenchmarkProtocol) -> dict[str, object]:
    return {
        "compatibility": protocol.compatibility.to_metadata(),
        "simulations": protocol.simulations,
        "c_puct": protocol.c_puct,
        "dirichlet_epsilon": protocol.dirichlet_epsilon,
        "decision_temperature": protocol.decision_temperature,
        "max_game_moves": protocol.max_game_moves,
        "opening_suite_version": protocol.opening_suite_version,
        "opening_suite_hash": protocol.opening_suite_hash,
        "rating_system_version": protocol.rating_system_version,
        "rating_parameters": protocol.rating_parameters,
        "inference_numeric_mode": protocol.inference_numeric_mode,
    }


def _protocol_from_payload(payload: object) -> BenchmarkProtocol:
    if not isinstance(payload, Mapping):
        raise ValueError("registry protocol must be a mapping")
    try:
        compatibility_payload = payload["compatibility"]
        if not isinstance(compatibility_payload, Mapping):
            raise ValueError("registry protocol compatibility must be a mapping")
        network_payload = compatibility_payload["network_config"]
        if not isinstance(network_payload, Mapping):
            raise ValueError("registry protocol network_config must be a mapping")
        compatibility_factory = {
            SOO_MODEL_NAME: CheckpointCompatibilitySpec.soo,
            MIN_MODEL_NAME: CheckpointCompatibilitySpec.min,
        }[compatibility_payload["model_name"]]
        compatibility = compatibility_factory(
            model_version=compatibility_payload["model_version"],
            network_config=NetworkConfig(**dict(network_payload)),
        )
        compatibility.assert_compatible(compatibility_payload)
        return BenchmarkProtocol(
            compatibility=compatibility,
            simulations=payload["simulations"],
            c_puct=payload["c_puct"],
            dirichlet_epsilon=payload["dirichlet_epsilon"],
            decision_temperature=payload["decision_temperature"],
            max_game_moves=payload["max_game_moves"],
            opening_suite_version=payload["opening_suite_version"],
            opening_suite_hash=payload["opening_suite_hash"],
            rating_system_version=payload["rating_system_version"],
            rating_parameters=dict(payload["rating_parameters"]),
            inference_numeric_mode=payload.get("inference_numeric_mode", "fp32"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid registry protocol: {exc}") from exc


def _participant_payload(participant: CheckpointParticipant) -> dict[str, object]:
    return {
        "participant_id": participant.participant_id,
        "model_name": participant.model_name,
        "model_version": participant.model_version,
        "training_step": participant.training_step,
        "checkpoint_sha256": participant.checkpoint_sha256,
        "compatibility_metadata": participant.compatibility_metadata,
        "display_name": participant.display_name,
    }


def _participant_from_payload(payload: object) -> CheckpointParticipant:
    if not isinstance(payload, Mapping):
        raise ValueError("registry participant must be a mapping")
    try:
        return CheckpointParticipant(
            participant_id=payload["participant_id"],
            model_name=payload["model_name"],
            model_version=payload["model_version"],
            training_step=payload["training_step"],
            checkpoint_sha256=payload["checkpoint_sha256"],
            compatibility_metadata=payload["compatibility_metadata"],
            display_name=payload["display_name"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid registry participant: {exc}") from exc


def _event_payload(event: SooRatingEvent) -> dict[str, object]:
    return {
        "event_type": "soo",
        "sequence_index": event.sequence_index,
        "protocol_id": event.protocol_id,
        "participant_ids": event.participant_ids,
        "seat_assignment": event.seat_assignment,
        "turn_order": event.turn_order,
        "opening_id": event.opening_id,
        "completed": event.completed,
        "winner_id": event.winner_id,
        "loser_id": event.loser_id,
    }


def _event_from_payload(payload: object) -> SooRatingEvent:
    if not isinstance(payload, Mapping):
        raise ValueError("registry event must be a mapping")
    if payload.get("event_type") != "soo":
        raise ValueError("Task 3 registry accepts only Soo events")
    try:
        return SooRatingEvent(
            sequence_index=payload["sequence_index"],
            protocol_id=payload["protocol_id"],
            participant_ids=tuple(payload["participant_ids"]),
            seat_assignment=tuple(payload["seat_assignment"]),
            turn_order=tuple(payload["turn_order"]),
            opening_id=payload["opening_id"],
            completed=payload["completed"],
            winner_id=payload.get("winner_id"),
            loser_id=payload.get("loser_id"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid registry event: {exc}") from exc


class RatingRegistry:
    """An append-only, protocol-scoped Soo Elo event registry."""

    def __init__(self, protocol: BenchmarkProtocol) -> None:
        if not isinstance(protocol, BenchmarkProtocol):
            raise ValueError("protocol must be a BenchmarkProtocol")
        if protocol.compatibility.identity.model_name != SOO_MODEL_NAME:
            raise ValueError("Task 3 registry requires a Soo compatibility protocol")
        if protocol.rating_system_version != EloConfig().rating_system_version:
            raise ValueError("Task 3 registry requires the soo-elo-v1 rating system")
        try:
            self._elo_config = EloConfig(**dict(protocol.rating_parameters))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid Soo Elo rating parameters: {exc}") from exc
        if self._elo_config.rating_system_version != protocol.rating_system_version:
            raise ValueError("Elo rating parameters do not match the registry protocol")

        self.protocol = protocol
        self._participants: dict[str, CheckpointParticipant] = {}
        self._events: list[SooRatingEvent] = []
        self._ratings: dict[str, float] = {}

    @property
    def participants(self) -> Mapping[str, CheckpointParticipant]:
        return MappingProxyType(self._participants)

    @property
    def events(self) -> tuple[SooRatingEvent, ...]:
        return tuple(self._events)

    @property
    def ratings(self) -> Mapping[str, float]:
        return MappingProxyType(self._ratings)

    def add_participant(self, participant: CheckpointParticipant) -> float:
        """Register one compatible checkpoint and return its current Elo."""
        if not isinstance(participant, CheckpointParticipant):
            raise ValueError("participant must be a CheckpointParticipant")
        try:
            self.protocol.compatibility.assert_compatible(participant.compatibility_metadata)
        except ValueError as exc:
            raise ValueError(f"participant is incompatible with registry protocol: {exc}") from exc

        existing = self._participants.get(participant.participant_id)
        if existing is not None:
            if existing != participant:
                raise ValueError("participant ID is already registered with different metadata")
            return self._ratings[participant.participant_id]

        self._participants[participant.participant_id] = participant
        self._ratings[participant.participant_id] = self._elo_config.initial_rating
        return self._elo_config.initial_rating

    def _validate_event(self, event: SooRatingEvent) -> None:
        if not isinstance(event, SooRatingEvent):
            raise ValueError("Task 3 registry accepts only Soo events")
        if event.protocol_id != self.protocol.protocol_id:
            raise ValueError("event protocol does not match registry protocol")
        missing_participants = set(event.participant_ids) - self._participants.keys()
        if missing_participants:
            raise ValueError("event references unregistered participants")

    def record_event(self, event: SooRatingEvent) -> bool:
        """Append one event, returning ``False`` for an already-recorded event."""
        self._validate_event(event)
        if any(recorded.event_id == event.event_id for recorded in self._events):
            return False
        self._events.append(event)
        self.rebuild()
        return True

    def rebuild(self) -> None:
        """Recompute cached ratings solely from registered participants and events."""
        ratings = {
            participant_id: self._elo_config.initial_rating
            for participant_id in self._participants
        }
        for event in self._events:
            self._validate_event(event)
            if not event.completed:
                continue
            assert event.winner_id is not None
            assert event.loser_id is not None
            winner_rating, loser_rating = rate_soo_match(
                ratings[event.winner_id],
                ratings[event.loser_id],
                True,
                self._elo_config,
            )
            ratings[event.winner_id] = winner_rating
            ratings[event.loser_id] = loser_rating
        self._ratings = ratings

    def soo_leaderboard(self) -> tuple[SooLeaderboardEntry, ...]:
        """Return all Soo participants in deterministic descending-Elo order."""
        games = {participant_id: 0 for participant_id in self._participants}
        for event in self._events:
            if event.completed:
                for participant_id in event.participant_ids:
                    games[participant_id] += 1
        entries = (
            SooLeaderboardEntry(
                participant_id=participant_id,
                display_name=participant.display_name,
                rating=self._ratings[participant_id],
                games=games[participant_id],
            )
            for participant_id, participant in self._participants.items()
        )
        return tuple(sorted(entries, key=lambda entry: (-entry.rating, entry.participant_id)))

    def save(self, path: str | Path) -> None:
        """Atomically persist event sources and a replaceable derived rating cache."""
        destination = Path(path)
        payload = {
            "format_version": _FORMAT_VERSION,
            "protocol": _protocol_payload(self.protocol),
            "participants": [_participant_payload(item) for item in self._participants.values()],
            "events": [_event_payload(event) for event in self._events],
            "ratings": self._ratings,
        }
        temporary = destination.with_name(f"{destination.name}.tmp")
        temporary.write_text(
            json.dumps(_json_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "RatingRegistry":
        """Load authoritative sources and always rebuild the derived Elo cache."""
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid registry file: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("registry file root must be a mapping")
        if payload.get("format_version") != _FORMAT_VERSION:
            raise ValueError("unsupported registry format version")

        registry = cls(_protocol_from_payload(payload.get("protocol")))
        participants = payload.get("participants")
        events = payload.get("events")
        if not isinstance(participants, list) or not isinstance(events, list):
            raise ValueError("registry participants and events must be lists")
        for participant_payload in participants:
            registry.add_participant(_participant_from_payload(participant_payload))
        for event_payload in events:
            registry.record_event(_event_from_payload(event_payload))
        registry.rebuild()
        return registry


__all__ = ["RatingRegistry", "SooLeaderboardEntry"]
