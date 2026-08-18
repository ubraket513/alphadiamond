from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.rating.events import SooRatingEvent
from diamond.alphazero.rating.participants import CheckpointParticipant
from diamond.alphazero.rating.protocol import BenchmarkProtocol, EloConfig
from diamond.alphazero.rating.registry import RatingRegistry


def _protocol(
    *, compatibility: CheckpointCompatibilitySpec | None = None
) -> BenchmarkProtocol:
    return BenchmarkProtocol(
        compatibility=compatibility
        or CheckpointCompatibilitySpec.soo(
            model_version="1.2.3",
            network_config=NetworkConfig(width=16, residual_blocks=1),
        ),
        simulations=200,
        c_puct=1.5,
        dirichlet_epsilon=0.0,
        decision_temperature=0.0,
        max_game_moves=2_000,
        opening_suite_version="benchmark-openings-v1",
        opening_suite_hash="sha256:openings-v1",
        rating_system_version="soo-elo-v1",
        rating_parameters=asdict(EloConfig()),
    )


def _participant(
    number: int, *, network_config: NetworkConfig | None = None
) -> CheckpointParticipant:
    compatibility = CheckpointCompatibilitySpec.soo(
        model_version="1.2.3",
        network_config=network_config or NetworkConfig(width=16, residual_blocks=1),
    )
    checkpoint_sha256 = f"{number:064x}"
    return CheckpointParticipant(
        participant_id=(
            f"checkpoint:Soo:1.2.3:{number}:{checkpoint_sha256}"
        ),
        model_name="Soo",
        model_version="1.2.3",
        training_step=number,
        checkpoint_sha256=checkpoint_sha256,
        compatibility_metadata=compatibility.to_metadata(),
        display_name=f"Soo 1.2.3 @ {number}",
    )


def _event(
    winner: CheckpointParticipant,
    loser: CheckpointParticipant,
    protocol: BenchmarkProtocol,
    sequence_index: int,
    *,
    completed: bool = True,
) -> SooRatingEvent:
    return SooRatingEvent(
        sequence_index=sequence_index,
        protocol_id=protocol.protocol_id,
        participant_ids=(winner.participant_id, loser.participant_id),
        seat_assignment=(2, 1),
        turn_order=(1, 2),
        opening_id=f"opening-{sequence_index}",
        completed=completed,
        winner_id=winner.participant_id if completed else None,
        loser_id=loser.participant_id if completed else None,
    )


def _registry_with(*participants: CheckpointParticipant) -> tuple[RatingRegistry, BenchmarkProtocol]:
    protocol = _protocol()
    registry = RatingRegistry(protocol)
    for participant in participants:
        registry.add_participant(participant)
    return registry, protocol


def test_new_participant_starts_at_default_elo() -> None:
    participant = _participant(1)
    registry, _ = _registry_with(participant)

    assert registry.ratings[participant.participant_id] == 1_000.0


def test_registry_rejects_elo_protocol_with_min_compatibility() -> None:
    protocol = _protocol(
        compatibility=CheckpointCompatibilitySpec.min(
            model_version="1.2.3",
            network_config=NetworkConfig(width=16, residual_blocks=1),
        )
    )

    with pytest.raises(ValueError, match="Soo"):
        RatingRegistry(protocol)


def test_duplicate_event_is_idempotent() -> None:
    winner, loser = _participant(1), _participant(2)
    registry, protocol = _registry_with(winner, loser)
    event = _event(winner, loser, protocol, 0)

    assert registry.record_event(event) is True
    ratings_after_first_record = dict(registry.ratings)
    assert registry.record_event(event) is False
    assert dict(registry.ratings) == ratings_after_first_record
    assert registry.events == (event,)


def test_event_with_different_protocol_is_rejected() -> None:
    winner, loser = _participant(1), _participant(2)
    registry, protocol = _registry_with(winner, loser)
    event = SooRatingEvent(
        sequence_index=0,
        protocol_id="sha256:other-protocol",
        participant_ids=(winner.participant_id, loser.participant_id),
        seat_assignment=(1, 2),
        turn_order=(1, 2),
        opening_id="opening-0",
        completed=True,
        winner_id=winner.participant_id,
        loser_id=loser.participant_id,
    )

    with pytest.raises(ValueError, match="protocol"):
        registry.record_event(event)
    assert registry.events == ()
    assert protocol.protocol_id != event.protocol_id


def test_participant_incompatible_with_registry_protocol_is_rejected() -> None:
    registry, _ = _registry_with()
    incompatible = _participant(1, network_config=NetworkConfig(width=32, residual_blocks=1))

    with pytest.raises(ValueError, match="compatib"):
        registry.add_participant(incompatible)
    assert registry.participants == {}


def test_rebuild_replays_events_in_the_recorded_order() -> None:
    first, second = _participant(1), _participant(2)
    registry, protocol = _registry_with(first, second)
    first_event = _event(first, second, protocol, 0)
    second_event = _event(second, first, protocol, 1)

    registry.record_event(first_event)
    registry.record_event(second_event)
    recorded_ratings = dict(registry.ratings)
    registry.rebuild()

    assert dict(registry.ratings) == recorded_ratings
    assert registry.events == (first_event, second_event)


def test_save_load_is_atomic_and_preserves_the_replayable_sources(tmp_path) -> None:
    winner, loser = _participant(1), _participant(2)
    registry, protocol = _registry_with(winner, loser)
    registry.record_event(_event(winner, loser, protocol, 0))
    destination = tmp_path / "ratings.json"

    registry.save(destination)
    restored = RatingRegistry.load(destination)

    assert not destination.with_name("ratings.json.tmp").exists()
    assert restored.protocol == registry.protocol
    assert restored.participants == registry.participants
    assert restored.events == registry.events
    assert dict(restored.ratings) == dict(registry.ratings)


def test_load_rebuilds_ratings_when_cached_ratings_are_corrupted(tmp_path) -> None:
    winner, loser = _participant(1), _participant(2)
    registry, protocol = _registry_with(winner, loser)
    registry.record_event(_event(winner, loser, protocol, 0))
    destination = tmp_path / "ratings.json"
    registry.save(destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["ratings"] = {winner.participant_id: "not-a-rating"}
    destination.write_text(json.dumps(payload), encoding="utf-8")

    restored = RatingRegistry.load(destination)

    assert dict(restored.ratings) == dict(registry.ratings)


def test_soo_leaderboard_sorts_descending_by_rating() -> None:
    first, second, third = _participant(1), _participant(2), _participant(3)
    registry, protocol = _registry_with(first, second, third)
    registry.record_event(_event(first, second, protocol, 0))

    leaderboard = registry.soo_leaderboard()

    assert [entry.participant_id for entry in leaderboard] == [
        first.participant_id,
        third.participant_id,
        second.participant_id,
    ]
    assert [entry.rating for entry in leaderboard] == sorted(
        (entry.rating for entry in leaderboard), reverse=True
    )
