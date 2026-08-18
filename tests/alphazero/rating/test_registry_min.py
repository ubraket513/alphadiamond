from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.rating.events import MinRatingEvent
from diamond.alphazero.rating.participants import CheckpointParticipant
from diamond.alphazero.rating.protocol import BenchmarkProtocol, TrueSkillConfig
from diamond.alphazero.rating.registry import RatingRegistry


def _protocol(
    *, compatibility: CheckpointCompatibilitySpec | None = None
) -> BenchmarkProtocol:
    return BenchmarkProtocol(
        compatibility=compatibility
        or CheckpointCompatibilitySpec.min(
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
        rating_system_version="min-trueskill-v1",
        rating_parameters=asdict(TrueSkillConfig()),
    )


def _participant(number: int) -> CheckpointParticipant:
    compatibility = CheckpointCompatibilitySpec.min(
        model_version="1.2.3",
        network_config=NetworkConfig(width=16, residual_blocks=1),
    )
    checkpoint_sha256 = f"{number:064x}"
    return CheckpointParticipant(
        participant_id=f"checkpoint:Min:1.2.3:{number}:{checkpoint_sha256}",
        model_name="Min",
        model_version="1.2.3",
        training_step=number,
        checkpoint_sha256=checkpoint_sha256,
        compatibility_metadata=compatibility.to_metadata(),
        display_name=f"Min 1.2.3 @ {number}",
    )


def _event(
    participants: tuple[CheckpointParticipant, CheckpointParticipant, CheckpointParticipant],
    protocol: BenchmarkProtocol,
    sequence_index: int,
    *,
    completed: bool = True,
    final_ranking: tuple[str, str, str] | None = None,
) -> MinRatingEvent:
    participant_ids = tuple(participant.participant_id for participant in participants)
    return MinRatingEvent(
        sequence_index=sequence_index,
        protocol_id=protocol.protocol_id,
        participant_ids=participant_ids,
        seat_assignment=(3, 1, 2),
        turn_order=(2, 3, 1),
        opening_id=f"opening-{sequence_index}",
        completed=completed,
        final_ranking=(final_ranking or participant_ids) if completed else None,
    )


def _registry_with(*participants: CheckpointParticipant) -> tuple[RatingRegistry, BenchmarkProtocol]:
    protocol = _protocol()
    registry = RatingRegistry(protocol)
    for participant in participants:
        registry.add_participant(participant)
    return registry, protocol


def test_min_registry_starts_with_configured_priors_and_reports_insufficient_history() -> None:
    first, second = _participant(1), _participant(2)
    registry, _ = _registry_with(first, second)

    assert registry.min_ratings[first.participant_id].mu == 25.0
    assert registry.min_ratings[first.participant_id].rated_games == 0
    assert registry.has_sufficient_min_history is False


def test_reregistering_a_min_artifact_returns_its_existing_prior() -> None:
    participant = _participant(1)
    registry, _ = _registry_with(participant)

    assert registry.add_participant(participant) == registry.min_ratings[participant.participant_id].exposure


def test_registry_rejects_trueskill_protocol_with_soo_compatibility() -> None:
    protocol = _protocol(
        compatibility=CheckpointCompatibilitySpec.soo(
            model_version="1.2.3",
            network_config=NetworkConfig(width=16, residual_blocks=1),
        )
    )

    with pytest.raises(ValueError, match="Min"):
        RatingRegistry(protocol)


def test_min_registry_replays_completed_events_in_recorded_order() -> None:
    first, second, third = _participant(1), _participant(2), _participant(3)
    participants = (first, second, third)
    registry, protocol = _registry_with(*participants)
    first_event = _event(participants, protocol, 0)
    second_event = _event(
        participants,
        protocol,
        1,
        final_ranking=(third.participant_id, second.participant_id, first.participant_id),
    )

    registry.record_event(first_event)
    registry.record_event(second_event)
    recorded_ratings = dict(registry.min_ratings)
    registry.rebuild()

    assert dict(registry.min_ratings) == recorded_ratings
    assert registry.events == (first_event, second_event)
    assert all(rating.rated_games == 2 for rating in registry.min_ratings.values())
    assert registry.has_sufficient_min_history is True


def test_min_registry_duplicate_and_aborted_events_are_noops() -> None:
    first, second, third = _participant(1), _participant(2), _participant(3)
    participants = (first, second, third)
    registry, protocol = _registry_with(*participants)
    completed = _event(participants, protocol, 0)
    aborted = _event(participants, protocol, 1, completed=False)

    assert registry.record_event(completed) is True
    ratings_after_completed = dict(registry.min_ratings)
    assert registry.record_event(completed) is False
    assert registry.record_event(aborted) is True
    assert dict(registry.min_ratings) == ratings_after_completed


def test_min_save_load_rebuilds_cached_ratings_and_preserves_sources(tmp_path) -> None:
    first, second, third = _participant(1), _participant(2), _participant(3)
    participants = (first, second, third)
    registry, protocol = _registry_with(*participants)
    registry.record_event(_event(participants, protocol, 0))
    destination = tmp_path / "ratings.json"

    registry.save(destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["min_ratings"] = {first.participant_id: {"mu": "not-a-rating"}}
    destination.write_text(json.dumps(payload), encoding="utf-8")
    restored = RatingRegistry.load(destination)

    assert not destination.with_name("ratings.json.tmp").exists()
    assert restored.protocol == registry.protocol
    assert restored.participants == registry.participants
    assert restored.events == registry.events
    assert dict(restored.min_ratings) == dict(registry.min_ratings)


def test_min_leaderboard_sorts_descending_by_exposure() -> None:
    first, second, third = _participant(1), _participant(2), _participant(3)
    participants = (first, second, third)
    registry, protocol = _registry_with(*participants)
    registry.record_event(_event(participants, protocol, 0))

    leaderboard = registry.min_leaderboard()

    assert [entry.participant_id for entry in leaderboard] == [
        first.participant_id,
        second.participant_id,
        third.participant_id,
    ]
    assert [entry.exposure for entry in leaderboard] == sorted(
        (entry.exposure for entry in leaderboard), reverse=True
    )
