"""Independent checkpoint-strength benchmark execution and persistence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeAlias

from ..config import MCTSConfig
from ..game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from ..identity import MIN_MODEL_NAME, SOO_MODEL_NAME, CheckpointCompatibilitySpec
from ..inference.coordinator import InferenceConfig, InferenceCoordinator
from ..inference.model_pool import InferenceModelPool
from ..inference.remote import RemoteEvaluator
from ..rating.events import MinRatingEvent, SooRatingEvent
from ..rating.openings import BenchmarkOpening, OpeningSuite
from ..rating.participants import CheckpointParticipant
from ..rating.protocol import BenchmarkProtocol
from ..rating.registry import RatingRegistry
from ..rating.schedule import (
    MinRatedMatch,
    SooRatedMatch,
    schedule_min_triple,
    schedule_soo_pair,
    validate_min_rated_batch,
    validate_soo_rated_batch,
)
from ..search_factory import SearchFactory, three_player_search, two_player_search
from ...game.state import build_players
from .coordinator import CandidateArtifact

RatingEvent: TypeAlias = SooRatingEvent | MinRatingEvent
RatedMatch: TypeAlias = SooRatedMatch | MinRatedMatch
MatchRunner: TypeAlias = Callable[
    [RatedMatch, BenchmarkOpening, Mapping[str, RemoteEvaluator]],
    tuple[str, ...] | None,
]


def _event_payload(event: RatingEvent) -> dict[str, object]:
    common: dict[str, object] = {
        "sequence_index": event.sequence_index,
        "protocol_id": event.protocol_id,
        "participant_ids": event.participant_ids,
        "seat_assignment": event.seat_assignment,
        "turn_order": event.turn_order,
        "opening_id": event.opening_id,
        "completed": event.completed,
    }
    if isinstance(event, SooRatingEvent):
        return {
            "event_type": "soo",
            **common,
            "winner_id": event.winner_id,
            "loser_id": event.loser_id,
        }
    return {"event_type": "min", **common, "final_ranking": event.final_ranking}


def _event_from_payload(value: object) -> RatingEvent:
    if not isinstance(value, Mapping):
        raise ValueError("benchmark event must be a JSON object")
    try:
        common = {
            "sequence_index": value["sequence_index"],
            "protocol_id": value["protocol_id"],
            "participant_ids": tuple(value["participant_ids"]),
            "seat_assignment": tuple(value["seat_assignment"]),
            "turn_order": tuple(value["turn_order"]),
            "opening_id": value["opening_id"],
            "completed": value["completed"],
        }
        if value.get("event_type") == "soo":
            return SooRatingEvent(
                **common,
                winner_id=value.get("winner_id"),
                loser_id=value.get("loser_id"),
            )
        if value.get("event_type") == "min":
            ranking = value.get("final_ranking")
            return MinRatingEvent(
                **common,
                final_ranking=tuple(ranking) if ranking is not None else None,
            )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid benchmark event: {error}") from error
    raise ValueError("benchmark event_type must be soo or min")


class ProductionBenchmarkStage:
    """Run complete versioned schedules and rate only completed real matches."""

    def __init__(
        self,
        *,
        protocol: BenchmarkProtocol,
        opening_suite: OpeningSuite,
        checkpoint_paths: Callable[[], tuple[Path, ...]],
        artifacts_root: Path,
        registry: RatingRegistry,
        registry_path: Path,
        model_pool: InferenceModelPool,
        inference_config: InferenceConfig,
        match_runner: MatchRunner | None = None,
        search_factory: SearchFactory | None = None,
    ) -> None:
        if protocol.compatibility.identity.player_count != opening_suite.player_count:
            raise ValueError("benchmark opening suite player_count is incompatible")
        if (
            protocol.opening_suite_version != opening_suite.version
            or protocol.opening_suite_hash != opening_suite.suite_hash
        ):
            raise ValueError("benchmark protocol opening suite identity changed")
        if registry.protocol != protocol:
            raise ValueError("benchmark registry protocol changed")
        self.protocol = protocol
        self.opening_suite = opening_suite
        self.checkpoint_paths = checkpoint_paths
        self.artifacts_root = Path(artifacts_root)
        self.registry = registry
        self.registry_path = Path(registry_path)
        self.model_pool = model_pool
        self.inference_config = inference_config
        self.match_runner = match_runner or self._run_match
        # Resolved on the first match rather than here: constructing the stage
        # with an injected ``match_runner`` must not require the extension.
        self.search_factory = search_factory
        self.status = "eligible"

    @property
    def compatibility(self) -> CheckpointCompatibilitySpec:
        return self.protocol.compatibility

    @property
    def protocol_id(self) -> str:
        return self.protocol.protocol_id

    def _artifact_path(self, operation_id: str) -> Path:
        digest = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()
        return self.artifacts_root / f"{digest}.json"

    def load(self, operation_id: str) -> tuple[RatingEvent, ...] | None:
        path = self._artifact_path(operation_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid benchmark artifact: {error}") from error
        if not isinstance(payload, Mapping) or payload.get("operation_id") != operation_id:
            raise ValueError("invalid benchmark artifact identity")
        status = payload.get("status")
        events = payload.get("events")
        if not isinstance(status, str) or not isinstance(events, list):
            raise ValueError("invalid benchmark artifact content")
        loaded = tuple(_event_from_payload(event) for event in events)
        if any(event.protocol_id != self.protocol_id for event in loaded):
            raise ValueError("benchmark artifact protocol changed")
        self.status = status
        return loaded

    def execute(
        self,
        operation_id: str,
        candidate: CandidateArtifact,
        champion_checkpoint: str | None,
    ) -> tuple[RatingEvent, ...]:
        existing = self.load(operation_id)
        if existing is not None:
            return existing
        selected = self._select_checkpoints(candidate, champion_checkpoint)
        if selected is None:
            self.status = "insufficient_history"
            self._write_artifact(operation_id, ())
            return ()

        participants = tuple(CheckpointParticipant.from_checkpoint(path) for path in selected)
        for participant in participants:
            self.compatibility.assert_compatible(participant.compatibility_metadata)
            self.registry.add_participant(participant)
        participant_ids = tuple(participant.participant_id for participant in participants)
        matches: tuple[RatedMatch, ...]
        if self.compatibility.identity.model_name == SOO_MODEL_NAME:
            matches = schedule_soo_pair(
                opening_suite=self.opening_suite,
                participant_ids=participant_ids,  # type: ignore[arg-type]
            )
            validate_soo_rated_batch(matches, opening_suite=self.opening_suite)  # type: ignore[arg-type]
        else:
            matches = schedule_min_triple(
                opening_suite=self.opening_suite,
                participant_ids=participant_ids,  # type: ignore[arg-type]
            )
            validate_min_rated_batch(matches, opening_suite=self.opening_suite)  # type: ignore[arg-type]

        keys = {}
        for participant, path in zip(participants, selected, strict=True):
            key = self.model_pool.activate_checkpoint(path, expected=self.compatibility)
            if key.checkpoint_sha256 != participant.checkpoint_sha256:
                raise ValueError("benchmark checkpoint identity changed during activation")
            keys[participant.participant_id] = key

        coordinator = InferenceCoordinator(self.model_pool, self.inference_config)
        evaluators = {
            participant_id: RemoteEvaluator(
                coordinator,
                model_key=key,
                client_id=f"benchmark-{key.checkpoint_sha256[:12]}",
            )
            for participant_id, key in keys.items()
        }
        openings = {opening.opening_id: opening for opening in self.opening_suite.openings}
        events: list[RatingEvent] = []
        coordinator.start()
        try:
            for sequence_index, match in enumerate(matches):
                ranking = self.match_runner(match, openings[match.opening_id], evaluators)
                if ranking is None:
                    continue
                if isinstance(match, SooRatedMatch):
                    if len(ranking) != 2 or set(ranking) != set(match.participant_ids):
                        raise ValueError("completed Soo benchmark returned an invalid ranking")
                    events.append(
                        SooRatingEvent(
                            sequence_index=sequence_index,
                            protocol_id=self.protocol_id,
                            participant_ids=match.participant_ids,
                            seat_assignment=match.seat_assignment,
                            turn_order=match.turn_order,
                            opening_id=match.opening_id,
                            completed=True,
                            winner_id=ranking[0],
                            loser_id=ranking[1],
                        )
                    )
                else:
                    if len(ranking) != 3 or set(ranking) != set(match.participant_ids):
                        raise ValueError("completed Min benchmark returned an invalid ranking")
                    events.append(
                        MinRatingEvent(
                            sequence_index=sequence_index,
                            protocol_id=self.protocol_id,
                            participant_ids=match.participant_ids,
                            seat_assignment=match.seat_assignment,
                            turn_order=match.turn_order,
                            opening_id=match.opening_id,
                            completed=True,
                            final_ranking=ranking,  # type: ignore[arg-type]
                        )
                    )
        finally:
            coordinator.stop()

        result = tuple(events)
        for event in result:
            self.registry.record_event(event)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry.save(self.registry_path)
        self.status = "eligible" if result else "no_completed_outcome"
        self._write_artifact(operation_id, result)
        return result

    def _select_checkpoints(
        self,
        candidate: CandidateArtifact,
        champion_checkpoint: str | None,
    ) -> tuple[Path, ...] | None:
        paths = [candidate.path]
        if champion_checkpoint is not None:
            paths.append(Path(champion_checkpoint))
        paths.extend(Path(path) for path in self.checkpoint_paths())
        by_hash: dict[str, tuple[CheckpointParticipant, Path]] = {}
        for path in paths:
            participant = CheckpointParticipant.from_checkpoint(path)
            self.compatibility.assert_compatible(participant.compatibility_metadata)
            by_hash.setdefault(participant.checkpoint_sha256, (participant, path))
        candidate_row = by_hash.get(candidate.checkpoint_sha256)
        if candidate_row is None:
            raise ValueError("candidate checkpoint is absent from benchmark history")
        remaining = sorted(
            (row for digest, row in by_hash.items() if digest != candidate.checkpoint_sha256),
            key=lambda row: (-row[0].training_step, row[0].checkpoint_sha256),
        )
        required_others = 1 if self.compatibility.identity.model_name == SOO_MODEL_NAME else 2
        if len(remaining) < required_others:
            return None
        return (candidate_row[1], *(row[1] for row in remaining[:required_others]))

    def _run_match(
        self,
        match: RatedMatch,
        opening: BenchmarkOpening,
        evaluators: Mapping[str, RemoteEvaluator],
    ) -> tuple[str, ...] | None:
        player_count = self.compatibility.identity.player_count
        players = build_players(player_count, order=match.turn_order)
        authoritative = AlphaZeroGameAdapter(players)
        state = authoritative.initial_state()
        for action_id in opening.action_ids:
            if action_id not in authoritative.legal_action_ids(state):
                raise ValueError("benchmark opening action is illegal for scheduled turn order")
            state = authoritative.apply_action(state, action_id)
        game = DiamondSearchAdapter(AlphaZeroGameAdapter(players, initial=state))
        participant_by_seat = {
            seat: participant_id
            for participant_id, seat in zip(
                match.participant_ids, match.seat_assignment, strict=True
            )
        }
        seed_base = int(match.match_id.removeprefix("sha256:")[:15], 16)
        search = self.search_factory or (
            two_player_search() if player_count == 2 else three_player_search()
        )
        moves = 0
        while not game.is_terminal(state) and moves < self.protocol.max_game_moves:
            participant_id = participant_by_seat[game.current_player_id(state)]
            mcts = MCTSConfig(
                simulations=self.protocol.simulations,
                c_puct=self.protocol.c_puct,
                dirichlet_epsilon=0.0,
                seed=seed_base + moves,
            )
            action = search(game, evaluators[participant_id], mcts).run(
                state, temperature=0.0
            ).selected_action
            state = game.apply_action(state, action)
            moves += 1
        if not game.is_terminal(state):
            return None
        return tuple(participant_by_seat[seat] for seat in game.final_order(state))

    def _write_artifact(
        self, operation_id: str, events: tuple[RatingEvent, ...]
    ) -> None:
        path = self._artifact_path(operation_id)
        payload = {
            "format_version": 1,
            "operation_id": operation_id,
            "protocol_id": self.protocol_id,
            "status": self.status,
            "events": [_event_payload(event) for event in events],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_text(encoding="utf-8") != encoded:
                raise ValueError("conflicting benchmark artifact")
            return
        temporary = path.with_suffix(".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
__all__ = ["ProductionBenchmarkStage"]
