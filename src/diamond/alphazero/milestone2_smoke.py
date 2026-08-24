"""Tiny, deterministic, headless Milestone 2 production smoke runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from ..contract.camps import PLAYABLE_HOLES
from ..contract.state import EMPTY, GameState, PlayerSpec, build_players
from .arena import MinArena, SooArena
from .checkpoint import load_checkpoint, load_inference_checkpoint, save_checkpoint
from .config import ArenaConfig, MCTSConfig, NetworkConfig, SelfPlayConfig, TrainingConfig
from .evaluator.base import EvalRequest, EvalResult
from .evaluator.torch import TorchEvaluator
from .game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from .identity import MIN_MODEL_NAME, SOO_MODEL_NAME, CheckpointCompatibilitySpec
from .inference.coordinator import InferenceConfig, InferenceCoordinator
from .inference.protocol import InferenceRequest, InferenceResponse, ModelKey
from .native import native_game, require_native
from .native.topology import camp_positions, neighbour_table
from .network import MinModel, SooModel
from .orchestration.coordinator import (
    CandidateArtifact,
    PersistenceConfig,
    PromotionArtifact,
    TrainingCoordinator,
    TrainingLoopConfig,
    TrainingStepArtifact,
    WorkerConfig,
)
from .orchestration.replay_store import PersistentReplayStore
from .orchestration.run_state import RunStateStore, TrainingRunState, validate_run_id
from .orchestration.selfplay_workers import EpisodeResult, SelfPlayJob, SelfPlayWorkerPool
from .rating.events import MinRatingEvent, SooRatingEvent
from .rating.participants import CheckpointParticipant
from .rating.protocol import BenchmarkProtocol, EloConfig, TrueSkillConfig
from .rating.registry import RatingRegistry
from .replay import ReplayBatch, ReplayBuffer, TrainingSample
from .trainer import AlphaZeroTrainer

_NETWORK = NetworkConfig(width=8, residual_blocks=1)
_TRAINING = TrainingConfig(batch_size=1, learning_rate=1e-3, weight_decay=0.0, seed=17)


def _near_terminal_state(
    players: tuple[PlayerSpec, ...],
    *,
    finishers: int,
) -> tuple[GameState, tuple[tuple[int, int], ...]]:
    """Build an authoritative state one move from each required placement."""
    occupied = [EMPTY] * PLAYABLE_HOLES
    reserved_targets = {
        position
        for player in players[:finishers]
        for position in camp_positions(player.target_camp)
    }
    used_entries: set[int] = set()
    entries: dict[int, int] = {}
    destinations: dict[int, int] = {}
    for player in players[:finishers]:
        choice: tuple[int, int] | None = None
        for destination in camp_positions(player.target_camp):
            for neighbour in neighbour_table()[destination]:
                if (
                    neighbour >= 0
                    and neighbour not in reserved_targets
                    and neighbour not in used_entries
                ):
                    choice = (neighbour, destination)
                    break
            if choice is not None:
                break
        if choice is None:
            raise RuntimeError("unable to create a near-terminal authoritative game")
        entry, destination = choice
        used_entries.add(entry)
        entries[player.id] = entry
        destinations[player.id] = destination
        for position in camp_positions(player.target_camp):
            if position != destination:
                occupied[position] = player.id
        occupied[entry] = player.id

    state = GameState(tuple(occupied), players[0].id, 40)
    adapter = AlphaZeroGameAdapter(players, initial=state)
    module = require_native()
    native = native_game(players)
    actions: list[tuple[int, int]] = []
    for player in players[:finishers]:
        physical = adapter.codec.encode(entries[player.id], destinations[player.id])
        canonical = adapter.encoder.to_canonical_action(physical, players, player.id)
        # The C++ core is the authority on legality. Each finisher moves on its
        # own turn, so the seat to move is what it is asked about -- which is why
        # this is a probe state rather than ``state`` itself.
        probe = module.State(
            occupancy=list(state.occupancy),
            current_player=player.id,
            turn_number=state.turn_number,
        )
        if canonical not in native.canonical_legal_action_ids(probe):
            raise RuntimeError("near-terminal move was rejected by the native rules")
        actions.append((player.id, canonical))
    return state, tuple(actions)


class _PreferredBatchEvaluator:
    """A deterministic evaluator fixture behind the real central coordinator."""

    def __init__(self, actions: Mapping[tuple[str, int], int]) -> None:
        self.actions = dict(actions)

    def evaluate(self, requests: tuple[InferenceRequest, ...]) -> tuple[InferenceResponse, ...]:
        responses: list[InferenceResponse] = []
        for request in requests:
            preferred = self.actions.get(
                (request.model_key.model_name, request.canonical_player_ids[0]),
                request.legal_action_ids[0],
            )
            if preferred not in request.legal_action_ids:
                preferred = request.legal_action_ids[0]
            priors = {
                action: 1.0 if action == preferred else 0.0
                for action in request.legal_action_ids
            }
            value: float | tuple[float, ...]
            value = 0.0 if request.model_key.player_count == 2 else (0.0, 0.0, 0.0)
            responses.append(
                InferenceResponse.from_eval_result(request, EvalResult(priors, value))
            )
        return tuple(responses)


class _ArtifactStore:
    """Atomic JSON journals plus checkpoint paths for smoke stage artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, stage: str, operation_id: str, suffix: str = ".json") -> Path:
        digest = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()
        return self.root / stage / f"{digest}{suffix}"

    def read(self, stage: str, operation_id: str) -> dict[str, object] | None:
        path = self.path(stage, operation_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid {stage} artifact: {error}") from error
        if not isinstance(payload, dict) or payload.get("operation_id") != operation_id:
            raise ValueError(f"invalid {stage} artifact identity")
        return payload

    def read_all(self, stage: str) -> tuple[dict[str, object], ...]:
        directory = self.root / stage
        if not directory.exists():
            return ()
        payloads: list[dict[str, object]] = []
        for path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid {stage} artifact: {error}") from error
            if not isinstance(payload, dict) or not isinstance(
                payload.get("operation_id"), str
            ):
                raise ValueError(f"invalid {stage} artifact identity")
            payloads.append(payload)
        return tuple(payloads)

    def write(self, stage: str, operation_id: str, payload: Mapping[str, object]) -> Path:
        destination = self.path(stage, operation_id)
        body = {"format_version": 1, "operation_id": operation_id, **dict(payload)}
        encoded = json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_text(encoding="utf-8") != encoded:
                raise ValueError(f"conflicting {stage} artifact")
            return destination
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(destination)
        return destination


def _sample_payload(sample: TrainingSample) -> dict[str, object]:
    return {
        "canonical_player_ids": sample.canonical_player_ids,
        "node_features": sample.node_features,
        "schema_version": sample.schema_version,
        "sparse_policy": sample.sparse_policy,
        "value_target": sample.value_target,
    }


def _sample_from_payload(
    payload: Mapping[str, object], compatibility: CheckpointCompatibilitySpec
) -> TrainingSample:
    return TrainingSample(
        compatibility=compatibility,
        node_features=tuple(  # type: ignore[arg-type]
            tuple(row) for row in payload["node_features"]
        ),
        canonical_player_ids=tuple(payload["canonical_player_ids"]),  # type: ignore[arg-type]
        sparse_policy=tuple(  # type: ignore[arg-type]
            tuple(row) for row in payload["sparse_policy"]
        ),
        value_target=tuple(payload["value_target"]),  # type: ignore[arg-type]
        schema_version=payload["schema_version"],  # type: ignore[arg-type]
    )


def _episode_payload(episode: EpisodeResult) -> dict[str, object]:
    return {
        "aborted_reason": episode.aborted_reason,
        "completed": episode.completed,
        "final_order": episode.final_order,
        "game_id": episode.game_id,
        "model_key": episode.model_key.to_payload(),
        "move_count": episode.move_count,
        "retry_id": episode.retry_id,
        "samples": [_sample_payload(sample) for sample in episode.samples],
        "seed": episode.seed,
        "worker_id": episode.worker_id,
    }


def _episode_from_payload(
    payload: Mapping[str, object], compatibility: CheckpointCompatibilitySpec
) -> EpisodeResult:
    final_order = payload["final_order"]
    return EpisodeResult(
        game_id=payload["game_id"],  # type: ignore[arg-type]
        seed=payload["seed"],  # type: ignore[arg-type]
        retry_id=payload["retry_id"],  # type: ignore[arg-type]
        model_key=ModelKey.from_payload(payload["model_key"]),
        compatibility=compatibility,
        samples=tuple(
            _sample_from_payload(sample, compatibility)
            for sample in payload["samples"]  # type: ignore[union-attr]
        ),
        final_order=(
            tuple(final_order) if final_order is not None else None  # type: ignore[arg-type]
        ),
        move_count=payload["move_count"],  # type: ignore[arg-type]
        completed=payload["completed"],  # type: ignore[arg-type]
        aborted_reason=payload["aborted_reason"],  # type: ignore[arg-type]
        worker_id=payload["worker_id"],  # type: ignore[arg-type]
    )


def _promotion_from_payload(payload: Mapping[str, object]) -> PromotionArtifact:
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("invalid promotion artifact result")
    return PromotionArtifact(
        operation_id=payload["operation_id"],  # type: ignore[arg-type]
        promotion_protocol_id=payload["promotion_protocol_id"],  # type: ignore[arg-type]
        candidate_sha256=payload["candidate_sha256"],  # type: ignore[arg-type]
        champion_checkpoint=payload["champion_checkpoint"],  # type: ignore[arg-type]
        promoted=payload["promoted"],  # type: ignore[arg-type]
        result=dict(result),
    )


def _event_payload(event: SooRatingEvent | MinRatingEvent) -> dict[str, object]:
    common = {
        "completed": event.completed,
        "opening_id": event.opening_id,
        "participant_ids": event.participant_ids,
        "protocol_id": event.protocol_id,
        "seat_assignment": event.seat_assignment,
        "sequence_index": event.sequence_index,
        "turn_order": event.turn_order,
    }
    if isinstance(event, SooRatingEvent):
        return {
            **common,
            "event_type": "soo",
            "loser_id": event.loser_id,
            "winner_id": event.winner_id,
        }
    return {**common, "event_type": "min", "final_ranking": event.final_ranking}


def _event_from_payload(payload: Mapping[str, object]) -> SooRatingEvent | MinRatingEvent:
    common = {
        "completed": payload["completed"],
        "opening_id": payload["opening_id"],
        "participant_ids": tuple(payload["participant_ids"]),  # type: ignore[arg-type]
        "protocol_id": payload["protocol_id"],
        "seat_assignment": tuple(payload["seat_assignment"]),  # type: ignore[arg-type]
        "sequence_index": payload["sequence_index"],
        "turn_order": tuple(payload["turn_order"]),  # type: ignore[arg-type]
    }
    if payload.get("event_type") == "soo":
        return SooRatingEvent(
            **common,  # type: ignore[arg-type]
            winner_id=payload["winner_id"],  # type: ignore[arg-type]
            loser_id=payload["loser_id"],  # type: ignore[arg-type]
        )
    if payload.get("event_type") == "min":
        ranking = payload["final_ranking"]
        return MinRatingEvent(
            **common,  # type: ignore[arg-type]
            final_ranking=tuple(ranking) if ranking is not None else None,  # type: ignore[arg-type]
        )
    raise ValueError("invalid rating artifact event type")


class _WorkerStage:
    def __init__(
        self,
        actions: Mapping[tuple[str, int], int],
        worker_count: int,
        compatibility: CheckpointCompatibilitySpec,
        artifacts: _ArtifactStore,
    ) -> None:
        self.actions = dict(actions)
        self.worker_count = worker_count
        self.compatibility = compatibility
        self.artifacts = artifacts

    def load(self, operation_id: str) -> tuple[EpisodeResult, ...] | None:
        payload = self.artifacts.read("self_play", operation_id)
        if payload is None:
            return None
        episodes = payload.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError("invalid self_play artifact episodes")
        return tuple(
            _episode_from_payload(episode, self.compatibility)
            for episode in episodes
            if isinstance(episode, Mapping)
        )

    def execute(
        self, operation_id: str, jobs: tuple[SelfPlayJob, ...]
    ) -> tuple[EpisodeResult, ...]:
        coordinator = InferenceCoordinator(
            _PreferredBatchEvaluator(self.actions),
            InferenceConfig(
                max_batch_size=2,
                max_wait_ms=1,
                request_queue_capacity=8,
                response_timeout_s=10.0,
            ),
        )
        coordinator.start()
        try:
            episodes = SelfPlayWorkerPool(
                coordinator,
                worker_count=self.worker_count,
                worker_timeout_s=20.0,
                join_timeout_s=2.0,
            ).run(jobs)
        finally:
            coordinator.stop()
        self.artifacts.write(
            "self_play",
            operation_id,
            {"episodes": [_episode_payload(episode) for episode in episodes]},
        )
        return episodes


class _TorchTrainingStage:
    def __init__(self, trainer: AlphaZeroTrainer, artifacts: _ArtifactStore) -> None:
        self.trainer = trainer
        self.artifacts = artifacts

    def load(self, operation_id: str) -> TrainingStepArtifact | None:
        payload = self.artifacts.read("training", operation_id)
        if payload is None:
            return None
        checkpoint = self.artifacts.path("training", operation_id, ".pt")
        load_checkpoint(checkpoint, self.trainer, expected=self.trainer.compatibility)
        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("invalid training artifact metrics")
        return TrainingStepArtifact(
            operation_id=operation_id,
            compatibility_namespace=payload["compatibility_namespace"],  # type: ignore[arg-type]
            input_training_step=payload["input_training_step"],  # type: ignore[arg-type]
            output_training_step=payload["output_training_step"],  # type: ignore[arg-type]
            metrics=dict(metrics),
        )

    def execute(
        self,
        operation_id: str,
        replay: PersistentReplayStore,
        batch_size: int,
        expected_training_step: int,
    ) -> TrainingStepArtifact:
        if self.trainer.training_step != expected_training_step:
            raise ValueError("trainer step does not match authoritative run state")
        samples = replay.sample(batch_size)
        # Scatters the sparse policy straight into a tensor instead of
        # building a dense 5329-wide Python row per sample: measured 310 ms ->
        # 13.6 ms per 512-sample batch, identical values.
        metrics = self.trainer.train_samples(samples, action_size=73 * 73)
        artifact = TrainingStepArtifact(
            operation_id=operation_id,
            compatibility_namespace=_compatibility_namespace(self.trainer.compatibility),
            input_training_step=expected_training_step,
            output_training_step=self.trainer.training_step,
            metrics=asdict(metrics),
        )
        save_checkpoint(
            self.artifacts.path("training", operation_id, ".pt"), self.trainer
        )
        self.artifacts.write(
            "training",
            operation_id,
            {
                "compatibility_namespace": artifact.compatibility_namespace,
                "input_training_step": artifact.input_training_step,
                "metrics": dict(artifact.metrics),
                "output_training_step": artifact.output_training_step,
            },
        )
        return artifact


class _CheckpointStage:
    def __init__(self, trainer: AlphaZeroTrainer, artifacts: _ArtifactStore) -> None:
        self.trainer = trainer
        self.artifacts = artifacts

    def load(self, operation_id: str) -> CandidateArtifact | None:
        payload = self.artifacts.read("candidate", operation_id)
        if payload is None:
            return None
        path = Path(payload["path"])  # type: ignore[arg-type]
        participant = CheckpointParticipant.from_checkpoint(path)
        return CandidateArtifact(
            operation_id=operation_id,
            path=path,
            checkpoint_sha256=payload["checkpoint_sha256"],  # type: ignore[arg-type]
            training_step=payload["training_step"],  # type: ignore[arg-type]
            compatibility_namespace=payload["compatibility_namespace"],  # type: ignore[arg-type]
            participant=participant,
        )

    def execute(
        self, operation_id: str, path: Path, training: TrainingStepArtifact
    ) -> CandidateArtifact:
        save_checkpoint(path, self.trainer)
        participant = CheckpointParticipant.from_checkpoint(path)
        artifact = CandidateArtifact(
            operation_id=operation_id,
            path=path,
            checkpoint_sha256=participant.checkpoint_sha256,
            training_step=training.output_training_step,
            compatibility_namespace=training.compatibility_namespace,
            participant=participant,
        )
        self.artifacts.write(
            "candidate",
            operation_id,
            {
                "checkpoint_sha256": artifact.checkpoint_sha256,
                "compatibility_namespace": artifact.compatibility_namespace,
                "path": str(artifact.path),
                "training_step": artifact.training_step,
            },
        )
        return artifact


class _ArenaStage:
    def __init__(
        self,
        compatibility: CheckpointCompatibilitySpec,
        protocol_id: str,
        artifacts: _ArtifactStore,
    ) -> None:
        self._compatibility = compatibility
        self._protocol_id = protocol_id
        self.artifacts = artifacts

    @property
    def compatibility(self) -> CheckpointCompatibilitySpec:
        return self._compatibility

    @property
    def protocol_id(self) -> str:
        return self._protocol_id

    def load(self, operation_id: str) -> PromotionArtifact | None:
        payload = self.artifacts.read("promotion", operation_id)
        return _promotion_from_payload(payload) if payload is not None else None

    def find(
        self, candidate_sha256: str, champion_checkpoint: str | None
    ) -> PromotionArtifact | None:
        matches = tuple(
            artifact
            for artifact in (
                _promotion_from_payload(payload)
                for payload in self.artifacts.read_all("promotion")
            )
            if artifact.candidate_sha256 == candidate_sha256
            and artifact.champion_checkpoint == champion_checkpoint
        )
        if len(matches) > 1:
            raise ValueError("multiple promotion artifacts match candidate and champion")
        return matches[0] if matches else None

    def execute(
        self,
        operation_id: str,
        candidate: CandidateArtifact,
        champion_checkpoint: str | None,
    ) -> PromotionArtifact:
        if champion_checkpoint is None:
            raise ValueError("promotion arena requires a champion checkpoint")
        player_count = self.compatibility.identity.player_count
        value_size = 1 if player_count == 2 else 3
        candidate_model = _new_model(self.compatibility)
        candidate_info = load_inference_checkpoint(
            candidate.path,
            candidate_model,
            expected=self.compatibility,
            device="cpu",
        )
        champion_model = _new_model(self.compatibility)
        champion_info = load_inference_checkpoint(
            champion_checkpoint,
            champion_model,
            expected=self.compatibility,
            device="cpu",
        )
        candidate_evaluator = TorchEvaluator(
            candidate_model, value_size=value_size, device="cpu"
        )
        champion_evaluator = TorchEvaluator(
            champion_model, value_size=value_size, device="cpu"
        )
        if player_count == 2:
            result = SooArena(
                candidate=candidate_evaluator,
                baseline=champion_evaluator,
                mcts_config=MCTSConfig(simulations=1, dirichlet_epsilon=0.0),
                arena_config=ArenaConfig(
                    games=4, seed=29, max_moves=2, promotion_threshold=0.5
                ),
            ).run(_arena_game_factory(2))
        else:
            result = MinArena(
                candidate=candidate_evaluator,
                baseline=champion_evaluator,
                mcts_config=MCTSConfig(simulations=1, dirichlet_epsilon=0.0),
                arena_config=ArenaConfig(
                    games=18, seed=29, max_moves=2, promotion_threshold=0.0
                ),
            ).run(_arena_game_factory(3))
        result_payload = {
            **asdict(result),
            "candidate_evaluator_sha256": candidate_info.checkpoint_sha256,
            "champion_evaluator_sha256": champion_info.checkpoint_sha256,
        }
        artifact = PromotionArtifact(
            operation_id=operation_id,
            promotion_protocol_id=self.protocol_id,
            candidate_sha256=candidate.checkpoint_sha256,
            champion_checkpoint=champion_checkpoint,
            promoted=result.promoted,
            result=result_payload,
        )
        self.artifacts.write(
            "promotion",
            operation_id,
            {
                "candidate_sha256": artifact.candidate_sha256,
                "champion_checkpoint": artifact.champion_checkpoint,
                "promoted": artifact.promoted,
                "promotion_protocol_id": artifact.promotion_protocol_id,
                "result": dict(artifact.result),
            },
        )
        return artifact


class _RatingStage:
    def __init__(
        self,
        protocol: BenchmarkProtocol,
        champion: CheckpointParticipant,
        artifacts: _ArtifactStore,
        promotion: _ArenaStage,
    ) -> None:
        self.protocol = protocol
        self.champion = champion
        self.artifacts = artifacts
        self.promotion = promotion
        self.status = "eligible"

    @property
    def compatibility(self) -> CheckpointCompatibilitySpec:
        return self.protocol.compatibility

    @property
    def protocol_id(self) -> str:
        return self.protocol.protocol_id

    def load(self, operation_id: str) -> tuple[SooRatingEvent, ...] | None:
        payload = self.artifacts.read("rating", operation_id)
        if payload is None:
            return None
        events = payload.get("events")
        if not isinstance(events, list):
            raise ValueError("invalid rating artifact events")
        status = payload.get("status")
        if not isinstance(status, str):
            raise ValueError("invalid rating artifact status")
        self.status = status
        loaded = tuple(
            _event_from_payload(event)
            for event in events
            if isinstance(event, Mapping)
        )
        if not all(isinstance(event, SooRatingEvent) for event in loaded):
            raise ValueError("Soo rating stage loaded an incompatible event")
        return loaded  # type: ignore[return-value]

    def execute(
        self,
        operation_id: str,
        candidate: CandidateArtifact,
        champion_checkpoint: str | None,
    ) -> tuple[SooRatingEvent, ...]:
        if self.compatibility.identity.model_name == MIN_MODEL_NAME:
            # A Min event needs three distinct artifact identities.  The tiny
            # fixture deliberately has only champion and candidate; never pad
            # the triple with a duplicate or invent a historical rating.
            self.status = "insufficient_history"
            self.artifacts.write(
                "rating", operation_id, {"events": [], "status": self.status}
            )
            return ()
        promotion = self.promotion.find(
            candidate.checkpoint_sha256, champion_checkpoint
        )
        if promotion is None:
            raise ValueError("rating benchmark requires a persisted promotion result")
        wins = promotion.result.get("wins")
        losses = promotion.result.get("losses")
        if (
            not isinstance(wins, int)
            or isinstance(wins, bool)
            or not isinstance(losses, int)
            or isinstance(losses, bool)
            or wins < 0
            or losses < 0
        ):
            raise ValueError("promotion result has invalid completed Soo outcomes")
        events: list[SooRatingEvent] = []
        outcomes = (
            *((candidate.participant.participant_id, self.champion.participant_id),) * wins,
            *((self.champion.participant_id, candidate.participant.participant_id),) * losses,
        )
        for index, (winner_id, loser_id) in enumerate(outcomes):
            events.append(
                SooRatingEvent(
                    sequence_index=index,
                    protocol_id=self.protocol.protocol_id,
                    participant_ids=(
                        candidate.participant.participant_id,
                        self.champion.participant_id,
                    ),
                    seat_assignment=(1, 2) if index % 2 == 0 else (2, 1),
                    turn_order=(1, 2) if (index // 2) % 2 == 0 else (2, 1),
                    opening_id=f"tiny-promotion-arena-{index}",
                    completed=True,
                    winner_id=winner_id,
                    loser_id=loser_id,
                )
            )
        self.status = "eligible" if events else "no_completed_outcome"
        result = tuple(events)
        self.artifacts.write(
            "rating",
            operation_id,
            {
                "events": [_event_payload(event) for event in result],
                "status": self.status,
            },
        )
        return result


def _compatibility_namespace(compatibility: CheckpointCompatibilitySpec) -> str:
    payload = json.dumps(
        compatibility.to_metadata(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _new_model(compatibility: CheckpointCompatibilitySpec) -> SooModel | MinModel:
    if compatibility.identity.model_name == SOO_MODEL_NAME:
        return SooModel(
            compatibility.network_config, model_version=compatibility.identity.model_version
        )
    return MinModel(
        compatibility.network_config, model_version=compatibility.identity.model_version
    )


def _arena_game_factory(player_count: int):
    def factory(order: tuple[int, ...]) -> DiamondSearchAdapter:
        players = build_players(player_count, order=order)
        return _forced_terminal_game(players)

    return factory


def _forced_terminal_game(players: tuple[PlayerSpec, ...]) -> DiamondSearchAdapter:
    """Return a game where every legal action completes the current placement."""
    already_finished = players[: len(players) - 2]
    active = players[len(players) - 2]
    occupancy = [EMPTY] * PLAYABLE_HOLES
    for player in already_finished:
        for position in camp_positions(player.target_camp):
            occupancy[position] = player.id
    active_target = camp_positions(active.target_camp)
    reserved_targets = {
        position
        for player in (*already_finished, active)
        for position in camp_positions(player.target_camp)
    }
    entry: int | None = None
    destination: int | None = None
    for candidate in active_target:
        for neighbour in neighbour_table()[candidate]:
            if neighbour >= 0 and neighbour not in reserved_targets:
                entry, destination = neighbour, candidate
                break
        if entry is not None:
            break
    if entry is None or destination is None:
        raise RuntimeError("unable to create forced terminal arena state")
    for position in active_target:
        if position != destination:
            occupancy[position] = active.id
    occupancy[entry] = active.id
    blocker = players[-1].id
    for position, occupant in enumerate(occupancy):
        if occupant == EMPTY and position != destination:
            occupancy[position] = blocker
    forced = GameState(
        tuple(occupancy),
        active.id,
        40,
        finish_order=tuple(player.id for player in already_finished),
    )
    game = DiamondSearchAdapter(
        AlphaZeroGameAdapter(players, initial=forced)
    )
    legal = game.legal_action_ids(forced)
    if not legal or any(
        not game.is_terminal(game.apply_action(forced, action)) for action in legal
    ):
        raise RuntimeError("forced arena actions must all reach terminal state")
    if any(
        active.id not in game.apply_action(forced, action).finish_order
        for action in legal
    ):
        raise RuntimeError("forced arena fixture did not reach an authoritative terminal state")
    return game


def _build_jobs(
    state: TrainingRunState,
    config: WorkerConfig,
    compatibility: CheckpointCompatibilitySpec,
    model_key: ModelKey,
) -> tuple[SelfPlayJob, ...]:
    players = build_players(compatibility.identity.player_count)
    initial, _actions = _near_terminal_state(
        players, finishers=compatibility.identity.player_count - 1
    )
    return tuple(
        SelfPlayJob(
            run_seed=state.run_seed,
            iteration=state.iteration,
            game_index=index,
            retry_id=config.retry_id,
            model_key=model_key,
            compatibility=compatibility,
            players=players,
            initial_state=initial,
            mcts_config=MCTSConfig(simulations=1, dirichlet_epsilon=0.0),
            selfplay_config=SelfPlayConfig(max_moves=2, temperature_moves=0),
        )
        for index in range(config.games_per_iteration)
    )


def _champion_model_key(state: TrainingRunState) -> ModelKey:
    if state.champion_model_key is None:
        raise ValueError("smoke run has no durable champion model key")
    return state.champion_model_key


class TinyTrainingServices:
    """Tiny fixture assembly used only by the Milestone 2 smoke and tests."""

    def __init__(self, root: Path, model_name: str) -> None:
        self.root = Path(root)
        if model_name not in (SOO_MODEL_NAME, MIN_MODEL_NAME):
            raise ValueError("model_name must be Soo or Min")
        self.model_name = model_name
        self.compatibility = (
            CheckpointCompatibilitySpec.soo(model_version="2.0.0", network_config=_NETWORK)
            if model_name == SOO_MODEL_NAME
            else CheckpointCompatibilitySpec.min(model_version="2.0.0", network_config=_NETWORK)
        )
        self.worker_config = WorkerConfig(worker_count=2, games_per_iteration=2)
        self.promotion_protocol_id = f"tiny-{model_name.lower()}-promotion-v1"
        self.protocol = BenchmarkProtocol(
            compatibility=self.compatibility,
            simulations=1,
            c_puct=1.5,
            dirichlet_epsilon=0.0,
            decision_temperature=0.0,
            max_game_moves=2,
            opening_suite_version="tiny-authoritative-openings-v1",
            opening_suite_hash="sha256:tiny-authoritative-openings",
            rating_system_version=(
                EloConfig().rating_system_version
                if model_name == SOO_MODEL_NAME
                else TrueSkillConfig().rating_system_version
            ),
            rating_parameters=asdict(
                EloConfig() if model_name == SOO_MODEL_NAME else TrueSkillConfig()
            ),
        )

    def _runtime(
        self, run_id: str, *, create: bool
    ) -> tuple[TrainingCoordinator, RunStateStore, _RatingStage, _CheckpointStage]:
        run_root = self.root / self.model_name.lower() / run_id
        bootstrap = run_root / "bootstrap.pt"
        if create:
            if bootstrap.exists():
                raise ValueError(f"training run already exists: {run_id}")
            trainer = AlphaZeroTrainer(
                _new_model(self.compatibility), self.compatibility, _TRAINING
            )
            save_checkpoint(bootstrap, trainer)
            champion = CheckpointParticipant.from_checkpoint(bootstrap)
            registry = RatingRegistry(self.protocol)
            registry.add_participant(champion)
        else:
            if not bootstrap.exists():
                raise ValueError(f"training run does not exist: {run_id} ({self.model_name})")
            champion = CheckpointParticipant.from_checkpoint(bootstrap)
            registry_path = run_root / "ratings" / "registry.json"
            registry = (
                RatingRegistry.load(registry_path)
                if registry_path.exists()
                else RatingRegistry(self.protocol)
            )
            if not registry.participants:
                registry.add_participant(champion)

        players = build_players(self.compatibility.identity.player_count)
        _state, actions = _near_terminal_state(
            players, finishers=self.compatibility.identity.player_count - 1
        )
        actions_by_model = {(self.model_name, player_id): action for player_id, action in actions}
        state_store = RunStateStore(self.root)
        replay = PersistentReplayStore(
            run_root / "replay", self.compatibility, capacity=8, seed=17
        )
        artifacts = _ArtifactStore(run_root / "artifacts")
        trainer = AlphaZeroTrainer(_new_model(self.compatibility), self.compatibility, _TRAINING)
        worker = _WorkerStage(
            actions_by_model,
            self.worker_config.worker_count,
            self.compatibility,
            artifacts,
        )
        checkpoints = _CheckpointStage(trainer, artifacts)
        promotion = _ArenaStage(
            self.compatibility, self.promotion_protocol_id, artifacts
        )
        rating = _RatingStage(self.protocol, champion, artifacts, promotion)
        coordinator = TrainingCoordinator(
            state_store=state_store,
            compatibility=self.compatibility,
            worker_config=self.worker_config,
            loop_config=TrainingLoopConfig(
                replay_batch_size=1,
                promotion_protocol_id=self.promotion_protocol_id,
                benchmark_protocol_id=self.protocol.protocol_id,
            ),
            persistence_config=PersistenceConfig(root=self.root),
            build_selfplay_jobs=lambda state, config: _build_jobs(
                state,
                config,
                self.compatibility,
                _champion_model_key(state),
            ),
            self_play=worker,
            replay_store=replay,
            training=_TorchTrainingStage(trainer, artifacts),
            checkpoints=checkpoints,
            promotion=promotion,
            benchmark=rating,
            rating_registry=registry,
        )
        return coordinator, state_store, rating, checkpoints

    def train(self, *, model_name: str, run_id: str) -> Mapping[str, object]:
        self._assert_model(model_name)
        validate_run_id(run_id)
        coordinator, state_store, rating, checkpoints = self._runtime(run_id, create=True)
        initial = state_store.initialize(
            run_id=run_id,
            compatibility=self.compatibility,
            run_seed=17,
            protocol_ids={
                "promotion": self.promotion_protocol_id,
                "rating": self.protocol.protocol_id,
            },
            champion_checkpoint=str(self.root / self.model_name.lower() / run_id / "bootstrap.pt"),
            champion_model_key=ModelKey(
                self.model_name,
                self.compatibility.identity.model_version,
                CheckpointParticipant.from_checkpoint(
                    self.root / self.model_name.lower() / run_id / "bootstrap.pt"
                ).checkpoint_sha256,
            ),
        )
        complete = coordinator.run_iteration(initial)
        return self._summary(complete, rating, checkpoints)

    def resume(self, *, model_name: str, run_id: str) -> Mapping[str, object]:
        self._assert_model(model_name)
        validate_run_id(run_id)
        coordinator, _state_store, rating, checkpoints = self._runtime(run_id, create=False)
        complete = coordinator.resume(run_id, self.model_name)
        return self._summary(complete, rating, checkpoints)

    def benchmark(self, *, model_name: str, run_id: str) -> Mapping[str, object]:
        self._assert_model(model_name)
        validate_run_id(run_id)
        registry = self._load_registry(run_id)
        return {
            "events": len(registry.events),
            "rating_status": (
                "eligible"
                if registry.has_sufficient_min_history
                or self.model_name == SOO_MODEL_NAME
                else "insufficient_history"
            ),
        }

    def leaderboard(self, *, model_name: str, run_id: str) -> Mapping[str, object]:
        self._assert_model(model_name)
        validate_run_id(run_id)
        registry = self._load_registry(run_id)
        entries: list[dict[str, object]] = []
        if self.model_name == SOO_MODEL_NAME:
            entries = [asdict(entry) for entry in registry.soo_leaderboard()]
        else:
            entries = [asdict(entry) for entry in registry.min_leaderboard()]
        return {"entries": entries}

    def profile(self, *, model_name: str, max_seconds: int):
        """Return a bounded real-model profile without creating a training run."""
        self._assert_model(model_name)
        from .inference.profile import detect_hardware, profile_evaluator

        hardware = detect_hardware()
        device = "cuda" if hardware.gpu_verified else "cpu"
        evaluator = TorchEvaluator(
            _new_model(self.compatibility),
            value_size=1 if model_name == SOO_MODEL_NAME else 3,
            device=device,
        )
        request = EvalRequest(
            node_features=tuple(
                (0.0,) * (self.compatibility.identity.player_count * 2) for _ in range(73)
            ),
            legal_action_ids=(0, 1, 2),
            canonical_player_ids=tuple(
                range(1, self.compatibility.identity.player_count + 1)
            ),
        )
        return profile_evaluator(
            evaluator,
            (request,),
            max_seconds=max_seconds,
            stage_operations=self._profile_stage_operations(),
        )

    def _profile_stage_operations(self):
        """Build one real tiny self-play, collation, and training operation."""
        players = build_players(self.compatibility.identity.player_count)
        initial, actions = _near_terminal_state(
            players, finishers=self.compatibility.identity.player_count - 1
        )
        actions_by_model = {
            (self.model_name, player_id): action for player_id, action in actions
        }
        model_key = ModelKey(
            self.model_name,
            self.compatibility.identity.model_version,
            "0" * 64,
        )
        job = SelfPlayJob(
            run_seed=17,
            iteration=0,
            game_index=0,
            retry_id="profile",
            model_key=model_key,
            compatibility=self.compatibility,
            players=players,
            initial_state=initial,
            mcts_config=MCTSConfig(simulations=1, dirichlet_epsilon=0.0),
            selfplay_config=SelfPlayConfig(max_moves=2, temperature_moves=0),
        )
        replay = ReplayBuffer(self.compatibility, capacity=8, seed=17)
        trainer = AlphaZeroTrainer(
            _new_model(self.compatibility), self.compatibility, _TRAINING
        )
        episodes: tuple[EpisodeResult, ...] | None = None
        batch: ReplayBatch | None = None

        def self_play() -> tuple[EpisodeResult, ...]:
            nonlocal episodes
            coordinator = InferenceCoordinator(
                _PreferredBatchEvaluator(actions_by_model),
                InferenceConfig(
                    max_batch_size=1,
                    max_wait_ms=1,
                    request_queue_capacity=2,
                    response_timeout_s=10.0,
                ),
            )
            coordinator.start()
            try:
                episodes = SelfPlayWorkerPool(
                    coordinator,
                    worker_count=1,
                    worker_timeout_s=20.0,
                    join_timeout_s=2.0,
                ).run((job,))
            finally:
                coordinator.stop()
            if not episodes or not episodes[0].completed:
                raise RuntimeError("tiny profile self-play did not complete")
            return episodes

        def replay_collation() -> ReplayBatch:
            nonlocal batch
            if episodes is None:
                raise RuntimeError("self-play must run before replay collation")
            samples = tuple(sample for episode in episodes for sample in episode.samples)
            if not samples:
                raise RuntimeError("tiny profile self-play produced no replay samples")
            replay.extend(samples)
            batch = replay.collate((replay.samples[0],), action_size=73 * 73)
            return batch

        def training():
            if batch is None:
                raise RuntimeError("replay collation must run before training")
            return trainer.train_batch(batch)

        return {
            "self_play": self_play,
            "replay_collation": replay_collation,
            "training": training,
        }

    def _load_registry(self, run_id: str) -> RatingRegistry:
        path = self.root / self.model_name.lower() / run_id / "ratings" / "registry.json"
        if not path.exists():
            raise ValueError(f"rating registry does not exist: {run_id} ({self.model_name})")
        return RatingRegistry.load(path)

    def _assert_model(self, model_name: str) -> None:
        if model_name != self.model_name:
            raise ValueError("command model does not match configured services")

    def _summary(
        self,
        state: TrainingRunState,
        rating: _RatingStage,
        checkpoints: _CheckpointStage,
    ) -> dict[str, object]:
        if state.rating_records:
            operation_id = state.rating_records[-1].get("operation_id")
            if not isinstance(operation_id, str):
                raise ValueError("persisted rating record has invalid operation identity")
            if rating.load(operation_id) is None:
                raise ValueError("completed rating artifact is missing")
        return {
            "candidate_checkpoint": state.candidate_checkpoint,
            "rating_events": len(state.rating_records[-1]["event_ids"]),
            "rating_status": rating.status,
            "read_only_checkpoint_loads": (
                2
                if state.promotion_records
                and "candidate_evaluator_sha256"
                in state.promotion_records[-1]["result"]
                and "champion_evaluator_sha256"
                in state.promotion_records[-1]["result"]
                else 0
            ),
            "stage": state.stage.value,
            "training_step": state.training_step,
        }


def build_tiny_services(root: Path, model_name: str) -> TinyTrainingServices:
    """Build the bounded smoke fixture runtime."""
    return TinyTrainingServices(root, model_name)


def _run_model(root: Path, model_name: str) -> dict[str, object]:
    services = build_tiny_services(root, model_name)
    run_id = f"smoke-{model_name.lower()}"
    result = dict(services.train(model_name=model_name, run_id=run_id))
    state_store = RunStateStore(root)
    persisted = state_store.load(run_id, model_name)
    resumed = build_tiny_services(root, model_name).resume(
        model_name=model_name, run_id=run_id
    )
    replay = PersistentReplayStore(
        root / model_name.lower() / run_id / "replay", services.compatibility, capacity=8, seed=17
    )
    registry = services._load_registry(run_id)
    return {
        "candidate_read_only_loaded": result["read_only_checkpoint_loads"] == 2,
        "participant_count": len(registry.participants),
        "rating_events": result["rating_events"],
        "rating_status": resumed["rating_status"],
        "replay_reloaded": len(replay.load_buffer()) > 0,
        "state_reloaded": (
            persisted == state_store.load(run_id, model_name)
            and resumed == result
        ),
        "training_step": result["training_step"],
        "worker_games": len(persisted.completed_game_ids),
    }


def run_smoke(runtime_dir: str | Path | None = None) -> dict[str, object]:
    """Run tiny authoritative Soo and Min iterations in a temporary runtime."""
    if runtime_dir is None:
        with tempfile.TemporaryDirectory(prefix="alphadiamond-m2-") as temporary:
            return run_smoke(Path(temporary))
    root = Path(runtime_dir)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "models": {
            SOO_MODEL_NAME: _run_model(root, SOO_MODEL_NAME),
            MIN_MODEL_NAME: _run_model(root, MIN_MODEL_NAME),
        },
        "status": "ok",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m diamond.alphazero.milestone2_smoke")
    parser.add_argument("--runtime-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_smoke(args.runtime_dir), sort_keys=True, separators=(",", ":")))
    except ValueError as error:
        print(json.dumps({"error": str(error), "status": "error"}, sort_keys=True))
        return 3
    except Exception as error:  # noqa: BLE001 - a CLI reports failures as JSON, never a traceback
        print(
            json.dumps(
                {"error": f"{type(error).__name__}: {error}", "status": "error"},
                sort_keys=True,
            )
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
