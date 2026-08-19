"""One-iteration, resumable coordination over durable AlphaZero stages."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, TypeAlias

from ..identity import CheckpointCompatibilitySpec
from ..rating.events import MinRatingEvent, SooRatingEvent
from ..rating.participants import CheckpointParticipant
from ..rating.registry import RatingRegistry
from .replay_store import PersistentReplayStore
from .run_state import RunStage, RunStateError, RunStateStore, TrainingRunState
from .selfplay_workers import EpisodeResult, SelfPlayJob

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
RatingEvent: TypeAlias = SooRatingEvent | MinRatingEvent


class CoordinatorError(ValueError):
    """A durable stage artifact is missing, incompatible, or inconsistent."""


def _positive_int(value: object, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _non_empty(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Only the worker settings that affect one self-play iteration."""

    worker_count: int
    games_per_iteration: int
    retry_id: str = "attempt-0"

    def __post_init__(self) -> None:
        _positive_int(self.worker_count, "worker_count")
        _positive_int(self.games_per_iteration, "games_per_iteration")
        _non_empty(self.retry_id, "retry_id")


@dataclass(frozen=True, slots=True)
class TrainingLoopConfig:
    """One optimizer update and its immutable evaluation namespaces."""

    replay_batch_size: int
    promotion_protocol_id: str
    benchmark_protocol_id: str

    def __post_init__(self) -> None:
        _positive_int(self.replay_batch_size, "replay_batch_size")
        _non_empty(self.promotion_protocol_id, "promotion_protocol_id")
        _non_empty(self.benchmark_protocol_id, "benchmark_protocol_id")


@dataclass(frozen=True, slots=True)
class PersistenceConfig:
    """Run-local immutable checkpoint and rating artifact layout."""

    root: Path
    checkpoint_suffix: str = ".pt"

    def __post_init__(self) -> None:
        root = Path(self.root)
        if not str(root):
            raise ValueError("root must be a path")
        if not self.checkpoint_suffix.startswith(".") or any(
            separator in self.checkpoint_suffix for separator in ("/", "\\")
        ):
            raise ValueError("checkpoint_suffix must be a filename suffix")
        object.__setattr__(self, "root", root)

    def run_path(self, state: TrainingRunState) -> Path:
        return self.root / state.model_name.lower() / state.run_id

    def candidate_path(self, state: TrainingRunState, training_step: int) -> Path:
        return self.run_path(state) / "checkpoints" / (
            f"candidate-i{state.iteration:06d}-step{training_step:012d}"
            f"{self.checkpoint_suffix}"
        )

    def rating_registry_path(self, state: TrainingRunState) -> Path:
        return self.run_path(state) / "ratings" / "registry.json"


@dataclass(frozen=True, slots=True)
class TrainingStepArtifact:
    """Durable proof that one exact optimizer operation has completed."""

    operation_id: str
    compatibility_namespace: str
    input_training_step: int
    output_training_step: int
    metrics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CandidateArtifact:
    """Identity of one immutable checkpoint produced by a training operation."""

    operation_id: str
    path: Path
    checkpoint_sha256: str
    training_step: int
    compatibility_namespace: str
    participant: CheckpointParticipant

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True, slots=True)
class PromotionArtifact:
    """Durable candidate-versus-champion arena decision."""

    operation_id: str
    promotion_protocol_id: str
    candidate_sha256: str
    champion_checkpoint: str | None
    promoted: bool
    result: Mapping[str, object] = field(default_factory=dict)


class SelfPlayStage(Protocol):
    def load(self, operation_id: str) -> tuple[EpisodeResult, ...] | None: ...

    def execute(
        self, operation_id: str, jobs: tuple[SelfPlayJob, ...]
    ) -> tuple[EpisodeResult, ...]: ...


class TrainingStage(Protocol):
    """A loader must restore the trainer state represented by its artifact."""

    def load(self, operation_id: str) -> TrainingStepArtifact | None: ...

    def execute(
        self,
        operation_id: str,
        replay: PersistentReplayStore,
        batch_size: int,
        expected_training_step: int,
    ) -> TrainingStepArtifact: ...


class CheckpointStage(Protocol):
    def load(self, operation_id: str) -> CandidateArtifact | None: ...

    def execute(
        self,
        operation_id: str,
        path: Path,
        training: TrainingStepArtifact,
    ) -> CandidateArtifact: ...


class PromotionStage(Protocol):
    @property
    def compatibility(self) -> CheckpointCompatibilitySpec: ...

    @property
    def protocol_id(self) -> str: ...

    def load(self, operation_id: str) -> PromotionArtifact | None: ...

    def execute(
        self,
        operation_id: str,
        candidate: CandidateArtifact,
        champion_checkpoint: str | None,
    ) -> PromotionArtifact: ...


class BenchmarkStage(Protocol):
    @property
    def compatibility(self) -> CheckpointCompatibilitySpec: ...

    @property
    def protocol_id(self) -> str: ...

    def load(self, operation_id: str) -> tuple[RatingEvent, ...] | None: ...

    def execute(
        self,
        operation_id: str,
        candidate: CandidateArtifact,
        champion_checkpoint: str | None,
    ) -> tuple[RatingEvent, ...]: ...


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CoordinatorError(f"operation identity must contain JSON data: {error}") from error


def _operation_id(stage: RunStage, state: TrainingRunState, payload: object) -> str:
    digest = hashlib.sha256(
        _canonical_json(
            {
                "compatibility_namespace": state.compatibility_namespace,
                "iteration": state.iteration,
                "payload": payload,
                "run_id": state.run_id,
                "stage": stage.value,
            }
        )
    ).hexdigest()
    return f"sha256:{digest}"


class TrainingCoordinator:
    """Advance one run through exactly one finite, durable iteration."""

    def __init__(
        self,
        *,
        state_store: RunStateStore,
        compatibility: CheckpointCompatibilitySpec,
        worker_config: WorkerConfig,
        loop_config: TrainingLoopConfig,
        persistence_config: PersistenceConfig,
        build_selfplay_jobs: Callable[
            [TrainingRunState, WorkerConfig], tuple[SelfPlayJob, ...]
        ],
        self_play: SelfPlayStage,
        replay_store: PersistentReplayStore,
        training: TrainingStage,
        checkpoints: CheckpointStage,
        promotion: PromotionStage,
        benchmark: BenchmarkStage,
        rating_registry: RatingRegistry,
    ) -> None:
        self.state_store = state_store
        self.compatibility = compatibility
        self.worker_config = worker_config
        self.loop_config = loop_config
        self.persistence_config = persistence_config
        self.build_selfplay_jobs = build_selfplay_jobs
        self.self_play = self_play
        self.replay_store = replay_store
        self.training = training
        self.checkpoints = checkpoints
        self.promotion = promotion
        self.benchmark = benchmark
        self.rating_registry = rating_registry

    def run_iteration(self, state: TrainingRunState) -> TrainingRunState:
        """Advance the supplied authoritative snapshot through one iteration."""
        self._validate_authoritative_state(state)
        current = state
        remaining = len(tuple(RunStage)) - tuple(RunStage).index(current.stage)
        for _ in range(remaining):
            if current.stage is RunStage.COMPLETE:
                break
            current = self._advance(current)
        return current

    def resume(self, run_id: str, model_name: str) -> TrainingRunState:
        """Load an interrupted run and advance only its unfinished stages."""
        return self.run_iteration(self.state_store.load(run_id, model_name))

    def start_next_iteration(self, state: TrainingRunState) -> TrainingRunState:
        """Atomically open the next iteration from an authoritative completed run."""
        self._validate_authoritative_state(state)
        return self.state_store.start_next_iteration(state)

    def _validate_authoritative_state(self, state: TrainingRunState) -> None:
        if not isinstance(state, TrainingRunState):
            raise CoordinatorError("state must be a TrainingRunState")
        authoritative = self.state_store.load(state.run_id, state.model_name)
        if authoritative != state:
            raise RunStateError("supplied training run state is stale")
        if dict(state.compatibility) != self.compatibility.to_metadata():
            raise CoordinatorError("training run compatibility does not match coordinator")
        if self.replay_store.compatibility != self.compatibility:
            raise CoordinatorError("replay compatibility does not match coordinator")
        if self.rating_registry.protocol.compatibility != self.compatibility:
            raise CoordinatorError("rating compatibility does not match coordinator")
        expected_protocols = {
            "promotion": self.loop_config.promotion_protocol_id,
            "rating": self.loop_config.benchmark_protocol_id,
        }
        if dict(state.protocol_ids) != expected_protocols:
            raise CoordinatorError("training run protocol namespace changed")
        if self.rating_registry.protocol.protocol_id != self.loop_config.benchmark_protocol_id:
            raise CoordinatorError("rating registry benchmark protocol changed")
        self._validate_evaluation_identity(
            "promotion",
            self.promotion.compatibility,
            self.promotion.protocol_id,
            self.loop_config.promotion_protocol_id,
        )
        self._validate_evaluation_identity(
            "benchmark",
            self.benchmark.compatibility,
            self.benchmark.protocol_id,
            self.loop_config.benchmark_protocol_id,
        )

    def _validate_evaluation_identity(
        self,
        stage_name: str,
        compatibility: CheckpointCompatibilitySpec,
        protocol_id: str,
        expected_protocol_id: str,
    ) -> None:
        if compatibility != self.compatibility:
            raise CoordinatorError(f"{stage_name} stage compatibility does not match run")
        if protocol_id != expected_protocol_id:
            raise CoordinatorError(f"{stage_name} stage protocol identity does not match run")

    def _advance(self, state: TrainingRunState) -> TrainingRunState:
        handlers = {
            RunStage.INITIALIZE: self._initialize,
            RunStage.SELF_PLAY: self._self_play,
            RunStage.REPLAY_INGEST: self._replay_ingest,
            RunStage.TRAIN: self._train,
            RunStage.SAVE_CANDIDATE: self._save_candidate,
            RunStage.PROMOTION_ARENA: self._promotion_arena,
            RunStage.RATING_BENCHMARK: self._rating_benchmark,
            RunStage.PROMOTE_OR_REJECT: self._promote_or_reject,
            RunStage.PERSIST: self._persist,
        }
        return handlers[state.stage](state)

    def _initialize(self, state: TrainingRunState) -> TrainingRunState:
        operation_id = _operation_id(
            RunStage.INITIALIZE,
            state,
            {
                "champion_checkpoint": state.champion_checkpoint,
                "loop_config": asdict(self.loop_config),
                "worker_config": asdict(self.worker_config),
            },
        )
        return self.state_store.transition(
            state,
            RunStage.SELF_PLAY,
            completion_marker=operation_id,
        )

    def _self_play(self, state: TrainingRunState) -> TrainingRunState:
        jobs = self.build_selfplay_jobs(state, self.worker_config)
        self._validate_jobs(state, jobs)
        operation_id = _operation_id(
            RunStage.SELF_PLAY,
            state,
            {
                "champion_checkpoint": state.champion_checkpoint,
                "jobs": [
                    {
                        "game_id": job.game_id,
                        "model_key": job.model_key.to_payload(),
                        "seed": job.seed,
                    }
                    for job in jobs
                ],
            },
        )
        episodes = self.self_play.load(operation_id)
        if episodes is None:
            episodes = self.self_play.execute(operation_id, jobs)
        self._validate_episodes(jobs, episodes)
        completed_ids = tuple(episode.game_id for episode in episodes if episode.completed)
        if set(completed_ids) & set(state.completed_game_ids):
            raise CoordinatorError("self-play produced a duplicate completed game ID")
        return self.state_store.transition(
            state,
            RunStage.REPLAY_INGEST,
            completion_marker=operation_id,
            completed_game_ids=(*state.completed_game_ids, *completed_ids),
        )

    def _replay_ingest(self, state: TrainingRunState) -> TrainingRunState:
        selfplay_id = self._completion(state, RunStage.SELF_PLAY)
        episodes = self.self_play.load(selfplay_id)
        if episodes is None:
            raise CoordinatorError("completed self-play artifact is missing")
        operation_id = _operation_id(
            RunStage.REPLAY_INGEST,
            state,
            {
                "episode_ids": [episode.game_id for episode in episodes],
                "replay_manifest": str(self.replay_store.manifest_path),
                "self_play_operation": selfplay_id,
            },
        )
        for episode in episodes:
            self.replay_store.ingest_episode(episode)
        completed = set(self.replay_store.manifest.game_ids)
        aborted = {row.game_id for row in self.replay_store.manifest.aborted}
        for episode in episodes:
            expected = completed if episode.completed else aborted
            if episode.game_id not in expected:
                raise CoordinatorError("replay manifest omitted a self-play artifact")
        manifest_path = str(self.replay_store.manifest_path)
        if state.replay_manifest is not None and state.replay_manifest != manifest_path:
            raise CoordinatorError("run state references an inconsistent replay manifest")
        return self.state_store.transition(
            state,
            RunStage.TRAIN,
            completion_marker=operation_id,
            replay_manifest=manifest_path,
        )

    def _train(self, state: TrainingRunState) -> TrainingRunState:
        operation_id = _operation_id(
            RunStage.TRAIN,
            state,
            {
                "batch_size": self.loop_config.replay_batch_size,
                "input_training_step": state.training_step,
                "replay_operation": self._completion(state, RunStage.REPLAY_INGEST),
            },
        )
        artifact = self.training.load(operation_id)
        if artifact is None:
            artifact = self.training.execute(
                operation_id,
                self.replay_store,
                self.loop_config.replay_batch_size,
                state.training_step,
            )
        self._validate_training_artifact(state, operation_id, artifact)
        return self.state_store.transition(
            state,
            RunStage.SAVE_CANDIDATE,
            completion_marker=operation_id,
            training_step=artifact.output_training_step,
        )

    def _save_candidate(self, state: TrainingRunState) -> TrainingRunState:
        training_id = self._completion(state, RunStage.TRAIN)
        training = self.training.load(training_id)
        if training is None:
            raise CoordinatorError("completed training artifact is missing")
        path = self.persistence_config.candidate_path(state, state.training_step)
        operation_id = _operation_id(
            RunStage.SAVE_CANDIDATE,
            state,
            {
                "candidate_path": str(path),
                "training_operation": training_id,
                "training_step": state.training_step,
            },
        )
        artifact = self.checkpoints.load(operation_id)
        if artifact is None:
            if path.exists():
                raise CoordinatorError("refusing to overwrite an unjournaled candidate checkpoint")
            artifact = self.checkpoints.execute(operation_id, path, training)
        self._validate_candidate(state, operation_id, path, artifact)
        pointer = str(path)
        if state.candidate_checkpoint is not None and state.candidate_checkpoint != pointer:
            raise CoordinatorError("run state references an inconsistent candidate checkpoint")
        return self.state_store.transition(
            state,
            RunStage.PROMOTION_ARENA,
            completion_marker=operation_id,
            candidate_checkpoint=pointer,
        )

    def _promotion_arena(self, state: TrainingRunState) -> TrainingRunState:
        candidate = self._load_candidate(state)
        operation_id = _operation_id(
            RunStage.PROMOTION_ARENA,
            state,
            {
                "candidate_sha256": candidate.checkpoint_sha256,
                "champion_checkpoint": state.champion_checkpoint,
                "promotion_protocol_id": self.loop_config.promotion_protocol_id,
            },
        )
        artifact = self.promotion.load(operation_id)
        if artifact is None:
            artifact = self.promotion.execute(
                operation_id,
                candidate,
                state.champion_checkpoint,
            )
        self._validate_promotion(state, operation_id, candidate, artifact)
        record = {
            "iteration": state.iteration,
            "operation_id": operation_id,
            "promotion_protocol_id": artifact.promotion_protocol_id,
            "candidate_checkpoint": str(candidate.path),
            "candidate_sha256": candidate.checkpoint_sha256,
            "champion_checkpoint": state.champion_checkpoint,
            "promoted": artifact.promoted,
            "result": dict(artifact.result),
        }
        return self.state_store.transition(
            state,
            RunStage.RATING_BENCHMARK,
            completion_marker=operation_id,
            promotion_records=(*state.promotion_records, record),
        )

    def _rating_benchmark(self, state: TrainingRunState) -> TrainingRunState:
        candidate = self._load_candidate(state)
        operation_id = _operation_id(
            RunStage.RATING_BENCHMARK,
            state,
            {
                "benchmark_protocol_id": self.loop_config.benchmark_protocol_id,
                "candidate_sha256": candidate.checkpoint_sha256,
                "champion_checkpoint": state.champion_checkpoint,
                "rating_record_count": len(state.rating_records),
            },
        )
        self.rating_registry.add_participant(candidate.participant)
        events = self.benchmark.load(operation_id)
        if events is None:
            events = self.benchmark.execute(
                operation_id,
                candidate,
                state.champion_checkpoint,
            )
        self._validate_rating_events(candidate, events)
        existing = {event.event_id: event for event in self.rating_registry.events}
        for event in events:
            recorded = existing.get(event.event_id)
            if recorded is not None:
                if recorded != event:
                    raise CoordinatorError("rating event identity has inconsistent content")
                continue
            self.rating_registry.record_event(event)
        self._persist_rating_registry(state)
        record = {
            "iteration": state.iteration,
            "operation_id": operation_id,
            "protocol_id": self.loop_config.benchmark_protocol_id,
            "candidate_checkpoint": str(candidate.path),
            "candidate_sha256": candidate.checkpoint_sha256,
            "event_ids": tuple(event.event_id for event in events),
        }
        return self.state_store.transition(
            state,
            RunStage.PROMOTE_OR_REJECT,
            completion_marker=operation_id,
            rating_records=(*state.rating_records, record),
        )

    def _promote_or_reject(self, state: TrainingRunState) -> TrainingRunState:
        if not state.promotion_records:
            raise CoordinatorError("promotion decision is missing")
        decision = state.promotion_records[-1]
        candidate = self._load_candidate(state)
        operation_id = _operation_id(
            RunStage.PROMOTE_OR_REJECT,
            state,
            {
                "candidate_sha256": candidate.checkpoint_sha256,
                "promotion_operation": decision["operation_id"],
                "promoted": decision["promoted"],
            },
        )
        champion = (
            str(candidate.path)
            if decision["promoted"] is True
            else state.champion_checkpoint
        )
        return self.state_store.transition(
            state,
            RunStage.PERSIST,
            completion_marker=operation_id,
            champion_checkpoint=champion,
        )

    def _persist(self, state: TrainingRunState) -> TrainingRunState:
        operation_id = _operation_id(
            RunStage.PERSIST,
            state,
            {
                "candidate_checkpoint": state.candidate_checkpoint,
                "champion_checkpoint": state.champion_checkpoint,
                "promotion_records": len(state.promotion_records),
                "rating_records": len(state.rating_records),
                "training_step": state.training_step,
            },
        )
        return self.state_store.transition(
            state,
            RunStage.COMPLETE,
            completion_marker=operation_id,
            iteration=state.iteration + 1,
        )

    def _validate_jobs(
        self, state: TrainingRunState, jobs: tuple[SelfPlayJob, ...]
    ) -> None:
        if not isinstance(jobs, tuple) or len(jobs) != self.worker_config.games_per_iteration:
            raise CoordinatorError("job builder returned the wrong number of self-play jobs")
        if not all(isinstance(job, SelfPlayJob) for job in jobs):
            raise CoordinatorError("job builder returned a malformed self-play job")
        if len({job.game_id for job in jobs}) != len(jobs):
            raise CoordinatorError("self-play jobs contain duplicate game IDs")
        for job in jobs:
            if (
                job.compatibility != self.compatibility
                or job.run_seed != state.run_seed
                or job.iteration != state.iteration
                or job.retry_id != self.worker_config.retry_id
            ):
                raise CoordinatorError("self-play job identity is incompatible with run state")

    def _validate_episodes(
        self,
        jobs: tuple[SelfPlayJob, ...],
        episodes: tuple[EpisodeResult, ...],
    ) -> None:
        if not isinstance(episodes, tuple) or len(episodes) != len(jobs):
            raise CoordinatorError("self-play artifact does not match the requested jobs")
        by_id = {episode.game_id: episode for episode in episodes}
        if len(by_id) != len(episodes) or set(by_id) != {job.game_id for job in jobs}:
            raise CoordinatorError("self-play artifact contains inconsistent game IDs")
        for job in jobs:
            episode = by_id[job.game_id]
            if (
                episode.seed != job.seed
                or episode.retry_id != job.retry_id
                or episode.model_key != job.model_key
                or episode.compatibility != job.compatibility
            ):
                raise CoordinatorError("self-play artifact identity does not match its job")

    def _validate_training_artifact(
        self,
        state: TrainingRunState,
        operation_id: str,
        artifact: TrainingStepArtifact,
    ) -> None:
        if not isinstance(artifact, TrainingStepArtifact):
            raise CoordinatorError("training stage returned a malformed artifact")
        if (
            artifact.operation_id != operation_id
            or artifact.compatibility_namespace != state.compatibility_namespace
            or artifact.input_training_step != state.training_step
            or artifact.output_training_step != state.training_step + 1
        ):
            raise CoordinatorError("training artifact identity is inconsistent")

    def _validate_candidate(
        self,
        state: TrainingRunState,
        operation_id: str,
        expected_path: Path,
        artifact: CandidateArtifact,
    ) -> None:
        if not isinstance(artifact, CandidateArtifact):
            raise CoordinatorError("checkpoint stage returned a malformed artifact")
        try:
            content = expected_path.read_bytes()
        except OSError as error:
            raise CoordinatorError("candidate checkpoint artifact is missing") from error
        digest = hashlib.sha256(content).hexdigest()
        participant = artifact.participant
        if (
            artifact.operation_id != operation_id
            or artifact.path.resolve() != expected_path.resolve()
            or not _SHA256.fullmatch(artifact.checkpoint_sha256)
            or artifact.checkpoint_sha256 != digest
            or artifact.training_step != state.training_step
            or artifact.compatibility_namespace != state.compatibility_namespace
            or participant.checkpoint_sha256 != digest
            or participant.training_step != state.training_step
            or participant.compatibility_metadata != state.compatibility
        ):
            raise CoordinatorError("candidate checkpoint identity or hash is inconsistent")

    def _load_candidate(self, state: TrainingRunState) -> CandidateArtifact:
        operation_id = self._completion(state, RunStage.SAVE_CANDIDATE)
        artifact = self.checkpoints.load(operation_id)
        if artifact is None or state.candidate_checkpoint is None:
            raise CoordinatorError("completed candidate checkpoint artifact is missing")
        self._validate_candidate(
            state,
            operation_id,
            Path(state.candidate_checkpoint),
            artifact,
        )
        return artifact

    def _validate_promotion(
        self,
        state: TrainingRunState,
        operation_id: str,
        candidate: CandidateArtifact,
        artifact: PromotionArtifact,
    ) -> None:
        if not isinstance(artifact, PromotionArtifact):
            raise CoordinatorError("promotion arena returned a malformed artifact")
        if (
            artifact.operation_id != operation_id
            or artifact.promotion_protocol_id != self.loop_config.promotion_protocol_id
            or artifact.candidate_sha256 != candidate.checkpoint_sha256
            or artifact.champion_checkpoint != state.champion_checkpoint
            or not isinstance(artifact.promoted, bool)
        ):
            raise CoordinatorError("promotion arena artifact identity is inconsistent")

    def _validate_rating_events(
        self,
        candidate: CandidateArtifact,
        events: tuple[RatingEvent, ...],
    ) -> None:
        if not isinstance(events, tuple) or not all(
            isinstance(event, (SooRatingEvent, MinRatingEvent)) for event in events
        ):
            raise CoordinatorError("rating benchmark returned malformed events")
        if len({event.event_id for event in events}) != len(events):
            raise CoordinatorError("rating benchmark returned duplicate event IDs")
        for event in events:
            if event.protocol_id != self.loop_config.benchmark_protocol_id:
                raise CoordinatorError("rating event benchmark protocol changed")
            if candidate.participant.participant_id not in event.participant_ids:
                raise CoordinatorError("rating event omits the candidate checkpoint")

    def _persist_rating_registry(self, state: TrainingRunState) -> None:
        path = self.persistence_config.rating_registry_path(state)
        if path.exists():
            try:
                persisted = RatingRegistry.load(path)
            except (OSError, ValueError) as error:
                raise CoordinatorError("persisted rating registry is inconsistent") from error
            if persisted.protocol != self.rating_registry.protocol:
                raise CoordinatorError("persisted rating registry protocol changed")
            for participant_id, participant in persisted.participants.items():
                if self.rating_registry.participants.get(participant_id) != participant:
                    raise CoordinatorError("persisted rating participant is inconsistent")
            current_events = {event.event_id: event for event in self.rating_registry.events}
            if any(current_events.get(event.event_id) != event for event in persisted.events):
                raise CoordinatorError("persisted rating event is inconsistent")
            if (
                persisted.participants == self.rating_registry.participants
                and persisted.events == self.rating_registry.events
            ):
                return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.rating_registry.save(path)

    @staticmethod
    def _completion(state: TrainingRunState, stage: RunStage) -> str:
        marker = state.stage_completions.get(stage.value)
        if not isinstance(marker, str) or not marker:
            raise CoordinatorError(f"completed {stage.value} marker is missing")
        return marker


__all__ = [
    "BenchmarkStage",
    "CandidateArtifact",
    "CheckpointStage",
    "CoordinatorError",
    "PersistenceConfig",
    "PromotionArtifact",
    "PromotionStage",
    "SelfPlayStage",
    "TrainingCoordinator",
    "TrainingLoopConfig",
    "TrainingStage",
    "TrainingStepArtifact",
    "WorkerConfig",
]
