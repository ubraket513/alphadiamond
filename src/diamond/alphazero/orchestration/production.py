"""Typed, checkpoint-backed production orchestration assembly."""

from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from ..arena import MinArena, SooArena
from ..config import (
    ArenaConfig,
    MCTSConfig,
    NetworkConfig,
    ReplayConfig,
    SelfPlayConfig,
    TrainingConfig,
)
from ..checkpoint import load_checkpoint, load_inference_checkpoint, save_checkpoint
from ..evaluator.base import EvalRequest
from ..identity import MIN_MODEL_NAME, SOO_MODEL_NAME, CheckpointCompatibilitySpec
from ..inference.coordinator import InferenceConfig
from ..inference.coordinator import InferenceCoordinator
from ..inference.model_pool import InferenceModelPool
from ..inference.protocol import ModelKey
from ..inference.remote import RemoteEvaluator
from ..network import MinModel, SooModel
from ..rating.events import MinRatingEvent, SooRatingEvent
from ..rating.openings import OpeningSuite
from ..rating.participants import CheckpointParticipant
from ..rating.protocol import BenchmarkProtocol, EloConfig, TrueSkillConfig
from ..rating.registry import RatingRegistry
from ..replay import ReplayBatch, ReplayBuffer, TrainingSample
from ..trainer import AlphaZeroTrainer
from .benchmark import ProductionBenchmarkStage
from .coordinator import (
    CandidateArtifact,
    PersistenceConfig,
    PromotionArtifact,
    TrainingCoordinator,
    TrainingLoopConfig,
    TrainingStepArtifact,
    WorkerConfig,
)
from .replay_store import PersistentReplayStore
from .run_state import RunStateStore, TrainingRunState, validate_run_id
from .selfplay_workers import EpisodeResult, SelfPlayJob, SelfPlayWorkerPool
from ...game.state import build_players, initial_state
from ..game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter

_CONFIG_VERSION = 1


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _exact_keys(payload: Mapping[str, object], expected: set[str], field: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unexpected = sorted(set(payload) - expected)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise ValueError(f"invalid {field} ({'; '.join(details)})")


def _dataclass_from_payload(cls, value: object, field: str):
    payload = _mapping(value, field)
    expected = set(cls.__dataclass_fields__)
    _exact_keys(payload, expected, field)
    try:
        return cls(**dict(payload))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field}: {error}") from error


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    opening_count: int
    opening_max_depth: int
    opening_seed: int
    opening_suite_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.opening_count, int) or isinstance(self.opening_count, bool) or self.opening_count <= 0:
            raise ValueError("opening_count must be a positive integer")
        if (
            not isinstance(self.opening_max_depth, int)
            or isinstance(self.opening_max_depth, bool)
            or self.opening_max_depth < 0
        ):
            raise ValueError("opening_max_depth must be a non-negative integer")
        if self.opening_count > 1 and self.opening_max_depth == 0:
            raise ValueError("opening_max_depth must be positive for multiple openings")
        if not isinstance(self.opening_seed, int) or isinstance(self.opening_seed, bool):
            raise ValueError("opening_seed must be an integer")
        if not isinstance(self.opening_suite_version, str) or not self.opening_suite_version:
            raise ValueError("opening_suite_version must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ProductionConfig:
    """Strict JSON configuration for one Soo or Min production run."""

    model_name: str
    model_version: str
    network: NetworkConfig
    mcts: MCTSConfig
    self_play: SelfPlayConfig
    replay: ReplayConfig
    training: TrainingConfig
    arena: ArenaConfig
    workers: WorkerConfig
    inference: InferenceConfig
    benchmark: BenchmarkConfig
    run_seed: int
    schema_version: int = _CONFIG_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _CONFIG_VERSION:
            raise ValueError(f"unsupported production config version: {self.schema_version}")
        if self.model_name not in (SOO_MODEL_NAME, MIN_MODEL_NAME):
            raise ValueError("model_name must be Soo or Min")
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("model_version must be a non-empty string")
        if not isinstance(self.run_seed, int) or isinstance(self.run_seed, bool) or self.run_seed < 0:
            raise ValueError("run_seed must be a non-negative integer")
        if self.network.width <= 0 or self.network.residual_blocks <= 0:
            raise ValueError("network dimensions must be positive")
        if self.mcts.simulations <= 0 or self.mcts.c_puct <= 0:
            raise ValueError("MCTS simulations and c_puct must be positive")
        if self.self_play.max_moves <= 0:
            raise ValueError("self_play max_moves must be positive")
        if self.replay.capacity <= 0:
            raise ValueError("replay capacity must be positive")
        if self.training.batch_size <= 0:
            raise ValueError("training batch_size must be positive")
        if self.arena.games <= 0 or self.arena.max_moves <= 0:
            raise ValueError("arena games and max_moves must be positive")
        balance_cycle = 4 if self.model_name == SOO_MODEL_NAME else 18
        if self.arena.games % balance_cycle != 0:
            raise ValueError(
                f"{self.model_name} arena games must be a multiple of {balance_cycle}"
            )
        if self.training.batch_size > self.replay.capacity:
            raise ValueError("training batch_size exceeds replay capacity")

    @property
    def compatibility(self) -> CheckpointCompatibilitySpec:
        factory = (
            CheckpointCompatibilitySpec.soo
            if self.model_name == SOO_MODEL_NAME
            else CheckpointCompatibilitySpec.min
        )
        return factory(model_version=self.model_version, network_config=self.network)

    @classmethod
    def from_payload(cls, value: object) -> ProductionConfig:
        payload = _mapping(value, "production config")
        expected = {
            "schema_version",
            "model_name",
            "model_version",
            "network",
            "mcts",
            "self_play",
            "replay",
            "training",
            "arena",
            "workers",
            "inference",
            "benchmark",
            "run_seed",
        }
        _exact_keys(payload, expected, "production config")
        try:
            return cls(
                schema_version=payload["schema_version"],  # type: ignore[arg-type]
                model_name=payload["model_name"],  # type: ignore[arg-type]
                model_version=payload["model_version"],  # type: ignore[arg-type]
                network=_dataclass_from_payload(NetworkConfig, payload["network"], "network"),
                mcts=_dataclass_from_payload(MCTSConfig, payload["mcts"], "mcts"),
                self_play=_dataclass_from_payload(
                    SelfPlayConfig, payload["self_play"], "self_play"
                ),
                replay=_dataclass_from_payload(ReplayConfig, payload["replay"], "replay"),
                training=_dataclass_from_payload(
                    TrainingConfig, payload["training"], "training"
                ),
                arena=_dataclass_from_payload(ArenaConfig, payload["arena"], "arena"),
                workers=_dataclass_from_payload(WorkerConfig, payload["workers"], "workers"),
                inference=_dataclass_from_payload(
                    InferenceConfig, payload["inference"], "inference"
                ),
                benchmark=_dataclass_from_payload(
                    BenchmarkConfig, payload["benchmark"], "benchmark"
                ),
                run_seed=payload["run_seed"],  # type: ignore[arg-type]
            )
        except KeyError as error:
            raise ValueError(f"invalid production config: missing {error.args[0]}") from error

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "network": asdict(self.network),
            "mcts": asdict(self.mcts),
            "self_play": asdict(self.self_play),
            "replay": asdict(self.replay),
            "training": asdict(self.training),
            "arena": asdict(self.arena),
            "workers": asdict(self.workers),
            "inference": asdict(self.inference),
            "benchmark": asdict(self.benchmark),
            "run_seed": self.run_seed,
        }


def load_production_config(path: str | Path) -> ProductionConfig:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read production config {source}: {error}") from error
    return ProductionConfig.from_payload(payload)


class ProductionArtifactStore:
    """Run-local idempotent JSON journals for production stage artifacts."""

    def __init__(self, root: str | Path) -> None:
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

    def write(
        self, stage: str, operation_id: str, payload: Mapping[str, object]
    ) -> Path:
        destination = self.path(stage, operation_id)
        body = {"format_version": 1, "operation_id": operation_id, **dict(payload)}
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_text(encoding="utf-8") != encoded:
                raise ValueError(f"conflicting {stage} artifact")
            return destination
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(destination)
        return destination


def _new_model(compatibility: CheckpointCompatibilitySpec):
    if compatibility.identity.model_name == SOO_MODEL_NAME:
        return SooModel(
            compatibility.network_config,
            model_version=compatibility.identity.model_version,
        )
    return MinModel(
        compatibility.network_config,
        model_version=compatibility.identity.model_version,
    )


class ProductionCheckpointStage:
    """Publish and strictly reconstruct deterministic candidate operations."""

    def __init__(
        self,
        *,
        trainer: AlphaZeroTrainer,
        artifacts: ProductionArtifactStore,
        compatibility: CheckpointCompatibilitySpec,
        persistence_root: str | Path,
    ) -> None:
        if trainer.compatibility != compatibility:
            raise ValueError("checkpoint trainer compatibility changed")
        self.trainer = trainer
        self.artifacts = artifacts
        self.compatibility = compatibility
        self.persistence_root = Path(persistence_root).resolve(strict=False)

    def _contained(self, path: str | Path) -> Path:
        resolved = Path(path).resolve(strict=False)
        try:
            resolved.relative_to(self.persistence_root)
        except ValueError as error:
            raise ValueError("candidate checkpoint path escapes persistence root") from error
        return resolved

    def load(self, operation_id: str) -> CandidateArtifact | None:
        payload = self.artifacts.read("candidate", operation_id)
        if payload is None:
            return None
        try:
            path = Path(payload["path"])
            self._contained(path)
            participant = CheckpointParticipant.from_checkpoint(path)
            return CandidateArtifact(
                operation_id=operation_id,
                path=path,
                checkpoint_sha256=payload["checkpoint_sha256"],  # type: ignore[arg-type]
                training_step=payload["training_step"],  # type: ignore[arg-type]
                compatibility_namespace=payload["compatibility_namespace"],  # type: ignore[arg-type]
                participant=participant,
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise ValueError(f"invalid candidate artifact: {error}") from error

    def execute(
        self,
        operation_id: str,
        path: Path,
        training: TrainingStepArtifact,
    ) -> CandidateArtifact:
        self._contained(path)
        if self.trainer.training_step != training.output_training_step:
            raise ValueError("candidate trainer step does not match training artifact")
        if path.exists():
            raise ValueError("candidate checkpoint already exists")
        save_checkpoint(path, self.trainer, operation_id=operation_id)
        artifact = self._validated_artifact(operation_id, path, training)
        self._write_journal(artifact)
        return artifact

    def recover(
        self,
        operation_id: str,
        path: Path,
        training: TrainingStepArtifact,
    ) -> CandidateArtifact:
        """Validate an unjournaled publication without mutating the live trainer."""
        self._contained(path)
        try:
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError) as error:
            raise ValueError(f"cannot recover candidate checkpoint: {error}") from error
        if not isinstance(payload, Mapping):
            raise ValueError("candidate checkpoint root must be a mapping")
        if payload.get("operation_id") != operation_id:
            raise ValueError("candidate checkpoint operation identity does not match")
        artifact = self._validated_artifact(operation_id, path, training)
        isolated = AlphaZeroTrainer(
            _new_model(self.compatibility), self.compatibility, self.trainer.config
        )
        load_checkpoint(path, isolated, expected=self.compatibility)
        if isolated.training_step != training.output_training_step:
            raise ValueError("candidate checkpoint training step does not match")
        self._write_journal(artifact)
        return artifact

    def _validated_artifact(
        self,
        operation_id: str,
        path: Path,
        training: TrainingStepArtifact,
    ) -> CandidateArtifact:
        participant = CheckpointParticipant.from_checkpoint(path)
        self.compatibility.assert_compatible(participant.compatibility_metadata)
        if participant.training_step != training.output_training_step:
            raise ValueError("candidate checkpoint training step does not match")
        validation_model = _new_model(self.compatibility)
        info = load_inference_checkpoint(
            path, validation_model, expected=self.compatibility, device="cpu"
        )
        if info.checkpoint_sha256 != participant.checkpoint_sha256:
            raise ValueError("candidate checkpoint hash changed during validation")
        return CandidateArtifact(
            operation_id=operation_id,
            path=path,
            checkpoint_sha256=participant.checkpoint_sha256,
            training_step=participant.training_step,
            compatibility_namespace=training.compatibility_namespace,
            participant=participant,
        )

    def _write_journal(self, artifact: CandidateArtifact) -> None:
        self.artifacts.write(
            "candidate",
            artifact.operation_id,
            {
                "checkpoint_sha256": artifact.checkpoint_sha256,
                "compatibility_namespace": artifact.compatibility_namespace,
                "path": str(artifact.path),
                "training_step": artifact.training_step,
            },
        )


def build_authoritative_selfplay_jobs(
    state: TrainingRunState,
    workers: WorkerConfig,
    config: ProductionConfig,
) -> tuple[SelfPlayJob, ...]:
    """Build standard authoritative initial-state jobs pinned to the champion."""
    if state.champion_model_key is None or state.champion_checkpoint is None:
        raise ValueError("training run has no durable champion checkpoint identity")
    if state.champion_model_key.model_name != config.model_name:
        raise ValueError("durable champion model does not match production config")
    players = build_players(config.compatibility.identity.player_count)
    start = initial_state(players)
    return tuple(
        SelfPlayJob(
            run_seed=state.run_seed,
            iteration=state.iteration,
            game_index=index,
            retry_id=workers.retry_id,
            model_key=state.champion_model_key,
            compatibility=config.compatibility,
            players=players,
            initial_state=start,
            mcts_config=config.mcts,
            selfplay_config=config.self_play,
        )
        for index in range(workers.games_per_iteration)
    )


def _sample_payload(sample: TrainingSample) -> dict[str, object]:
    return {
        "canonical_player_ids": sample.canonical_player_ids,
        "node_features": sample.node_features,
        "schema_version": sample.schema_version,
        "sparse_policy": sample.sparse_policy,
        "value_target": sample.value_target,
    }


def _sample_from_payload(
    value: object, compatibility: CheckpointCompatibilitySpec
) -> TrainingSample:
    if not isinstance(value, Mapping):
        raise ValueError("self-play sample must be a JSON object")
    try:
        return TrainingSample(
            compatibility=compatibility,
            node_features=tuple(tuple(row) for row in value["node_features"]),
            canonical_player_ids=tuple(value["canonical_player_ids"]),
            sparse_policy=tuple(tuple(row) for row in value["sparse_policy"]),
            value_target=tuple(value["value_target"]),
            schema_version=value["schema_version"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid self-play sample: {error}") from error


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
    value: object, compatibility: CheckpointCompatibilitySpec
) -> EpisodeResult:
    if not isinstance(value, Mapping):
        raise ValueError("self-play episode must be a JSON object")
    try:
        final_order = value["final_order"]
        return EpisodeResult(
            game_id=value["game_id"],
            seed=value["seed"],
            retry_id=value["retry_id"],
            model_key=ModelKey.from_payload(value["model_key"]),
            compatibility=compatibility,
            samples=tuple(
                _sample_from_payload(sample, compatibility)
                for sample in value["samples"]
            ),
            final_order=tuple(final_order) if final_order is not None else None,
            move_count=value["move_count"],
            completed=value["completed"],
            aborted_reason=value["aborted_reason"],
            worker_id=value["worker_id"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid self-play episode: {error}") from error


def _compatibility_namespace(compatibility: CheckpointCompatibilitySpec) -> str:
    encoded = json.dumps(
        compatibility.to_metadata(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class ProductionSelfPlayStage:
    def __init__(
        self,
        *,
        model_pool: InferenceModelPool,
        inference_config: InferenceConfig,
        worker_count: int,
        artifacts: ProductionArtifactStore,
        compatibility: CheckpointCompatibilitySpec,
    ) -> None:
        self.model_pool = model_pool
        self.inference_config = inference_config
        self.worker_count = worker_count
        self.artifacts = artifacts
        self.compatibility = compatibility

    def load(self, operation_id: str) -> tuple[EpisodeResult, ...] | None:
        payload = self.artifacts.read("self_play", operation_id)
        if payload is None:
            return None
        episodes = payload.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError("invalid self_play artifact episodes")
        return tuple(
            _episode_from_payload(episode, self.compatibility) for episode in episodes
        )

    def execute(
        self, operation_id: str, jobs: tuple[SelfPlayJob, ...]
    ) -> tuple[EpisodeResult, ...]:
        for job in jobs:
            self.model_pool.evaluator(job.model_key)
        coordinator = InferenceCoordinator(self.model_pool, self.inference_config)
        coordinator.start()
        try:
            episodes = SelfPlayWorkerPool(
                coordinator,
                worker_count=self.worker_count,
                worker_timeout_s=max(60.0, self.inference_config.response_timeout_s * 4),
            ).run(jobs)
        finally:
            coordinator.stop()
        self.artifacts.write(
            "self_play",
            operation_id,
            {"episodes": [_episode_payload(episode) for episode in episodes]},
        )
        return episodes


class ProductionTrainingStage:
    def __init__(
        self, trainer: AlphaZeroTrainer, artifacts: ProductionArtifactStore
    ) -> None:
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
        metrics = self.trainer.train_batch(
            replay.load_buffer().collate(samples, action_size=73 * 73)
        )
        artifact = TrainingStepArtifact(
            operation_id=operation_id,
            compatibility_namespace=_compatibility_namespace(
                self.trainer.compatibility
            ),
            input_training_step=expected_training_step,
            output_training_step=self.trainer.training_step,
            metrics=asdict(metrics),
        )
        save_checkpoint(
            self.artifacts.path("training", operation_id, ".pt"),
            self.trainer,
            operation_id=operation_id,
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


class ProductionPromotionStage:
    def __init__(
        self,
        *,
        compatibility: CheckpointCompatibilitySpec,
        protocol_id: str,
        config: ProductionConfig,
        model_pool: InferenceModelPool,
        artifacts: ProductionArtifactStore,
    ) -> None:
        self._compatibility = compatibility
        self._protocol_id = protocol_id
        self.config = config
        self.model_pool = model_pool
        self.artifacts = artifacts

    @property
    def compatibility(self) -> CheckpointCompatibilitySpec:
        return self._compatibility

    @property
    def protocol_id(self) -> str:
        return self._protocol_id

    def load(self, operation_id: str) -> PromotionArtifact | None:
        payload = self.artifacts.read("promotion", operation_id)
        if payload is None:
            return None
        result = payload.get("result")
        if not isinstance(result, Mapping):
            raise ValueError("invalid promotion artifact result")
        return PromotionArtifact(
            operation_id=operation_id,
            promotion_protocol_id=payload["promotion_protocol_id"],  # type: ignore[arg-type]
            candidate_sha256=payload["candidate_sha256"],  # type: ignore[arg-type]
            champion_checkpoint=payload["champion_checkpoint"],  # type: ignore[arg-type]
            promoted=payload["promoted"],  # type: ignore[arg-type]
            result=dict(result),
        )

    def execute(
        self,
        operation_id: str,
        candidate: CandidateArtifact,
        champion_checkpoint: str | None,
    ) -> PromotionArtifact:
        if champion_checkpoint is None:
            raise ValueError("promotion arena requires a champion checkpoint")
        candidate_key = self.model_pool.activate_checkpoint(
            candidate.path, expected=self.compatibility
        )
        champion_key = self.model_pool.activate_checkpoint(
            champion_checkpoint, expected=self.compatibility
        )
        coordinator = InferenceCoordinator(self.model_pool, self.config.inference)
        candidate_evaluator = RemoteEvaluator(
            coordinator,
            model_key=candidate_key,
            client_id=f"promotion-candidate-{candidate_key.checkpoint_sha256[:12]}",
        )
        champion_evaluator = RemoteEvaluator(
            coordinator,
            model_key=champion_key,
            client_id=f"promotion-champion-{champion_key.checkpoint_sha256[:12]}",
        )

        def game_factory(order: tuple[int, ...]) -> DiamondSearchAdapter:
            return DiamondSearchAdapter(AlphaZeroGameAdapter(build_players(len(order), order=order)))

        coordinator.start()
        try:
            if self.compatibility.identity.model_name == SOO_MODEL_NAME:
                result = SooArena(
                    candidate=candidate_evaluator,
                    baseline=champion_evaluator,
                    mcts_config=self.config.mcts,
                    arena_config=self.config.arena,
                ).run(game_factory)
            else:
                result = MinArena(
                    candidate=candidate_evaluator,
                    baseline=champion_evaluator,
                    mcts_config=self.config.mcts,
                    arena_config=self.config.arena,
                ).run(game_factory)
        finally:
            coordinator.stop()
        artifact = PromotionArtifact(
            operation_id=operation_id,
            promotion_protocol_id=self.protocol_id,
            candidate_sha256=candidate.checkpoint_sha256,
            champion_checkpoint=champion_checkpoint,
            promoted=result.promoted,
            result=asdict(result),
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


class ProductionTrainingServices:
    """Preflighted production services; runtime assembly stays command-scoped."""

    def __init__(
        self,
        root: Path,
        model_name: str,
        config_path: Path,
        checkpoint_path: Path,
    ) -> None:
        self.root = Path(root)
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.config = load_production_config(self.config_path)
        if self.config.model_name != model_name:
            raise ValueError("command model does not match production config model_name")
        self.model_name = model_name
        self.compatibility = self.config.compatibility
        preflight_trainer = AlphaZeroTrainer(
            _new_model(self.compatibility), self.compatibility, self.config.training
        )
        checkpoint_info = load_checkpoint(
            self.checkpoint_path,
            preflight_trainer,
            expected=self.compatibility,
        )
        if checkpoint_info.training_config != self.config.training:
            raise ValueError("checkpoint training_config does not match production config")
        self.checkpoint_participant = CheckpointParticipant.from_checkpoint(
            self.checkpoint_path
        )
        self.model_pool = InferenceModelPool(device=self.config.training.device)
        self.checkpoint_key = self.model_pool.activate_checkpoint(
            self.checkpoint_path, expected=self.compatibility
        )
        if self.checkpoint_key.checkpoint_sha256 != self.checkpoint_participant.checkpoint_sha256:
            raise ValueError("checkpoint identity changed during production preflight")
        self.opening_suite = OpeningSuite.generate(
            player_count=self.compatibility.identity.player_count,
            seed=self.config.benchmark.opening_seed,
            opening_count=self.config.benchmark.opening_count,
            max_depth=self.config.benchmark.opening_max_depth,
            version=self.config.benchmark.opening_suite_version,
        )
        rating_config = (
            EloConfig() if self.model_name == SOO_MODEL_NAME else TrueSkillConfig()
        )
        self.protocol = BenchmarkProtocol(
            compatibility=self.compatibility,
            simulations=self.config.mcts.simulations,
            c_puct=self.config.mcts.c_puct,
            dirichlet_epsilon=0.0,
            decision_temperature=0.0,
            max_game_moves=self.config.arena.max_moves,
            opening_suite_version=self.opening_suite.version,
            opening_suite_hash=self.opening_suite.suite_hash,
            rating_system_version=rating_config.rating_system_version,
            rating_parameters=asdict(rating_config),
        )
        promotion_payload = json.dumps(
            {
                "arena": asdict(self.config.arena),
                "compatibility": self.compatibility.to_metadata(),
                "mcts": asdict(self.config.mcts),
                "namespace": "production-promotion-v1",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.promotion_protocol_id = (
            f"sha256:{hashlib.sha256(promotion_payload).hexdigest()}"
        )

    def train(self, *, model_name: str, run_id: str) -> Mapping[str, object]:
        self._assert_command(model_name, run_id)
        state_store = RunStateStore(self.root)
        state_path = state_store.state_path(run_id, model_name)
        if state_path.exists():
            raise ValueError(f"training run already exists: {run_id}")
        self._write_run_config(run_id)
        initial = state_store.initialize(
            run_id=run_id,
            compatibility=self.compatibility,
            run_seed=self.config.run_seed,
            protocol_ids={
                "promotion": self.promotion_protocol_id,
                "rating": self.protocol.protocol_id,
            },
            champion_checkpoint=str(self.checkpoint_path.resolve(strict=True)),
            champion_model_key=self.checkpoint_key,
            training_step=self.checkpoint_participant.training_step,
        )
        coordinator, benchmark = self._coordinator(initial)
        complete = coordinator.run_iteration(initial)
        return self._summary(complete, benchmark)

    def resume(self, *, model_name: str, run_id: str) -> Mapping[str, object]:
        self._assert_command(model_name, run_id)
        state = RunStateStore(self.root).load(run_id, model_name)
        self._assert_state(state, supplied_checkpoint_must_be_champion=True)
        coordinator, benchmark = self._coordinator(state)
        complete = coordinator.resume(run_id, model_name)
        return self._summary(complete, benchmark)

    def benchmark(self, *, model_name: str, run_id: str) -> Mapping[str, object]:
        self._assert_command(model_name, run_id)
        state = RunStateStore(self.root).load(run_id, model_name)
        self._assert_state(state, supplied_checkpoint_must_be_champion=False)
        registry = self._load_registry(run_id)
        stage = self._benchmark_stage(run_id, registry)
        participant = self.checkpoint_participant
        candidate = CandidateArtifact(
            operation_id="standalone-benchmark-candidate",
            path=self.checkpoint_path,
            checkpoint_sha256=participant.checkpoint_sha256,
            training_step=participant.training_step,
            compatibility_namespace=state.compatibility_namespace,
            participant=participant,
        )
        operation_payload = json.dumps(
            {
                "candidate_sha256": participant.checkpoint_sha256,
                "champion_checkpoint": state.champion_checkpoint,
                "protocol_id": self.protocol.protocol_id,
                "run_id": run_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        operation_id = f"sha256:{hashlib.sha256(operation_payload).hexdigest()}"
        events = stage.load(operation_id)
        if events is None:
            events = stage.execute(
                operation_id,
                candidate,
                state.champion_checkpoint,
            )
        return {
            "events": len(events),
            "event_ids": [event.event_id for event in events],
            "rating_status": stage.status,
        }

    def leaderboard(self, *, model_name: str, run_id: str) -> Mapping[str, object]:
        self._assert_command(model_name, run_id)
        registry = self._load_registry(run_id)
        entries = (
            registry.soo_leaderboard()
            if model_name == SOO_MODEL_NAME
            else registry.min_leaderboard()
        )
        return {"entries": [asdict(entry) for entry in entries]}

    def profile(self, *, model_name: str, max_seconds: int) -> object:
        if model_name != self.model_name:
            raise ValueError("command model does not match production config")
        from ..inference.profile import profile_evaluator

        request = EvalRequest(
            node_features=tuple(
                (0.0,) * (self.compatibility.identity.player_count * 2)
                for _ in range(73)
            ),
            legal_action_ids=(0, 1, 2),
            canonical_player_ids=tuple(
                range(1, self.compatibility.identity.player_count + 1)
            ),
        )
        return profile_evaluator(
            self.model_pool.evaluator(self.checkpoint_key),
            (request,),
            max_seconds=max_seconds,
            model_key=self.checkpoint_key,
            stage_operations=self._profile_stage_operations(),
        )

    def _assert_command(self, model_name: str, run_id: str) -> None:
        if model_name != self.model_name:
            raise ValueError("command model does not match production config")
        validate_run_id(run_id)
        root = self.root.resolve(strict=False)
        run_root = self._run_root(run_id).resolve(strict=False)
        try:
            run_root.relative_to(root)
        except ValueError as error:
            raise ValueError("run_id escapes runtime root") from error

    def _assert_state(
        self,
        state: TrainingRunState,
        *,
        supplied_checkpoint_must_be_champion: bool,
    ) -> None:
        if dict(state.compatibility) != self.compatibility.to_metadata():
            raise ValueError("training run compatibility does not match production config")
        if dict(state.protocol_ids) != {
            "promotion": self.promotion_protocol_id,
            "rating": self.protocol.protocol_id,
        }:
            raise ValueError("training run protocol identity does not match production config")
        if state.champion_checkpoint is None or state.champion_model_key is None:
            raise ValueError("training run has no durable champion identity")
        champion_path = Path(state.champion_checkpoint)
        champion_pool = InferenceModelPool(device=self.config.training.device)
        champion_key = champion_pool.activate_checkpoint(
            champion_path, expected=self.compatibility
        )
        if champion_key != state.champion_model_key:
            raise ValueError("durable champion checkpoint hash does not match run state")
        if supplied_checkpoint_must_be_champion and (
            self.checkpoint_key != state.champion_model_key
            or self.checkpoint_path.resolve(strict=True)
            != champion_path.resolve(strict=True)
        ):
            raise ValueError("supplied checkpoint is not the exact durable champion")
        if state.replay_manifest is not None and not Path(state.replay_manifest).exists():
            raise ValueError("durable replay manifest is missing")

    def _run_root(self, run_id: str) -> Path:
        return self.root / self.model_name.lower() / run_id

    def _write_run_config(self, run_id: str) -> None:
        path = self._run_root(run_id) / "config.json"
        encoded = json.dumps(
            self.config.to_payload(), sort_keys=True, separators=(",", ":")
        )
        if path.exists():
            if path.read_text(encoding="utf-8") != encoded:
                raise ValueError("persisted production config changed")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)

    def _load_registry(self, run_id: str) -> RatingRegistry:
        path = self._run_root(run_id) / "ratings" / "registry.json"
        if not path.exists():
            raise ValueError(f"rating registry does not exist: {run_id} ({self.model_name})")
        registry = RatingRegistry.load(path)
        if registry.protocol != self.protocol:
            raise ValueError("persisted rating registry protocol changed")
        return registry

    def _checkpoint_history(self, run_id: str) -> tuple[Path, ...]:
        paths = [self.checkpoint_path]
        state = RunStateStore(self.root).load(run_id, self.model_name)
        if state.champion_checkpoint is not None:
            paths.append(Path(state.champion_checkpoint))
        paths.extend(sorted((self._run_root(run_id) / "checkpoints").glob("*.pt")))
        return tuple(paths)

    def _benchmark_stage(
        self, run_id: str, registry: RatingRegistry
    ) -> ProductionBenchmarkStage:
        run_root = self._run_root(run_id)
        return ProductionBenchmarkStage(
            protocol=self.protocol,
            opening_suite=self.opening_suite,
            checkpoint_paths=lambda: self._checkpoint_history(run_id),
            artifacts_root=run_root / "artifacts" / "benchmark",
            registry=registry,
            registry_path=run_root / "ratings" / "registry.json",
            model_pool=self.model_pool,
            inference_config=self.config.inference,
        )

    def _coordinator(
        self, state: TrainingRunState
    ) -> tuple[TrainingCoordinator, ProductionBenchmarkStage]:
        run_root = self._run_root(state.run_id)
        trainer = AlphaZeroTrainer(
            _new_model(self.compatibility), self.compatibility, self.config.training
        )
        checkpoint_info = load_checkpoint(
            self.checkpoint_path, trainer, expected=self.compatibility
        )
        if checkpoint_info.training_config != self.config.training:
            raise ValueError("checkpoint training_config does not match production config")
        replay = PersistentReplayStore(
            run_root / "replay",
            self.compatibility,
            capacity=self.config.replay.capacity,
            seed=self.config.replay.seed,
        )
        artifacts = ProductionArtifactStore(run_root / "artifacts")
        registry_path = run_root / "ratings" / "registry.json"
        registry = (
            RatingRegistry.load(registry_path)
            if registry_path.exists()
            else RatingRegistry(self.protocol)
        )
        if registry.protocol != self.protocol:
            raise ValueError("persisted rating registry protocol changed")
        champion = CheckpointParticipant.from_checkpoint(state.champion_checkpoint)  # type: ignore[arg-type]
        registry.add_participant(champion)
        benchmark = self._benchmark_stage(state.run_id, registry)
        coordinator = TrainingCoordinator(
            state_store=RunStateStore(self.root),
            compatibility=self.compatibility,
            worker_config=self.config.workers,
            loop_config=TrainingLoopConfig(
                replay_batch_size=self.config.training.batch_size,
                promotion_protocol_id=self.promotion_protocol_id,
                benchmark_protocol_id=self.protocol.protocol_id,
            ),
            persistence_config=PersistenceConfig(self.root),
            build_selfplay_jobs=lambda current, workers: build_authoritative_selfplay_jobs(
                current, workers, self.config
            ),
            self_play=ProductionSelfPlayStage(
                model_pool=self.model_pool,
                inference_config=self.config.inference,
                worker_count=self.config.workers.worker_count,
                artifacts=artifacts,
                compatibility=self.compatibility,
            ),
            replay_store=replay,
            training=ProductionTrainingStage(trainer, artifacts),
            checkpoints=ProductionCheckpointStage(
                trainer=trainer,
                artifacts=artifacts,
                compatibility=self.compatibility,
                persistence_root=self.root,
            ),
            promotion=ProductionPromotionStage(
                compatibility=self.compatibility,
                protocol_id=self.promotion_protocol_id,
                config=self.config,
                model_pool=self.model_pool,
                artifacts=artifacts,
            ),
            benchmark=benchmark,
            rating_registry=registry,
        )
        return coordinator, benchmark

    def _summary(
        self,
        state: TrainingRunState,
        benchmark: ProductionBenchmarkStage,
    ) -> dict[str, object]:
        event_ids = (
            list(state.rating_records[-1]["event_ids"])
            if state.rating_records
            else []
        )
        return {
            "candidate_checkpoint": state.candidate_checkpoint,
            "champion_checkpoint": state.champion_checkpoint,
            "champion_model_key": (
                state.champion_model_key.to_payload()
                if state.champion_model_key is not None
                else None
            ),
            "rating_events": len(event_ids),
            "rating_status": benchmark.status,
            "stage": state.stage.value,
            "training_step": state.training_step,
        }

    def _profile_stage_operations(self):
        players = build_players(self.compatibility.identity.player_count)
        job = SelfPlayJob(
            run_seed=self.config.run_seed,
            iteration=0,
            game_index=0,
            retry_id="profile",
            model_key=self.checkpoint_key,
            compatibility=self.compatibility,
            players=players,
            initial_state=initial_state(players),
            mcts_config=self.config.mcts,
            selfplay_config=self.config.self_play,
        )
        replay = ReplayBuffer(
            self.compatibility,
            capacity=self.config.replay.capacity,
            seed=self.config.replay.seed,
        )
        trainer = AlphaZeroTrainer(
            _new_model(self.compatibility), self.compatibility, self.config.training
        )
        load_checkpoint(self.checkpoint_path, trainer, expected=self.compatibility)
        episodes: tuple[EpisodeResult, ...] | None = None
        batch: ReplayBatch | None = None

        def self_play() -> tuple[EpisodeResult, ...]:
            nonlocal episodes
            coordinator = InferenceCoordinator(self.model_pool, self.config.inference)
            coordinator.start()
            try:
                episodes = SelfPlayWorkerPool(coordinator, worker_count=1).run((job,))
            finally:
                coordinator.stop()
            return episodes

        def replay_collation() -> ReplayBatch:
            nonlocal batch
            if not episodes:
                raise RuntimeError("profile self-play produced no episode")
            samples = tuple(sample for episode in episodes for sample in episode.samples)
            if not samples:
                raise RuntimeError("profile self-play produced no completed samples")
            replay.extend(samples)
            batch = replay.collate((replay.samples[0],), action_size=73 * 73)
            return batch

        def training():
            if batch is None:
                raise RuntimeError("profile replay collation did not complete")
            return trainer.train_batch(batch)

        return {
            "self_play": self_play,
            "replay_collation": replay_collation,
            "training": training,
        }


def build_production_services(
    root: Path,
    model_name: str,
    config_path: Path,
    checkpoint_path: Path,
) -> ProductionTrainingServices:
    return ProductionTrainingServices(root, model_name, config_path, checkpoint_path)


__all__ = [
    "BenchmarkConfig",
    "ProductionConfig",
    "ProductionArtifactStore",
    "ProductionCheckpointStage",
    "ProductionTrainingServices",
    "build_production_services",
    "build_authoritative_selfplay_jobs",
    "load_production_config",
]
