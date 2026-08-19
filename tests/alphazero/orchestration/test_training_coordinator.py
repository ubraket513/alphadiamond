from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from diamond.alphazero.config import MCTSConfig, NetworkConfig, SelfPlayConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.inference.protocol import ModelKey
from diamond.alphazero.orchestration.coordinator import (
    CandidateArtifact,
    PersistenceConfig,
    PromotionArtifact,
    TrainingCoordinator,
    TrainingLoopConfig,
    TrainingStepArtifact,
    WorkerConfig,
)
from diamond.alphazero.orchestration.replay_store import PersistentReplayStore
from diamond.alphazero.orchestration.run_state import RunStage, RunStateStore
from diamond.alphazero.orchestration.selfplay_workers import EpisodeResult, SelfPlayJob
from diamond.alphazero.rating.events import SooRatingEvent
from diamond.alphazero.rating.participants import CheckpointParticipant
from diamond.alphazero.rating.protocol import BenchmarkProtocol, EloConfig
from diamond.alphazero.rating.registry import RatingRegistry
from diamond.alphazero.replay import TrainingSample
from diamond.game.state import build_players, initial_state


def _compatibility() -> CheckpointCompatibilitySpec:
    return CheckpointCompatibilitySpec.soo(
        model_version="2.0.0",
        network_config=NetworkConfig(width=8, residual_blocks=1),
    )


def _protocol(compatibility: CheckpointCompatibilitySpec) -> BenchmarkProtocol:
    return BenchmarkProtocol(
        compatibility=compatibility,
        simulations=1,
        c_puct=1.5,
        dirichlet_epsilon=0.0,
        decision_temperature=0.0,
        max_game_moves=2,
        opening_suite_version="tiny-openings-v1",
        opening_suite_hash="sha256:tiny-openings",
        rating_system_version="soo-elo-v1",
        rating_parameters=asdict(EloConfig()),
    )


def _participant(
    compatibility: CheckpointCompatibilitySpec,
    *,
    training_step: int,
    checkpoint_sha256: str,
) -> CheckpointParticipant:
    identity = compatibility.identity
    return CheckpointParticipant(
        participant_id=(
            f"checkpoint:{identity.model_name}:{identity.model_version}:"
            f"{training_step}:{checkpoint_sha256}"
        ),
        model_name=identity.model_name,
        model_version=identity.model_version,
        training_step=training_step,
        checkpoint_sha256=checkpoint_sha256,
        compatibility_metadata=compatibility.to_metadata(),
        display_name=f"{identity.model_name} {identity.model_version} @ {training_step}",
    )


def _job(compatibility: CheckpointCompatibilitySpec) -> SelfPlayJob:
    players = build_players(2)
    return SelfPlayJob(
        run_seed=91,
        iteration=0,
        game_index=0,
        retry_id="attempt-0",
        model_key=ModelKey("Soo", "2.0.0", "a" * 64),
        compatibility=compatibility,
        players=players,
        initial_state=initial_state(players),
        mcts_config=MCTSConfig(simulations=1, dirichlet_epsilon=0.0),
        selfplay_config=SelfPlayConfig(max_moves=2, temperature_moves=0),
    )


def _episode(job: SelfPlayJob) -> EpisodeResult:
    sample = TrainingSample(
        compatibility=job.compatibility,
        node_features=((1.0, 0.0, 0.0, 1.0),) * 73,
        canonical_player_ids=(1, 2),
        sparse_policy=((0, 1.0),),
        value_target=(1.0,),
    )
    return EpisodeResult(
        game_id=job.game_id,
        seed=job.seed,
        retry_id=job.retry_id,
        model_key=job.model_key,
        compatibility=job.compatibility,
        samples=(sample,),
        final_order=(1, 2),
        move_count=1,
        completed=True,
    )


class _SelfPlayStage:
    def __init__(self, order: list[str], episode: EpisodeResult) -> None:
        self.order = order
        self.episode = episode
        self.artifacts: dict[str, tuple[EpisodeResult, ...]] = {}

    def load(self, operation_id: str) -> tuple[EpisodeResult, ...] | None:
        return self.artifacts.get(operation_id)

    def execute(
        self, operation_id: str, jobs: tuple[SelfPlayJob, ...]
    ) -> tuple[EpisodeResult, ...]:
        self.order.append("self_play")
        result = (self.episode,)
        self.artifacts[operation_id] = result
        return result


class _TrainingStage:
    def __init__(self, order: list[str], compatibility_namespace: str) -> None:
        self.order = order
        self.compatibility_namespace = compatibility_namespace
        self.artifacts: dict[str, TrainingStepArtifact] = {}

    def load(self, operation_id: str) -> TrainingStepArtifact | None:
        return self.artifacts.get(operation_id)

    def execute(
        self,
        operation_id: str,
        replay: PersistentReplayStore,
        batch_size: int,
        expected_training_step: int,
    ) -> TrainingStepArtifact:
        self.order.append("train")
        assert len(replay.load_buffer()) == 1
        artifact = TrainingStepArtifact(
            operation_id=operation_id,
            compatibility_namespace=self.compatibility_namespace,
            input_training_step=expected_training_step,
            output_training_step=expected_training_step + 1,
            metrics={"total_loss": 0.25},
        )
        self.artifacts[operation_id] = artifact
        return artifact


class _CheckpointStage:
    def __init__(
        self,
        order: list[str],
        compatibility: CheckpointCompatibilitySpec,
    ) -> None:
        self.order = order
        self.compatibility = compatibility
        self.artifacts: dict[str, CandidateArtifact] = {}

    def load(self, operation_id: str) -> CandidateArtifact | None:
        return self.artifacts.get(operation_id)

    def execute(
        self,
        operation_id: str,
        path: Path,
        training: TrainingStepArtifact,
    ) -> CandidateArtifact:
        self.order.append("candidate")
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"candidate:{training.output_training_step}".encode()
        path.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        artifact = CandidateArtifact(
            operation_id=operation_id,
            path=path,
            checkpoint_sha256=digest,
            training_step=training.output_training_step,
            compatibility_namespace=training.compatibility_namespace,
            participant=_participant(
                self.compatibility,
                training_step=training.output_training_step,
                checkpoint_sha256=digest,
            ),
        )
        self.artifacts[operation_id] = artifact
        return artifact


class _PromotionStage:
    def __init__(self, order: list[str], protocol_id: str) -> None:
        self.order = order
        self.protocol_id = protocol_id
        self.artifacts: dict[str, PromotionArtifact] = {}

    def load(self, operation_id: str) -> PromotionArtifact | None:
        return self.artifacts.get(operation_id)

    def execute(
        self,
        operation_id: str,
        candidate: CandidateArtifact,
        champion_checkpoint: str | None,
    ) -> PromotionArtifact:
        self.order.append("arena")
        artifact = PromotionArtifact(
            operation_id=operation_id,
            promotion_protocol_id=self.protocol_id,
            candidate_sha256=candidate.checkpoint_sha256,
            champion_checkpoint=champion_checkpoint,
            promoted=True,
            result={"wins": 4, "losses": 0},
        )
        self.artifacts[operation_id] = artifact
        return artifact


class _BenchmarkStage:
    def __init__(
        self,
        order: list[str],
        protocol: BenchmarkProtocol,
        champion: CheckpointParticipant,
    ) -> None:
        self.order = order
        self.protocol = protocol
        self.champion = champion
        self.artifacts: dict[str, tuple[SooRatingEvent, ...]] = {}

    def load(self, operation_id: str) -> tuple[SooRatingEvent, ...] | None:
        return self.artifacts.get(operation_id)

    def execute(
        self,
        operation_id: str,
        candidate: CandidateArtifact,
        champion_checkpoint: str | None,
    ) -> tuple[SooRatingEvent, ...]:
        self.order.append("rating")
        event = SooRatingEvent(
            sequence_index=0,
            protocol_id=self.protocol.protocol_id,
            participant_ids=(candidate.participant.participant_id, self.champion.participant_id),
            seat_assignment=(1, 2),
            turn_order=(1, 2),
            opening_id="tiny-opening-0",
            completed=True,
            winner_id=candidate.participant.participant_id,
            loser_id=self.champion.participant_id,
        )
        result = (event,)
        self.artifacts[operation_id] = result
        return result


class _RecordingReplay(PersistentReplayStore):
    def __init__(self, *args: object, order: list[str], **kwargs: object) -> None:
        self.order = order
        super().__init__(*args, **kwargs)

    def ingest_episode(self, episode: EpisodeResult) -> bool:
        self.order.append("replay")
        return super().ingest_episode(episode)


def test_run_iteration_executes_durable_stages_in_run_stage_order(tmp_path: Path) -> None:
    compatibility = _compatibility()
    protocol = _protocol(compatibility)
    promotion_protocol_id = "soo-promotion-arena-v1"
    champion_hash = "1" * 64
    champion = _participant(
        compatibility,
        training_step=0,
        checkpoint_sha256=champion_hash,
    )
    registry = RatingRegistry(protocol)
    registry.add_participant(champion)
    state_store = RunStateStore(tmp_path / "runs")
    state = state_store.initialize(
        run_id="run-order",
        compatibility=compatibility,
        run_seed=91,
        protocol_ids={
            "promotion": promotion_protocol_id,
            "rating": protocol.protocol_id,
        },
        champion_checkpoint=str(tmp_path / "champion.pt"),
    )
    job = _job(compatibility)
    order: list[str] = []
    replay = _RecordingReplay(
        tmp_path / "replay",
        compatibility,
        capacity=8,
        order=order,
    )
    coordinator = TrainingCoordinator(
        state_store=state_store,
        compatibility=compatibility,
        worker_config=WorkerConfig(worker_count=1, games_per_iteration=1),
        loop_config=TrainingLoopConfig(
            replay_batch_size=1,
            promotion_protocol_id=promotion_protocol_id,
            benchmark_protocol_id=protocol.protocol_id,
        ),
        persistence_config=PersistenceConfig(root=tmp_path / "runs"),
        build_selfplay_jobs=lambda _state, _config: (job,),
        self_play=_SelfPlayStage(order, _episode(job)),
        replay_store=replay,
        training=_TrainingStage(order, state.compatibility_namespace),
        checkpoints=_CheckpointStage(order, compatibility),
        promotion=_PromotionStage(order, promotion_protocol_id),
        benchmark=_BenchmarkStage(order, protocol, champion),
        rating_registry=registry,
    )

    completed = coordinator.run_iteration(state)

    assert order == ["self_play", "replay", "train", "candidate", "arena", "rating"]
    assert completed.stage is RunStage.COMPLETE
    assert completed.iteration == 1
    assert completed.training_step == 1
    assert completed.completed_game_ids == (job.game_id,)
    assert completed.replay_manifest == str(replay.manifest_path)
    assert completed.candidate_checkpoint is not None
    assert completed.champion_checkpoint == completed.candidate_checkpoint
    assert completed.promotion_records[0]["promoted"] is True
    assert completed.rating_records[0]["event_ids"] == (registry.events[0].event_id,)
    assert tuple(completed.stage_completions) == tuple(
        stage.value for stage in tuple(RunStage)[:-1]
    )
    assert Path(completed.candidate_checkpoint).exists()
    assert (tmp_path / "runs" / "soo" / "run-order" / "ratings" / "registry.json").exists()
