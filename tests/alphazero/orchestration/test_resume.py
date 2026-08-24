from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from diamond.alphazero.config import MCTSConfig, NetworkConfig, SelfPlayConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.inference.protocol import ModelKey
from diamond.alphazero.orchestration.coordinator import (
    CandidateArtifact,
    CoordinatorError,
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
from diamond.contract.state import build_players, initial_state


class _Interrupted(RuntimeError):
    pass


class _InterruptingStateStore(RunStateStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.interrupt_stage: RunStage | None = None
        self.interrupted = False

    def transition(self, state, next_stage, *, completion_marker, **changes):
        if state.stage is self.interrupt_stage and not self.interrupted:
            self.interrupted = True
            raise _Interrupted(state.stage.value)
        return super().transition(
            state,
            next_stage,
            completion_marker=completion_marker,
            **changes,
        )


class _DurableStage:
    def __init__(
        self,
        execute: Callable[..., object],
        *,
        compatibility: CheckpointCompatibilitySpec | None = None,
        protocol_id: str | None = None,
    ) -> None:
        self._execute = execute
        self._compatibility = compatibility
        self._protocol_id = protocol_id
        self.artifacts: dict[str, object] = {}
        self.loads = 0
        self.executions = 0

    @property
    def compatibility(self) -> CheckpointCompatibilitySpec:
        assert self._compatibility is not None
        return self._compatibility

    @property
    def protocol_id(self) -> str:
        assert self._protocol_id is not None
        return self._protocol_id

    def load(self, operation_id: str) -> object | None:
        self.loads += 1
        return self.artifacts.get(operation_id)

    def execute(self, operation_id: str, *args: object) -> object:
        self.executions += 1
        artifact = self._execute(operation_id, *args)
        self.artifacts[operation_id] = artifact
        return artifact


class _CountingReplay(PersistentReplayStore):
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.ingest_attempts = 0
        super().__init__(*args, **kwargs)

    def ingest_episode(self, episode: EpisodeResult) -> bool:
        self.ingest_attempts += 1
        return super().ingest_episode(episode)


def _compatibility(kind: str = "soo") -> CheckpointCompatibilitySpec:
    factory = (
        CheckpointCompatibilitySpec.soo
        if kind == "soo"
        else CheckpointCompatibilitySpec.min
    )
    return factory(
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
        run_seed=17,
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


def _harness(
    tmp_path: Path,
    *,
    promoted: bool = True,
    promotion_compatibility: CheckpointCompatibilitySpec | None = None,
    promotion_identity_protocol: str | None = None,
    benchmark_compatibility: CheckpointCompatibilitySpec | None = None,
    benchmark_identity_protocol: str | None = None,
) -> SimpleNamespace:
    compatibility = _compatibility()
    protocol = _protocol(compatibility)
    promotion_protocol = "soo-promotion-v1"
    champion = _participant(compatibility, 0, "1" * 64)
    registry = RatingRegistry(protocol)
    registry.add_participant(champion)
    state_store = _InterruptingStateStore(tmp_path / "runs")
    state = state_store.initialize(
        run_id="resume-run",
        compatibility=compatibility,
        run_seed=17,
        protocol_ids={"promotion": promotion_protocol, "rating": protocol.protocol_id},
        champion_checkpoint=str(tmp_path / "champion.pt"),
        # `job` is created below, so this never read anything else.
        champion_model_key=ModelKey("Soo", "2.0.0", "a" * 64),
    )
    job = _job(compatibility)
    episode = _episode(job)
    replay = _CountingReplay(tmp_path / "replay", compatibility, capacity=8)

    self_play = _DurableStage(lambda operation_id, jobs: (episode,))

    def train(
        operation_id: str,
        replay_store: PersistentReplayStore,
        batch_size: int,
        expected_step: int,
    ) -> TrainingStepArtifact:
        assert len(replay_store.load_buffer()) == batch_size
        return TrainingStepArtifact(
            operation_id=operation_id,
            compatibility_namespace=state.compatibility_namespace,
            input_training_step=expected_step,
            output_training_step=expected_step + 1,
            metrics={"total_loss": 0.5},
        )

    training = _DurableStage(train)

    def save_candidate(
        operation_id: str,
        path: Path,
        artifact: TrainingStepArtifact,
    ) -> CandidateArtifact:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"candidate-{artifact.output_training_step}".encode()
        path.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        return CandidateArtifact(
            operation_id=operation_id,
            path=path,
            checkpoint_sha256=digest,
            training_step=artifact.output_training_step,
            compatibility_namespace=state.compatibility_namespace,
            participant=_participant(
                compatibility,
                artifact.output_training_step,
                digest,
            ),
        )

    checkpoints = _DurableStage(save_candidate)

    def run_arena(
        operation_id: str,
        candidate: CandidateArtifact,
        champion_checkpoint: str | None,
    ) -> PromotionArtifact:
        return PromotionArtifact(
            operation_id=operation_id,
            promotion_protocol_id=promotion_protocol,
            candidate_sha256=candidate.checkpoint_sha256,
            champion_checkpoint=champion_checkpoint,
            promoted=promoted,
            result={"wins": 4 if promoted else 0, "losses": 0 if promoted else 4},
        )

    promotion = _DurableStage(
        run_arena,
        compatibility=promotion_compatibility or compatibility,
        protocol_id=promotion_identity_protocol or promotion_protocol,
    )

    def run_benchmark(
        operation_id: str,
        candidate: CandidateArtifact,
        champion_checkpoint: str | None,
    ) -> tuple[SooRatingEvent, ...]:
        return (
            SooRatingEvent(
                sequence_index=0,
                protocol_id=protocol.protocol_id,
                participant_ids=(
                    candidate.participant.participant_id,
                    champion.participant_id,
                ),
                seat_assignment=(1, 2),
                turn_order=(1, 2),
                opening_id="opening-0",
                completed=True,
                winner_id=candidate.participant.participant_id,
                loser_id=champion.participant_id,
            ),
        )

    benchmark = _DurableStage(
        run_benchmark,
        compatibility=benchmark_compatibility or compatibility,
        protocol_id=benchmark_identity_protocol or protocol.protocol_id,
    )
    coordinator = TrainingCoordinator(
        state_store=state_store,
        compatibility=compatibility,
        worker_config=WorkerConfig(worker_count=1, games_per_iteration=1),
        loop_config=TrainingLoopConfig(
            replay_batch_size=1,
            promotion_protocol_id=promotion_protocol,
            benchmark_protocol_id=protocol.protocol_id,
        ),
        persistence_config=PersistenceConfig(tmp_path / "runs"),
        build_selfplay_jobs=lambda current, config: (job,),
        self_play=self_play,
        replay_store=replay,
        training=training,
        checkpoints=checkpoints,
        promotion=promotion,
        benchmark=benchmark,
        rating_registry=registry,
    )
    return SimpleNamespace(**locals())


@pytest.mark.parametrize(
    "interrupt_stage",
    [
        RunStage.SELF_PLAY,
        RunStage.REPLAY_INGEST,
        RunStage.TRAIN,
        RunStage.SAVE_CANDIDATE,
        RunStage.PROMOTION_ARENA,
        RunStage.RATING_BENCHMARK,
    ],
)
def test_resume_reuses_every_completed_side_effect_exactly_once(
    tmp_path: Path,
    interrupt_stage: RunStage,
) -> None:
    harness = _harness(tmp_path)
    harness.state_store.interrupt_stage = interrupt_stage

    with pytest.raises(_Interrupted, match=interrupt_stage.value):
        harness.coordinator.run_iteration(harness.state)

    completed = harness.coordinator.resume(harness.state.run_id, "Soo")

    assert completed.stage is RunStage.COMPLETE
    assert completed.completed_game_ids == (harness.job.game_id,)
    assert harness.replay.manifest.game_ids == (harness.job.game_id,)
    assert len(harness.replay.manifest.chunks) == 1
    assert harness.self_play.executions == 1
    assert harness.training.executions == 1
    assert harness.checkpoints.executions == 1
    assert harness.promotion.executions == 1
    assert harness.benchmark.executions == 1
    assert len(harness.registry.events) == 1
    assert len(completed.promotion_records) == 1
    assert len(completed.rating_records) == 1
    assert completed.candidate_checkpoint == completed.champion_checkpoint
    assert Path(completed.candidate_checkpoint).read_bytes() == b"candidate-1"


def test_rejected_candidate_survives_while_champion_pointer_is_preserved(tmp_path: Path) -> None:
    harness = _harness(tmp_path, promoted=False)

    completed = harness.coordinator.run_iteration(harness.state)

    assert completed.candidate_checkpoint is not None
    assert completed.champion_checkpoint == harness.state.champion_checkpoint
    assert completed.candidate_checkpoint != completed.champion_checkpoint
    assert completed.promotion_records[0]["promoted"] is False


def test_resume_rejects_a_candidate_file_whose_recorded_hash_changed(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    harness.state_store.interrupt_stage = RunStage.SAVE_CANDIDATE
    with pytest.raises(_Interrupted):
        harness.coordinator.run_iteration(harness.state)
    candidate = next(iter(harness.checkpoints.artifacts.values()))
    assert isinstance(candidate, CandidateArtifact)
    candidate.path.write_bytes(b"overwritten")

    with pytest.raises(CoordinatorError, match="hash is inconsistent"):
        harness.coordinator.resume(harness.state.run_id, "Soo")

    assert harness.checkpoints.executions == 1


def test_fresh_resume_recovers_a_valid_candidate_published_before_its_journal(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    published: dict[str, Path] = {}

    class CrashAfterPublish:
        def __init__(self, *, crash: bool) -> None:
            self.crash = crash
            self.executions = 0
            self.recoveries = 0
            self.artifacts: dict[str, CandidateArtifact] = {}

        def load(self, operation_id: str) -> CandidateArtifact | None:
            return self.artifacts.get(operation_id)

        def execute(
            self,
            operation_id: str,
            path: Path,
            training: TrainingStepArtifact,
        ) -> CandidateArtifact:
            self.executions += 1
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"candidate-{training.output_training_step}".encode())
            published[operation_id] = path
            if self.crash:
                raise _Interrupted("after checkpoint publish before journal")
            return self.recover(operation_id, path, training)

        def recover(
            self,
            operation_id: str,
            path: Path,
            training: TrainingStepArtifact,
        ) -> CandidateArtifact:
            self.recoveries += 1
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            artifact = CandidateArtifact(
                operation_id=operation_id,
                path=path,
                checkpoint_sha256=digest,
                training_step=training.output_training_step,
                compatibility_namespace=training.compatibility_namespace,
                participant=_participant(
                    harness.compatibility,
                    training.output_training_step,
                    digest,
                ),
            )
            self.artifacts[operation_id] = artifact
            return artifact

    crashing = CrashAfterPublish(crash=True)
    harness.coordinator.checkpoints = crashing

    with pytest.raises(_Interrupted, match="after checkpoint publish"):
        harness.coordinator.run_iteration(harness.state)

    operation_id, candidate_path = next(iter(published.items()))
    assert candidate_path.exists()
    assert crashing.load(operation_id) is None

    fresh = CrashAfterPublish(crash=False)
    harness.coordinator.checkpoints = fresh
    completed = harness.coordinator.resume(harness.state.run_id, "Soo")

    assert completed.stage is RunStage.COMPLETE
    assert fresh.executions == 0
    assert fresh.recoveries == 1
    assert fresh.load(operation_id) is not None


def test_resume_rejects_soo_min_compatibility_crossover_before_side_effects(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    harness.coordinator.compatibility = _compatibility("min")

    with pytest.raises(CoordinatorError, match="compatibility"):
        harness.coordinator.run_iteration(harness.state)

    assert harness.self_play.executions == 0


def test_resume_rejects_changed_benchmark_protocol_namespace(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    harness.coordinator.loop_config = TrainingLoopConfig(
        replay_batch_size=1,
        promotion_protocol_id="soo-promotion-v1",
        benchmark_protocol_id="sha256:changed-benchmark",
    )

    with pytest.raises(CoordinatorError, match="protocol namespace changed"):
        harness.coordinator.run_iteration(harness.state)

    assert harness.self_play.executions == 0


@pytest.mark.parametrize(
    ("stage_name", "identity_field"),
    [
        ("promotion", "compatibility"),
        ("promotion", "protocol"),
        ("benchmark", "compatibility"),
        ("benchmark", "protocol"),
    ],
)
def test_evaluation_identity_mismatch_fails_before_any_load_or_execute(
    tmp_path: Path,
    stage_name: str,
    identity_field: str,
) -> None:
    changes: dict[str, object] = {}
    if stage_name == "promotion" and identity_field == "compatibility":
        changes["promotion_compatibility"] = _compatibility("min")
    elif stage_name == "promotion":
        changes["promotion_identity_protocol"] = "changed-promotion"
    elif identity_field == "compatibility":
        changes["benchmark_compatibility"] = _compatibility("min")
    else:
        changes["benchmark_identity_protocol"] = "changed-benchmark"
    harness = _harness(tmp_path, **changes)

    with pytest.raises(CoordinatorError, match=f"{stage_name}.*{identity_field}"):
        harness.coordinator.run_iteration(harness.state)

    assert harness.self_play.loads == 0
    assert harness.self_play.executions == 0
    assert harness.promotion.loads == 0
    assert harness.promotion.executions == 0
    assert harness.benchmark.loads == 0
    assert harness.benchmark.executions == 0
