from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from diamond.alphazero.checkpoint import save_checkpoint
from diamond.alphazero.config import NetworkConfig, TrainingConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.network import SooModel
from diamond.alphazero.orchestration.coordinator import TrainingStepArtifact
from diamond.alphazero.orchestration.production import (
    ProductionArtifactStore,
    ProductionCheckpointStage,
    ProductionConfig,
    build_authoritative_selfplay_jobs,
)
from diamond.alphazero.orchestration.coordinator import WorkerConfig
from diamond.alphazero.orchestration.run_state import RunStateStore
from diamond.alphazero.inference.protocol import ModelKey
from diamond.game.state import build_players, initial_state
from diamond.alphazero.trainer import AlphaZeroTrainer


def _compatibility() -> CheckpointCompatibilitySpec:
    return CheckpointCompatibilitySpec.soo(
        model_version="2.0.0",
        network_config=NetworkConfig(width=8, residual_blocks=1),
    )


def _trainer() -> AlphaZeroTrainer:
    compatibility = _compatibility()
    trainer = AlphaZeroTrainer(
        SooModel(compatibility.network_config, model_version="2.0.0"),
        compatibility,
        TrainingConfig(batch_size=1, weight_decay=0.0),
    )
    trainer.training_step = 1
    return trainer


def _training_artifact() -> TrainingStepArtifact:
    return TrainingStepArtifact(
        operation_id="training-operation",
        compatibility_namespace="sha256:test-compatibility",
        input_training_step=0,
        output_training_step=1,
    )


def test_production_config_is_strictly_json_round_trippable() -> None:
    payload = {
        "schema_version": 1,
        "model_name": "Soo",
        "model_version": "2.0.0",
        "network": {"width": 8, "residual_blocks": 1},
        "mcts": {
            "simulations": 1,
            "c_puct": 1.5,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
            "seed": 7,
        },
        "self_play": {
            "max_moves": 2000,
            "temperature_moves": 20,
            "temperature": 1.0,
            "seed": 7,
            "bootstrap_prior": "none",
            "max_game_seconds": None,
        },
        "replay": {"capacity": 128, "seed": 7},
        "training": {
            "batch_size": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "device": "cpu",
            "seed": 7,
        },
        "arena": {
            "games": 4,
            "seed": 7,
            "max_moves": 2000,
            "promotion_threshold": 0.55,
        },
        "workers": {
            "worker_count": 2,
            "games_per_iteration": 2,
            "retry_id": "attempt-0",
        },
        "inference": {
            "max_batch_size": 8,
            "max_wait_ms": 2,
            "request_queue_capacity": 32,
            "response_timeout_s": 10.0,
        },
        "benchmark": {
            "opening_count": 1,
            "opening_max_depth": 0,
            "opening_seed": 7,
            "opening_suite_version": "production-openings-v1",
        },
        "run_seed": 7,
    }

    config = ProductionConfig.from_payload(json.loads(json.dumps(payload)))

    assert config.to_payload() == payload
    assert config.compatibility == _compatibility()
    with pytest.raises(ValueError, match="unexpected"):
        ProductionConfig.from_payload(payload | {"tiny_fixture": True})


def test_candidate_publish_crash_is_strictly_recovered_by_a_fresh_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    path = runtime / "soo" / "run-1" / "checkpoints" / "candidate.pt"
    operation_id = "sha256:" + "a" * 64
    artifacts = ProductionArtifactStore(runtime / "soo" / "run-1" / "artifacts")
    crashing = ProductionCheckpointStage(
        trainer=_trainer(),
        artifacts=artifacts,
        compatibility=_compatibility(),
        persistence_root=runtime,
    )

    def fail_journal(*_args, **_kwargs):
        raise RuntimeError("simulated crash after publish")

    monkeypatch.setattr(artifacts, "write", fail_journal)
    with pytest.raises(RuntimeError, match="after publish"):
        crashing.execute(operation_id, path, _training_artifact())

    assert path.exists()
    fresh = ProductionCheckpointStage(
        trainer=_trainer(),
        artifacts=ProductionArtifactStore(runtime / "soo" / "run-1" / "artifacts"),
        compatibility=_compatibility(),
        persistence_root=runtime,
    )
    recovered = fresh.recover(operation_id, path, _training_artifact())

    assert recovered.operation_id == operation_id
    assert recovered.path == path
    assert recovered.training_step == 1
    assert fresh.load(operation_id) == recovered


def test_candidate_recovery_rejects_valid_but_orphan_checkpoint_content(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    path = runtime / "soo" / "run-1" / "checkpoints" / "candidate.pt"
    save_checkpoint(path, _trainer(), operation_id="sha256:" + "b" * 64)
    stage = ProductionCheckpointStage(
        trainer=_trainer(),
        artifacts=ProductionArtifactStore(runtime / "soo" / "run-1" / "artifacts"),
        compatibility=_compatibility(),
        persistence_root=runtime,
    )

    with pytest.raises(ValueError, match="operation identity"):
        stage.recover("sha256:" + "a" * 64, path, _training_artifact())

    assert stage.load("sha256:" + "a" * 64) is None


def test_production_selfplay_jobs_start_from_authoritative_initial_state_and_champion(
    tmp_path: Path,
) -> None:
    compatibility = _compatibility()
    champion_key = ModelKey("Soo", "2.0.0", "c" * 64)
    state = RunStateStore(tmp_path).initialize(
        run_id="production-jobs",
        compatibility=compatibility,
        run_seed=7,
        protocol_ids={"promotion": "promotion-v1", "rating": "rating-v1"},
        champion_checkpoint=str(tmp_path / "champion.pt"),
        champion_model_key=champion_key,
    )
    config = ProductionConfig.from_payload(
        {
            "schema_version": 1,
            "model_name": "Soo",
            "model_version": "2.0.0",
            "network": {"width": 8, "residual_blocks": 1},
            "mcts": {
                "simulations": 1,
                "c_puct": 1.5,
                "dirichlet_alpha": 0.3,
                "dirichlet_epsilon": 0.25,
                "seed": 7,
            },
            "self_play": {
                "max_moves": 2000,
                "temperature_moves": 20,
                "temperature": 1.0,
                "seed": 7,
                "bootstrap_prior": "none",
                "max_game_seconds": None,
            },
            "replay": {"capacity": 128, "seed": 7},
            "training": {
                "batch_size": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "device": "cpu",
                "seed": 7,
            },
            "arena": {
                "games": 4,
                "seed": 7,
                "max_moves": 2000,
                "promotion_threshold": 0.55,
            },
            "workers": {
                "worker_count": 2,
                "games_per_iteration": 2,
                "retry_id": "attempt-0",
            },
            "inference": {
                "max_batch_size": 8,
                "max_wait_ms": 2,
                "request_queue_capacity": 32,
                "response_timeout_s": 10.0,
            },
            "benchmark": {
                "opening_count": 1,
                "opening_max_depth": 0,
                "opening_seed": 7,
                "opening_suite_version": "production-openings-v1",
            },
            "run_seed": 7,
        }
    )

    jobs = build_authoritative_selfplay_jobs(
        state,
        WorkerConfig(worker_count=2, games_per_iteration=2),
        config,
    )

    players = build_players(2)
    assert len(jobs) == 2
    assert all(job.model_key == champion_key for job in jobs)
    assert all(job.initial_state == initial_state(players) for job in jobs)
