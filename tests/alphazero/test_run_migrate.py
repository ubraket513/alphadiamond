"""Moving an existing run between CPU and GPU without losing its history.

The run continues in place, so migration is a one-time, reversible edit to the
run's durable checkpoint -- not a fork.  The backup is what makes it reversible.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from diamond.alphazero.checkpoint import load_checkpoint, save_checkpoint
from diamond.alphazero.config import NetworkConfig, TrainingConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.network import SooModel
from diamond.alphazero.replay import ReplayBatch
from diamond.alphazero.run_migrate import (
    MigrationRecord,
    migrate_run_to_device,
    recorded_checkpoint_device,
)
from diamond.alphazero.trainer import AlphaZeroTrainer

NETWORK = NetworkConfig(width=16, residual_blocks=1)


def _spec() -> CheckpointCompatibilitySpec:
    return CheckpointCompatibilitySpec.soo(model_version="0.4.0", network_config=NETWORK)


def _batch(spec: CheckpointCompatibilitySpec) -> ReplayBatch:
    policy = [0.0] * 5329
    policy[73] = 1.0
    return ReplayBatch(
        compatibility=spec,
        node_features=(tuple((0.0,) * 4 for _ in range(73)),),
        policy_targets=(tuple(policy),),
        value_targets=((1.0,),),
    )


def _trainer(device: str = "cpu", *, trained: bool = True) -> AlphaZeroTrainer:
    spec = _spec()
    trainer = AlphaZeroTrainer(
        SooModel(NETWORK, model_version="0.4.0"),
        spec,
        TrainingConfig(
            batch_size=1, learning_rate=2e-3, weight_decay=3e-4, device=device, seed=11
        ),
    )
    if trained:
        trainer.train_batch(_batch(spec))
    return trainer


def _seeded_run(tmp_path, device: str = "cpu", *, recorded_device: str | None = None):
    """A run directory holding a durable latest.pt plus an archived checkpoint.

    ``recorded_device`` rewrites only the device recorded in the metadata,
    leaving the tensors where they are.  That is what lets a CPU-only host
    exercise the real cross-device branch: the recorded device is a string the
    loader compares, not a claim about where the storage lives.
    """
    run_root = tmp_path / "run"
    trainer = _trainer(device=device)
    save_checkpoint(run_root / "latest.pt", trainer, operation_id="seed")
    save_checkpoint(
        run_root / "checkpoints" / "B0-i000000-step000000001.pt",
        trainer,
        operation_id="seed",
    )
    if recorded_device is not None:
        target = run_root / "latest.pt"
        payload = torch.load(target, map_location="cpu", weights_only=True)
        payload["training_config"] = dict(payload["training_config"]) | {
            "device": recorded_device
        }
        torch.save(payload, target)
    return run_root, trainer


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_recorded_device_is_readable_without_mutating_anything(tmp_path) -> None:
    run_root, _trainer_ = _seeded_run(tmp_path)
    before = _digest(run_root / "latest.pt")

    assert recorded_checkpoint_device(run_root / "latest.pt") == torch.device("cpu")
    assert _digest(run_root / "latest.pt") == before


def test_a_matching_device_needs_no_migration(tmp_path) -> None:
    """Re-running the migration is a no-op, so it is safe to leave in a script."""
    run_root, _trainer_ = _seeded_run(tmp_path)
    before = _digest(run_root / "latest.pt")
    destination = _trainer(device="cpu", trained=False)

    record = migrate_run_to_device(
        run_root=run_root,
        trainer=destination,
        expected=destination.compatibility,
        operation_id="migrate",
    )

    assert record is None
    assert _digest(run_root / "latest.pt") == before
    assert not (run_root / "latest.pt.cpu-backup").exists()


def test_migration_requires_an_initialized_run(tmp_path) -> None:
    destination = _trainer(trained=False)

    with pytest.raises(FileNotFoundError):
        migrate_run_to_device(
            run_root=tmp_path / "empty",
            trainer=destination,
            expected=destination.compatibility,
            operation_id="migrate",
        )


def test_migration_refuses_to_clobber_an_existing_backup(tmp_path) -> None:
    """A surviving backup means a migration already happened; never overwrite it."""
    run_root, _trainer_ = _seeded_run(tmp_path, recorded_device="cuda:0")
    backup = run_root / "latest.pt.cpu-backup"
    backup.write_bytes(b"an earlier backup")
    destination = _trainer(device="cpu", trained=False)

    with pytest.raises(FileExistsError):
        migrate_run_to_device(
            run_root=run_root,
            trainer=destination,
            expected=destination.compatibility,
            operation_id="migrate",
        )

    assert backup.read_bytes() == b"an earlier backup"


def test_migration_backs_up_before_rewriting(tmp_path) -> None:
    run_root, source = _seeded_run(tmp_path, recorded_device="cuda:0")
    original = _digest(run_root / "latest.pt")
    archived = _digest(run_root / "checkpoints" / "B0-i000000-step000000001.pt")
    destination = _trainer(device="cpu", trained=False)

    record = migrate_run_to_device(
        run_root=run_root,
        trainer=destination,
        expected=destination.compatibility,
        operation_id="migrate",
    )

    assert isinstance(record, MigrationRecord)
    assert record.training_step == source.training_step
    assert record.source_device == "cuda:0"
    assert record.target_device == "cpu"
    assert record.checkpoint_sha256 == original
    # The backup is a byte-exact copy of what was there before.
    assert _digest(run_root / "latest.pt.cpu-backup") == original
    # Archived per-iteration checkpoints are historical artifacts; leave them.
    assert _digest(run_root / "checkpoints" / "B0-i000000-step000000001.pt") == archived


def test_migration_preserves_the_training_step_and_run_history(tmp_path) -> None:
    run_root, source = _seeded_run(tmp_path, recorded_device="cuda:0")
    destination = _trainer(device="cpu", trained=False)

    migrate_run_to_device(
        run_root=run_root,
        trainer=destination,
        expected=destination.compatibility,
        operation_id="migrate",
    )

    assert destination.training_step == source.training_step
    assert destination.training_step > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_a_migrated_run_reloads_on_cuda_without_the_flag(tmp_path) -> None:
    """After migration the run resumes normally; the flag is a one-time need."""
    run_root, source = _seeded_run(tmp_path, device="cpu")
    destination = _trainer(device="cuda:0", trained=False)

    record = migrate_run_to_device(
        run_root=run_root,
        trainer=destination,
        expected=destination.compatibility,
        operation_id="migrate",
    )

    assert record is not None
    assert record.source_device == "cpu"
    assert record.target_device == "cuda:0"
    assert recorded_checkpoint_device(run_root / "latest.pt") == torch.device("cuda:0")

    # An ordinary resume, with the strict device gate back in force.
    resumed = _trainer(device="cuda:0", trained=False)
    info = load_checkpoint(
        run_root / "latest.pt", resumed, expected=resumed.compatibility
    )
    assert info.training_step == source.training_step


def test_a_migrated_run_reloads_without_the_flag_on_cpu(tmp_path) -> None:
    """The rewritten checkpoint records the destination, so resume is ordinary."""
    run_root, source = _seeded_run(tmp_path, recorded_device="cuda:0")
    destination = _trainer(device="cpu", trained=False)

    migrate_run_to_device(
        run_root=run_root,
        trainer=destination,
        expected=destination.compatibility,
        operation_id="migrate",
    )

    assert recorded_checkpoint_device(run_root / "latest.pt") == torch.device("cpu")
    resumed = _trainer(device="cpu", trained=False)
    info = load_checkpoint(
        run_root / "latest.pt", resumed, expected=resumed.compatibility
    )
    assert info.training_step == source.training_step
