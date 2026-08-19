"""One-time cross-device migration of an existing run's durable checkpoint.

A run continues in place when it moves from CPU to GPU: its loop state, ledger
and replay buffer all carry forward, and only ``latest.pt`` is rewritten so it
records the new device.  Because that rewrite is destructive, it always leaves a
byte-exact backup behind, and it refuses to run a second time over one.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from .checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from .config import TrainingConfig
from .identity import CheckpointCompatibilitySpec
from .trainer import AlphaZeroTrainer

BACKUP_SUFFIX = ".cpu-backup"
"""Kept beside ``latest.pt``; the only way back from a bad migration."""


@dataclass(frozen=True, slots=True)
class MigrationRecord:
    """Provenance for one device migration, for the run's ledger."""

    checkpoint_path: str
    checkpoint_sha256: str
    training_step: int
    source_device: str
    target_device: str
    backup_path: str

    def to_payload(self) -> dict[str, object]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "training_step": self.training_step,
            "source_device": self.source_device,
            "target_device": self.target_device,
            "backup_path": self.backup_path,
        }


def recorded_checkpoint_device(path: str | Path) -> torch.device:
    """Return the device a checkpoint records, without touching the file."""
    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as exc:
        raise CheckpointError(f"cannot read checkpoint {source}: {exc}") from exc
    if not isinstance(payload, Mapping) or "training_config" not in payload:
        raise CheckpointError("checkpoint is missing its training configuration")
    training_config = payload["training_config"]
    if not isinstance(training_config, Mapping) or "device" not in training_config:
        raise CheckpointError("checkpoint training_config does not record a device")
    try:
        return torch.device(training_config["device"])
    except (RuntimeError, ValueError, TypeError) as exc:
        raise CheckpointError(f"invalid checkpoint training device: {exc}") from exc


def migrate_run_to_device(
    *,
    run_root: str | Path,
    trainer: AlphaZeroTrainer,
    expected: CheckpointCompatibilitySpec,
    operation_id: str,
) -> MigrationRecord | None:
    """Load ``run_root/latest.pt`` onto the trainer's device, once.

    Returns ``None`` when the checkpoint already records the trainer's device,
    which makes the call idempotent and therefore safe to leave in a launch
    script.  Otherwise the checkpoint is backed up, loaded across devices, and
    rewritten so that every later resume is an ordinary same-device load.
    """
    root = Path(run_root)
    latest = root / "latest.pt"
    if not latest.exists():
        raise FileNotFoundError(
            f"cannot migrate an uninitialized run: {latest} does not exist"
        )

    source_device = recorded_checkpoint_device(latest)
    if source_device == trainer.device:
        return None

    backup = latest.with_suffix(latest.suffix + BACKUP_SUFFIX)
    if backup.exists():
        raise FileExistsError(
            f"{backup} already exists: this run has already been migrated, and "
            "overwriting the backup would discard the only way back"
        )

    checkpoint_sha256 = hashlib.sha256(latest.read_bytes()).hexdigest()
    # Back up first: after this point the rewrite is recoverable.
    shutil.copy2(latest, backup)

    target_config: TrainingConfig = trainer.config
    info = load_checkpoint(
        latest, trainer, expected=expected, allow_device_migration=True
    )
    # load_checkpoint installs the checkpoint's own config, which still names the
    # old device.  Restore the destination's so the rewritten artifact records
    # where it now lives and no later resume needs the migration flag.
    trainer.config = target_config
    save_checkpoint(latest, trainer, operation_id=operation_id)

    return MigrationRecord(
        checkpoint_path=str(latest),
        checkpoint_sha256=checkpoint_sha256,
        training_step=info.training_step,
        source_device=str(source_device),
        target_device=str(trainer.device),
        backup_path=str(backup),
    )


__all__ = [
    "BACKUP_SUFFIX",
    "MigrationRecord",
    "migrate_run_to_device",
    "recorded_checkpoint_device",
]
