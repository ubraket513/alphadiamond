"""Strict Soo/Min checkpoint persistence and compatibility validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import TrainingConfig, config_dict
from .identity import CheckpointCompatibilitySpec
from .trainer import AlphaZeroTrainer

CHECKPOINT_FORMAT_VERSION = 1
_REQUIRED_FIELDS = {
    "format_version",
    "metadata",
    "training_config",
    "training_step",
    "model_state_dict",
    "optimizer_state_dict",
}


class CheckpointError(ValueError):
    """Checkpoint data is missing, malformed, or cannot be restored."""


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    path: Path
    training_step: int
    training_config: TrainingConfig
    metadata: dict[str, Any]


def save_checkpoint(path: str | Path, trainer: AlphaZeroTrainer) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "metadata": trainer.compatibility.to_metadata(),
        "training_config": config_dict(trainer.config),
        "training_step": trainer.training_step,
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_checkpoint(
    path: str | Path,
    trainer: AlphaZeroTrainer,
    *,
    expected: CheckpointCompatibilitySpec,
) -> CheckpointInfo:
    source = Path(path)
    if expected != trainer.compatibility:
        raise ValueError("expected checkpoint compatibility must match the trainer")
    try:
        payload = torch.load(source, map_location=trainer.device, weights_only=True)
    except (OSError, RuntimeError, EOFError) as exc:
        raise CheckpointError(f"cannot read checkpoint {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint root must be a mapping")
    missing = sorted(_REQUIRED_FIELDS - set(payload))
    if missing:
        raise CheckpointError(f"checkpoint is missing fields: {', '.join(missing)}")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint format version: {payload['format_version']!r}"
        )
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise CheckpointError("checkpoint metadata must be a mapping")

    # Semantic gates run before state_dict application, so an incompatible
    # checkpoint cannot partially mutate the destination model.
    expected.assert_compatible(metadata)

    step = payload["training_step"]
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise CheckpointError("checkpoint training_step must be a non-negative integer")
    training_payload = payload["training_config"]
    if not isinstance(training_payload, Mapping):
        raise CheckpointError("checkpoint training_config must be a mapping")
    try:
        training_config = TrainingConfig(**dict(training_payload))
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"invalid checkpoint training_config: {exc}") from exc
    if not isinstance(payload["model_state_dict"], Mapping):
        raise CheckpointError("checkpoint model_state_dict must be a mapping")
    if not isinstance(payload["optimizer_state_dict"], Mapping):
        raise CheckpointError("checkpoint optimizer_state_dict must be a mapping")
    try:
        trainer.model.load_state_dict(payload["model_state_dict"], strict=True)
        trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
    except (RuntimeError, ValueError, KeyError) as exc:
        raise CheckpointError(f"checkpoint state is incompatible: {exc}") from exc
    trainer.training_step = step
    return CheckpointInfo(
        path=source,
        training_step=step,
        training_config=training_config,
        metadata=dict(metadata),
    )


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointError",
    "CheckpointInfo",
    "load_checkpoint",
    "save_checkpoint",
]
