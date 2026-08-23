"""Strict Soo/Min checkpoint persistence and compatibility validation."""

from __future__ import annotations

import copy
import hashlib
import io
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .config import NetworkConfig, TrainingConfig, config_dict
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
_INFERENCE_REQUIRED_FIELDS = {"format_version", "metadata", "model_state_dict"}


class CheckpointError(ValueError):
    """Checkpoint data is missing, malformed, or cannot be restored."""


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    path: Path
    training_step: int
    training_config: TrainingConfig
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class InferenceCheckpointInfo:
    """Validated immutable checkpoint data sufficient for model inference."""

    path: Path
    checkpoint_sha256: str
    metadata: dict[str, Any]


def save_checkpoint(
    path: str | Path,
    trainer: AlphaZeroTrainer,
    *,
    operation_id: str | None = None,
) -> Path:
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
    if operation_id is not None:
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("operation_id must be a non-empty string")
        payload["operation_id"] = operation_id
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_checkpoint(
    path: str | Path,
    trainer: AlphaZeroTrainer,
    *,
    expected: CheckpointCompatibilitySpec,
    allow_device_migration: bool = False,
) -> CheckpointInfo:
    """Restore a checkpoint into ``trainer``, enforcing semantic compatibility.

    ``allow_device_migration`` waives only the recorded-device check, for the
    deliberate act of moving a run between CPU and GPU.  Every other gate still
    applies, and it stays off by default so an accidental cross-device load
    remains loud.
    """
    source = Path(path)
    if expected != trainer.compatibility:
        raise ValueError("expected checkpoint compatibility must match the trainer")
    try:
        payload = torch.load(source, map_location=trainer.device, weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
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
    if (
        training_config.learning_rate <= 0
        or training_config.batch_size <= 0
        or training_config.weight_decay < 0
    ):
        raise CheckpointError("invalid checkpoint training_config values")
    try:
        checkpoint_device = torch.device(training_config.device)
    except (RuntimeError, ValueError) as exc:
        raise CheckpointError(f"invalid checkpoint training device: {exc}") from exc
    if checkpoint_device != trainer.device and not allow_device_migration:
        raise CheckpointError(
            f"checkpoint training device {checkpoint_device} does not match "
            f"trainer device {trainer.device}"
        )
    if not isinstance(payload["model_state_dict"], Mapping):
        raise CheckpointError("checkpoint model_state_dict must be a mapping")
    if not isinstance(payload["optimizer_state_dict"], Mapping):
        raise CheckpointError("checkpoint optimizer_state_dict must be a mapping")
    parameter_groups = payload["optimizer_state_dict"].get("param_groups")
    if not isinstance(parameter_groups, list) or not parameter_groups:
        raise CheckpointError(
            "checkpoint state is incompatible: optimizer param_groups are malformed"
        )
    for group in parameter_groups:
        if not isinstance(group, Mapping):
            raise CheckpointError(
                "checkpoint state is incompatible: optimizer param_groups are malformed"
            )
        if group.get("lr") != training_config.learning_rate:
            raise CheckpointError("checkpoint optimizer learning_rate mismatch")
        if group.get("weight_decay") != training_config.weight_decay:
            raise CheckpointError("checkpoint optimizer weight_decay mismatch")

    # Validate both states against isolated copies before mutating the live
    # trainer. A malformed optimizer must not leave checkpoint model weights
    # installed in an otherwise rejected destination.
    try:
        staged_model = copy.deepcopy(trainer.model)
        staged_optimizer = copy.deepcopy(trainer.optimizer)
        staged_model.load_state_dict(payload["model_state_dict"], strict=True)
        staged_optimizer.load_state_dict(payload["optimizer_state_dict"])
    except (RuntimeError, ValueError, KeyError) as exc:
        raise CheckpointError(f"checkpoint state is incompatible: {exc}") from exc

    trainer.model.load_state_dict(payload["model_state_dict"], strict=True)
    trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
    trainer.config = training_config
    trainer.training_step = step
    return CheckpointInfo(
        path=source,
        training_step=step,
        training_config=training_config,
        metadata=dict(metadata),
    )


def load_inference_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    expected: CheckpointCompatibilitySpec,
    device: str | torch.device = "cpu",
) -> InferenceCheckpointInfo:
    """Strictly load model weights without requiring training-only state.

    This intentionally has separate required fields from :func:`load_checkpoint`:
    inference artifacts need semantic metadata and exact model weights, but not an
    optimizer, training configuration, or live trainer.
    """
    source = Path(path)
    try:
        checkpoint_bytes = source.read_bytes()
        payload = torch.load(
            io.BytesIO(checkpoint_bytes), map_location=device, weights_only=True
        )
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise CheckpointError(f"cannot read checkpoint {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint root must be a mapping")
    missing = sorted(_INFERENCE_REQUIRED_FIELDS - set(payload))
    if missing:
        raise CheckpointError(f"checkpoint is missing fields: {', '.join(missing)}")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint format version: {payload['format_version']!r}"
        )
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise CheckpointError("checkpoint metadata must be a mapping")

    # Preserve the training loader's ordering: validate semantic identity before
    # applying any state to the destination model.
    expected.assert_compatible(metadata)
    model_state = payload["model_state_dict"]
    if not isinstance(model_state, Mapping):
        raise CheckpointError("checkpoint model_state_dict must be a mapping")
    try:
        staged_model = copy.deepcopy(model)
        staged_model.load_state_dict(model_state, strict=True)
    except (RuntimeError, ValueError, KeyError) as exc:
        raise CheckpointError(f"checkpoint state is incompatible: {exc}") from exc
    model.load_state_dict(model_state, strict=True)
    return InferenceCheckpointInfo(
        path=source,
        checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        metadata=dict(metadata),
    )


def checkpoint_network_config(path: str | Path) -> NetworkConfig:
    """Return the network shape a checkpoint was trained at.

    Every checkpoint records its own ``network_config`` in the compatibility
    metadata, so a tool comparing two checkpoints of *different* shapes -- a
    depth or width sweep -- can build each side correctly instead of forcing
    both through one config file's ``network`` block.  Reading the shape from
    the file is also the only way to be sure which parent a transplant came
    from; a config file merely says what someone expected.

    Every other compatibility field is left to ``assert_compatible``: this
    waives nothing, it only discovers the one field that is legitimately
    allowed to differ between the two sides of an architecture comparison.
    """
    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise CheckpointError(f"cannot read checkpoint {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CheckpointError(f"checkpoint {source} root must be a mapping")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise CheckpointError(f"checkpoint {source} has no metadata mapping")
    network = metadata.get("network_config")
    if not isinstance(network, Mapping):
        raise CheckpointError(f"checkpoint {source} metadata has no network_config")
    try:
        return NetworkConfig(**network)
    except TypeError as exc:
        raise CheckpointError(f"checkpoint {source} network_config is malformed: {exc}") from exc


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointError",
    "CheckpointInfo",
    "checkpoint_network_config",
    "InferenceCheckpointInfo",
    "load_checkpoint",
    "load_inference_checkpoint",
    "save_checkpoint",
]
