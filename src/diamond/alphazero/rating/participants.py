"""Immutable identities for rated checkpoint artifacts."""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from ..config import NetworkConfig
from ..identity import (
    MIN_MODEL_NAME,
    SOO_MODEL_NAME,
    CheckpointCompatibilitySpec,
    ModelIdentity,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _freeze_metadata(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_metadata(item) for item in value)
    return value


def _compatibility_from_metadata(metadata: Mapping[str, Any]) -> CheckpointCompatibilitySpec:
    try:
        identity = ModelIdentity(
            model_name=metadata["model_name"],
            model_version=metadata["model_version"],
            player_count=metadata["player_count"],
            value_semantics_version=metadata["value_semantics_version"],
        )
        network_payload = metadata["network_config"]
        if not isinstance(network_payload, Mapping):
            raise ValueError("checkpoint network_config must be a mapping")
        factory = {
            SOO_MODEL_NAME: CheckpointCompatibilitySpec.soo,
            MIN_MODEL_NAME: CheckpointCompatibilitySpec.min,
        }[identity.model_name]
        compatibility = factory(
            model_version=identity.model_version,
            network_config=NetworkConfig(**dict(network_payload)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid checkpoint compatibility metadata: {exc}") from exc
    compatibility.assert_compatible(metadata)
    return compatibility


@dataclass(frozen=True, slots=True)
class CheckpointParticipant:
    """An immutable rating identity for one exact checkpoint file."""

    participant_id: str
    model_name: str
    model_version: str
    training_step: int
    checkpoint_sha256: str
    compatibility_metadata: Mapping[str, Any]
    display_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.training_step, int) or isinstance(self.training_step, bool):
            raise ValueError("training_step must be an integer")
        if self.training_step < 0:
            raise ValueError("training_step must be non-negative")
        if not _SHA256.fullmatch(self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")

        expected_id = self._participant_id(
            self.model_name,
            self.model_version,
            self.training_step,
            self.checkpoint_sha256,
        )
        if self.participant_id != expected_id:
            raise ValueError("participant_id does not match checkpoint metadata and hash")
        expected_display_name = f"{self.model_name} {self.model_version} @ {self.training_step}"
        if self.display_name != expected_display_name:
            raise ValueError("display_name does not match checkpoint metadata")
        if not isinstance(self.compatibility_metadata, Mapping):
            raise ValueError("compatibility_metadata must be a mapping")

        metadata = copy.deepcopy(dict(self.compatibility_metadata))
        compatibility = _compatibility_from_metadata(metadata)
        if compatibility.identity.model_name != self.model_name:
            raise ValueError("compatibility metadata model_name does not match participant")
        if compatibility.identity.model_version != self.model_version:
            raise ValueError("compatibility metadata model_version does not match participant")
        object.__setattr__(self, "compatibility_metadata", _freeze_metadata(metadata))

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "CheckpointParticipant":
        """Read artifact identity metadata without restoring a trainer."""
        source = Path(path)
        checkpoint_bytes = source.read_bytes()
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint root must be a mapping")
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("checkpoint metadata must be a mapping")
        training_step = payload.get("training_step")
        if not isinstance(training_step, int) or isinstance(training_step, bool) or training_step < 0:
            raise ValueError("checkpoint training_step must be a non-negative integer")

        compatibility = _compatibility_from_metadata(metadata)
        checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        identity = compatibility.identity
        return cls(
            participant_id=cls._participant_id(
                identity.model_name,
                identity.model_version,
                training_step,
                checkpoint_sha256,
            ),
            model_name=identity.model_name,
            model_version=identity.model_version,
            training_step=training_step,
            checkpoint_sha256=checkpoint_sha256,
            compatibility_metadata=dict(metadata),
            display_name=f"{identity.model_name} {identity.model_version} @ {training_step}",
        )

    @staticmethod
    def _participant_id(
        model_name: str, model_version: str, training_step: int, checkpoint_sha256: str
    ) -> str:
        return f"checkpoint:{model_name}:{model_version}:{training_step}:{checkpoint_sha256}"


__all__ = ["CheckpointParticipant"]
