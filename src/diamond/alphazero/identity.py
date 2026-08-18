"""Official Soo/Min model identity and checkpoint compatibility contract."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .config import NetworkConfig, config_dict

RULESET_VERSION = "diamond-authoritative-rules-v1"
BOARD_TOPOLOGY_VERSION = "diamond73-v1"
ENCODER_VERSION = "diamond-camp-relative-v1"
ACTION_SPACE_VERSION = "diamond73-srcdst-v1"

SOO_MODEL_NAME = "Soo"
MIN_MODEL_NAME = "Min"
SOO_VALUE_SEMANTICS_VERSION = "current-player-scalar-winloss-v1"
MIN_VALUE_SEMANTICS_VERSION = "canonical-placement-utility-1-0-minus1-v1"

_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][0-9A-Za-z.-]+)?$")


class CheckpointCompatibilityError(ValueError):
    """A checkpoint disagrees with one or more semantic compatibility gates."""


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    model_name: str
    model_version: str
    player_count: int
    value_semantics_version: str

    def __post_init__(self) -> None:
        expected = {
            SOO_MODEL_NAME: (2, SOO_VALUE_SEMANTICS_VERSION),
            MIN_MODEL_NAME: (3, MIN_VALUE_SEMANTICS_VERSION),
        }.get(self.model_name)
        if expected is None:
            raise ValueError(f"unknown AlphaZero model name: {self.model_name}")
        if (self.player_count, self.value_semantics_version) != expected:
            raise ValueError(
                f"{self.model_name} requires player_count={expected[0]} and "
                f"value_semantics_version={expected[1]}"
            )
        if not _SEMVER.fullmatch(self.model_version):
            raise ValueError("model_version must be a semantic version such as 0.1.0")

    @classmethod
    def soo(cls, model_version: str) -> "ModelIdentity":
        return cls(
            model_name=SOO_MODEL_NAME,
            model_version=model_version,
            player_count=2,
            value_semantics_version=SOO_VALUE_SEMANTICS_VERSION,
        )

    @classmethod
    def min(cls, model_version: str) -> "ModelIdentity":
        return cls(
            model_name=MIN_MODEL_NAME,
            model_version=model_version,
            player_count=3,
            value_semantics_version=MIN_VALUE_SEMANTICS_VERSION,
        )


@dataclass(frozen=True, slots=True)
class CheckpointCompatibilitySpec:
    """Independent exact-match gates; model_version never substitutes for them."""

    identity: ModelIdentity
    network_config: NetworkConfig
    ruleset_version: str = RULESET_VERSION
    board_topology_version: str = BOARD_TOPOLOGY_VERSION
    encoder_version: str = ENCODER_VERSION
    action_space_version: str = ACTION_SPACE_VERSION

    @classmethod
    def soo(
        cls, *, model_version: str, network_config: NetworkConfig
    ) -> "CheckpointCompatibilitySpec":
        return cls(ModelIdentity.soo(model_version), network_config)

    @classmethod
    def min(
        cls, *, model_version: str, network_config: NetworkConfig
    ) -> "CheckpointCompatibilitySpec":
        return cls(ModelIdentity.min(model_version), network_config)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.identity.model_name,
            "model_version": self.identity.model_version,
            "player_count": self.identity.player_count,
            "ruleset_version": self.ruleset_version,
            "board_topology_version": self.board_topology_version,
            "encoder_version": self.encoder_version,
            "action_space_version": self.action_space_version,
            "value_semantics_version": self.identity.value_semantics_version,
            "network_config": config_dict(self.network_config),
        }

    def assert_compatible(self, checkpoint_metadata: Mapping[str, Any]) -> None:
        for field, expected in self.to_metadata().items():
            if field not in checkpoint_metadata:
                raise CheckpointCompatibilityError(
                    f"checkpoint metadata is missing compatibility field {field}"
                )
            actual = checkpoint_metadata[field]
            if actual != expected:
                raise CheckpointCompatibilityError(
                    f"checkpoint {field} mismatch: expected {expected!r}, got {actual!r}"
                )


__all__ = [
    "ACTION_SPACE_VERSION",
    "BOARD_TOPOLOGY_VERSION",
    "CheckpointCompatibilityError",
    "CheckpointCompatibilitySpec",
    "ENCODER_VERSION",
    "MIN_MODEL_NAME",
    "MIN_VALUE_SEMANTICS_VERSION",
    "ModelIdentity",
    "RULESET_VERSION",
    "SOO_MODEL_NAME",
    "SOO_VALUE_SEMANTICS_VERSION",
]
