"""Versioned, immutable benchmark protocol identities."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..identity import CheckpointCompatibilitySpec


@dataclass(frozen=True, slots=True)
class EloConfig:
    initial_rating: float = 1000.0
    k_factor: float = 32.0
    logistic_scale: float = 400.0
    rating_system_version: str = "soo-elo-v1"


@dataclass(frozen=True, slots=True)
class TrueSkillConfig:
    mu: float = 25.0
    sigma: float = 25.0 / 3.0
    beta: float = 25.0 / 6.0
    tau: float = 0.0
    draw_probability: float = 0.0
    backend: str | None = None
    rating_system_version: str = "min-trueskill-v1"


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("rating_parameters keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in copy.deepcopy(dict(value)).items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            _json_value(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"benchmark protocol must be JSON-serializable: {exc}") from exc


@dataclass(frozen=True, slots=True)
class BenchmarkProtocol:
    compatibility: CheckpointCompatibilitySpec
    simulations: int
    c_puct: float
    dirichlet_epsilon: float
    decision_temperature: float
    max_game_moves: int
    opening_suite_version: str
    opening_suite_hash: str
    rating_system_version: str
    rating_parameters: dict[str, object]
    inference_numeric_mode: str = "fp32"

    def __post_init__(self) -> None:
        if not isinstance(self.compatibility, CheckpointCompatibilitySpec):
            raise ValueError("compatibility must be a CheckpointCompatibilitySpec")
        if not isinstance(self.simulations, int) or isinstance(self.simulations, bool) or self.simulations <= 0:
            raise ValueError("simulations must be a positive integer")
        if not math.isfinite(self.c_puct) or self.c_puct <= 0.0:
            raise ValueError("c_puct must be a positive finite value")
        if self.dirichlet_epsilon != 0.0:
            raise ValueError("dirichlet_epsilon must be 0.0 for benchmark ratings")
        if self.decision_temperature != 0.0:
            raise ValueError("decision_temperature must be 0.0 for benchmark ratings")
        if (
            not isinstance(self.max_game_moves, int)
            or isinstance(self.max_game_moves, bool)
            or self.max_game_moves <= 0
        ):
            raise ValueError("max_game_moves must be a positive integer")
        if not isinstance(self.rating_parameters, Mapping):
            raise ValueError("rating_parameters must be a mapping")
        for name, value in (
            ("opening_suite_version", self.opening_suite_version),
            ("opening_suite_hash", self.opening_suite_hash),
            ("rating_system_version", self.rating_system_version),
            ("inference_numeric_mode", self.inference_numeric_mode),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "rating_parameters", _freeze_json(self.rating_parameters))
        self._canonical_payload()

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "compatibility": self.compatibility.to_metadata(),
            "simulations": self.simulations,
            "c_puct": self.c_puct,
            "dirichlet_epsilon": self.dirichlet_epsilon,
            "decision_temperature": self.decision_temperature,
            "max_game_moves": self.max_game_moves,
            "opening_suite_version": self.opening_suite_version,
            "opening_suite_hash": self.opening_suite_hash,
            "rating_system_version": self.rating_system_version,
            "rating_parameters": self.rating_parameters,
            "inference_numeric_mode": self.inference_numeric_mode,
        }

    @property
    def protocol_id(self) -> str:
        digest = hashlib.sha256(_canonical_json(self._canonical_payload())).hexdigest()
        return f"sha256:{digest}"


def benchmark_protocol_id(protocol: BenchmarkProtocol) -> str:
    """Return the deterministic namespace identity for a benchmark protocol."""
    return protocol.protocol_id


__all__ = [
    "BenchmarkProtocol",
    "EloConfig",
    "TrueSkillConfig",
    "benchmark_protocol_id",
]
