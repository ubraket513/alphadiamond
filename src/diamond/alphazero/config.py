"""Serializable configuration for the Milestone-1 reference system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    width: int = 128
    residual_blocks: int = 6


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    seed: int = 0


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    max_moves: int = 2000
    temperature_moves: int = 20
    temperature: float = 1.0
    seed: int = 0


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    capacity: int = 100_000
    seed: int = 0


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    games: int = 20
    seed: int = 0


def config_dict(config: object) -> dict[str, Any]:
    """Return a JSON-compatible dataclass payload for checkpoint metadata."""
    return asdict(config)  # type: ignore[arg-type]


__all__ = [
    "ArenaConfig",
    "MCTSConfig",
    "NetworkConfig",
    "ReplayConfig",
    "SelfPlayConfig",
    "TrainingConfig",
    "config_dict",
]
