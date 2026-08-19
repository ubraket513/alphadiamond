"""Serializable configuration for the Milestone-1 reference system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .bootstrap.heuristic import (
    BOOTSTRAP_PRIOR_NONE,
    CANONICAL_TARGET_DISTANCE_V1,
    CANONICAL_TARGET_VACANCY_DISTANCE_V2,
)

BOOTSTRAP_PRIORS = frozenset(
    {
        BOOTSTRAP_PRIOR_NONE,
        CANONICAL_TARGET_DISTANCE_V1,
        CANONICAL_TARGET_VACANCY_DISTANCE_V2,
    }
)


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
    bootstrap_prior: str = BOOTSTRAP_PRIOR_NONE
    """Opt-in cold-start scaffolding; self-play only, never arena or rating."""

    def __post_init__(self) -> None:
        if self.bootstrap_prior not in BOOTSTRAP_PRIORS:
            raise ValueError(
                f"bootstrap_prior must be one of {sorted(BOOTSTRAP_PRIORS)}"
            )


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    capacity: int = 100_000
    seed: int = 0


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    seed: int = 0


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    # 36 is divisible by the complete Soo (4) and Min (18) balance cycles.
    games: int = 36
    seed: int = 0
    max_moves: int = 2000
    promotion_threshold: float = 0.55


def config_dict(config: object) -> dict[str, Any]:
    """Return a JSON-compatible dataclass payload for checkpoint metadata."""
    return asdict(config)  # type: ignore[arg-type]


__all__ = [
    "BOOTSTRAP_PRIORS",
    "ArenaConfig",
    "MCTSConfig",
    "NetworkConfig",
    "ReplayConfig",
    "SelfPlayConfig",
    "TrainingConfig",
    "config_dict",
]
