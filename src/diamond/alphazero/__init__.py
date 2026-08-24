"""AlphaZero correctness baseline for the authoritative Diamond engine."""

from .game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from .identity import CheckpointCompatibilitySpec, ModelIdentity

__all__ = [
    "AlphaZeroGameAdapter",
    "CheckpointCompatibilitySpec",
    "DiamondSearchAdapter",
    "ModelIdentity",
]
