"""AlphaZero correctness baseline for the authoritative Diamond engine."""

from .action_codec import ActionCodec, ActionSpaceSpec
from .encoder import CanonicalEncoder, EncodedState
from .game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from .identity import CheckpointCompatibilitySpec, ModelIdentity

__all__ = [
    "ActionCodec",
    "ActionSpaceSpec",
    "AlphaZeroGameAdapter",
    "CanonicalEncoder",
    "CheckpointCompatibilitySpec",
    "DiamondSearchAdapter",
    "EncodedState",
    "ModelIdentity",
]
