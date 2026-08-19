"""Bootstrap-only scaffolding that helps cold-start self-play reach terminals."""

from .heuristic import (
    BOOTSTRAP_PRIOR_NONE,
    CANONICAL_TARGET_DISTANCE_V1,
    CanonicalTargetDistancePrior,
    target_distance_table,
)
from .evaluator import BootstrapPriorEvaluator

__all__ = [
    "BOOTSTRAP_PRIOR_NONE",
    "CANONICAL_TARGET_DISTANCE_V1",
    "BootstrapPriorEvaluator",
    "CanonicalTargetDistancePrior",
    "target_distance_table",
]
