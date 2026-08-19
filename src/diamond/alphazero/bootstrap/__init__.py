"""Bootstrap-only scaffolding that helps cold-start self-play reach terminals."""

from .evaluator import (
    BootstrapPriorEvaluator,
    VacancyPriorEvaluator,
    bootstrap_evaluator,
)
from .heuristic import (
    BOOTSTRAP_PRIOR_NONE,
    CANONICAL_TARGET_DISTANCE_V1,
    CANONICAL_TARGET_VACANCY_DISTANCE_V2,
    CanonicalTargetDistancePrior,
    CanonicalTargetVacancyDistancePrior,
    pairwise_distance_table,
    target_distance_table,
)
from .probe import ProbeReport, format_report, run_probe

__all__ = [
    "BOOTSTRAP_PRIOR_NONE",
    "CANONICAL_TARGET_DISTANCE_V1",
    "CANONICAL_TARGET_VACANCY_DISTANCE_V2",
    "BootstrapPriorEvaluator",
    "CanonicalTargetDistancePrior",
    "CanonicalTargetVacancyDistancePrior",
    "ProbeReport",
    "VacancyPriorEvaluator",
    "bootstrap_evaluator",
    "format_report",
    "pairwise_distance_table",
    "run_probe",
    "target_distance_table",
]
