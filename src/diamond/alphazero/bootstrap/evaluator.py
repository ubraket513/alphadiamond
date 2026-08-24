"""Evaluator decorators that swap neural priors for a bootstrap prior.

Framework-neutral by construction: these import no Torch and wrap any object
satisfying the public ``Evaluator`` protocol, local or remote alike.
"""

from __future__ import annotations

from ...contract.camps import Camp
from ..evaluator.base import EvalRequest, EvalResult, Evaluator
from ..native.topology import camp_positions
from .heuristic import (
    CANONICAL_TARGET_DISTANCE_V1,
    CANONICAL_TARGET_VACANCY_DISTANCE_V2,
    CanonicalTargetDistancePrior,
    CanonicalTargetVacancyDistancePrior,
    pairwise_distance_table,
    target_distance_table,
)


class _BasePriorEvaluator:
    """Preserve every base value exactly; replace only the prior.

    MCTS stays unaware that the prior is heuristic, and the base evaluator keeps
    doing the batching, model routing and GPU inference it already does.
    """

    identity: str

    def __init__(
        self,
        base: Evaluator,
        *,
        board=None,
    ) -> None:
        if not hasattr(base, "evaluate"):
            raise TypeError("base must satisfy the Evaluator protocol")
        self.base = base
        # `board` is accepted and ignored: the tables come from the core.

    def _priors(self, request: EvalRequest) -> dict[int, float]:
        raise NotImplementedError

    def evaluate(self, requests: tuple[EvalRequest, ...]) -> tuple[EvalResult, ...]:
        results = self.base.evaluate(requests)
        if len(results) != len(requests):
            raise ValueError("base evaluator returned a mismatched result count")
        return tuple(
            EvalResult(priors=self._priors(request), value=result.value)
            for request, result in zip(requests, results)
        )


class BootstrapPriorEvaluator(_BasePriorEvaluator):
    """``canonical-target-distance-v1``: per-move progress against a fixed table.

    Kept as a measured experimental result.  It drives the opening and midgame
    well but stalls in target-camp packing, because a state-independent table
    cannot see which target holes are already filled.
    """

    def __init__(
        self,
        base: Evaluator,
        *,
        board=None,
        prior: CanonicalTargetDistancePrior | None = None,
    ) -> None:
        super().__init__(base, board=board)
        self.prior = CanonicalTargetDistancePrior() if prior is None else prior
        self.distance = target_distance_table()

    @property
    def identity(self) -> str:
        return self.prior.identity

    def _priors(self, request: EvalRequest) -> dict[int, float]:
        return self.prior.priors(request.legal_action_ids, self.distance)


class VacancyPriorEvaluator(_BasePriorEvaluator):
    """``canonical-target-vacancy-distance-v2``: potential over unfilled targets."""

    def __init__(
        self,
        base: Evaluator,
        *,
        board=None,
        prior: CanonicalTargetVacancyDistancePrior | None = None,
    ) -> None:
        super().__init__(base, board=board)
        self.prior = CanonicalTargetVacancyDistancePrior() if prior is None else prior
        self.pairwise = pairwise_distance_table()
        self.target = frozenset(camp_positions(Camp.Z_NEG))

    @property
    def identity(self) -> str:
        return self.prior.identity

    def _priors(self, request: EvalRequest) -> dict[int, float]:
        return self.prior.priors(
            request.legal_action_ids,
            self.target,
            self.pairwise,
            request.node_features,
        )


def bootstrap_evaluator(base: Evaluator, bootstrap_prior: str) -> Evaluator:
    """Wrap ``base`` for the configured bootstrap prior, or return it unchanged."""
    if bootstrap_prior == CANONICAL_TARGET_DISTANCE_V1:
        return BootstrapPriorEvaluator(base)
    if bootstrap_prior == CANONICAL_TARGET_VACANCY_DISTANCE_V2:
        return VacancyPriorEvaluator(base)
    return base


__all__ = [
    "BootstrapPriorEvaluator",
    "VacancyPriorEvaluator",
    "bootstrap_evaluator",
]
