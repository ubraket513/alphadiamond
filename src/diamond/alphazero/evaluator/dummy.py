"""Deterministic evaluator for search and integration tests."""

from __future__ import annotations

from .base import EvalRequest, EvalResult, EvalValue


class DummyEvaluator:
    def __init__(self, value: EvalValue = 0.0) -> None:
        self.value = value

    def evaluate(self, requests: tuple[EvalRequest, ...]) -> tuple[EvalResult, ...]:
        results: list[EvalResult] = []
        for request in requests:
            if not request.legal_action_ids:
                raise ValueError("evaluation requires at least one legal action")
            probability = 1.0 / len(request.legal_action_ids)
            results.append(
                EvalResult(
                    priors={action: probability for action in request.legal_action_ids},
                    value=self.value,
                )
            )
        return tuple(results)


__all__ = ["DummyEvaluator"]
