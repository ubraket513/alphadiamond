"""Batch-shaped evaluator API consumed by MCTS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

EvalValue: TypeAlias = float | tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EvalRequest:
    node_features: tuple[tuple[float, ...], ...]
    legal_action_ids: tuple[int, ...]
    canonical_player_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EvalResult:
    priors: dict[int, float]
    value: EvalValue


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(self, requests: tuple[EvalRequest, ...]) -> tuple[EvalResult, ...]: ...


__all__ = ["EvalRequest", "EvalResult", "EvalValue", "Evaluator"]
