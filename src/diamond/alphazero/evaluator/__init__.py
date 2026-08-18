"""Framework-neutral evaluator contract and implementations."""

from .base import EvalRequest, EvalResult, Evaluator
from .dummy import DummyEvaluator

__all__ = ["DummyEvaluator", "EvalRequest", "EvalResult", "Evaluator"]
