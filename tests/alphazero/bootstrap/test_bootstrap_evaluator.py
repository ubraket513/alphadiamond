"""The decorator replaces priors and nothing else."""

from __future__ import annotations

import math
import subprocess
import sys

import pytest

from diamond.alphazero.bootstrap.evaluator import BootstrapPriorEvaluator
from diamond.alphazero.evaluator.base import EvalRequest, EvalResult, Evaluator
from diamond.alphazero.evaluator.dummy import DummyEvaluator


def request(*actions: int) -> EvalRequest:
    return EvalRequest(
        node_features=((0.0, 0.0),),
        legal_action_ids=tuple(actions),
        canonical_player_ids=(0, 1),
    )


class RecordingEvaluator:
    """Distinct value and prior per request, so swaps and reorders show up."""

    def __init__(self, values) -> None:
        self.values = values
        self.seen: tuple[EvalRequest, ...] = ()

    def evaluate(self, requests: tuple[EvalRequest, ...]) -> tuple[EvalResult, ...]:
        self.seen = requests
        return tuple(
            EvalResult(
                priors={action: 1.0 / len(r.legal_action_ids) for action in r.legal_action_ids},
                value=value,
            )
            for r, value in zip(requests, self.values)
        )


def test_preserves_scalar_soo_value_exactly() -> None:
    base = RecordingEvaluator([0.3141592653589793])
    result = BootstrapPriorEvaluator(base).evaluate((request(1, 2),))
    assert result[0].value == 0.3141592653589793


def test_preserves_vector_min_value_exactly() -> None:
    base = RecordingEvaluator([(1.0, 0.0, -1.0)])
    result = BootstrapPriorEvaluator(base).evaluate((request(1, 2),))
    assert result[0].value == (1.0, 0.0, -1.0)


def test_replaces_priors_only() -> None:
    base = RecordingEvaluator([0.0])
    # From hole 36, hole 29 is closer to the target camp than hole 43.
    forward, backward = 73 * 36 + 29, 73 * 36 + 43
    actions = (forward, backward)
    result = BootstrapPriorEvaluator(base).evaluate((request(*actions),))
    assert tuple(sorted(result[0].priors)) == tuple(sorted(actions))
    # Uniform base priors are gone; the heuristic separates the two actions.
    assert result[0].priors[forward] > result[0].priors[backward]
    assert math.isclose(sum(result[0].priors.values()), 1.0, rel_tol=1e-12)


def test_preserves_batch_order() -> None:
    base = RecordingEvaluator([1.0, 2.0, 3.0])
    requests = (request(1, 2), request(3, 4), request(5, 6))
    results = BootstrapPriorEvaluator(base).evaluate(requests)
    assert [r.value for r in results] == [1.0, 2.0, 3.0]
    assert base.seen == requests


def test_passes_requests_through_untouched() -> None:
    base = RecordingEvaluator([0.0])
    requests = (request(7, 8),)
    BootstrapPriorEvaluator(base).evaluate(requests)
    assert base.seen is requests


def test_rejects_mismatched_result_count() -> None:
    class ShortEvaluator:
        def evaluate(self, requests):
            return ()

    with pytest.raises(ValueError, match="mismatched result count"):
        BootstrapPriorEvaluator(ShortEvaluator()).evaluate((request(1, 2),))


def test_rejects_empty_legal_actions() -> None:
    with pytest.raises(ValueError):
        BootstrapPriorEvaluator(RecordingEvaluator([0.0])).evaluate((request(),))


def test_rejects_base_without_evaluate() -> None:
    with pytest.raises(TypeError):
        BootstrapPriorEvaluator(object())


def test_works_over_the_deterministic_dummy_evaluator() -> None:
    wrapped = BootstrapPriorEvaluator(DummyEvaluator(value=-1.0))
    result = wrapped.evaluate((request(10, 20, 30),))
    assert result[0].value == -1.0
    assert math.isclose(sum(result[0].priors.values()), 1.0, rel_tol=1e-12)


def test_satisfies_the_public_evaluator_protocol() -> None:
    """The same protocol RemoteEvaluator is consumed through."""
    assert isinstance(BootstrapPriorEvaluator(DummyEvaluator()), Evaluator)


def test_imports_no_torch() -> None:
    code = (
        "import sys;"
        "import diamond.alphazero.bootstrap.evaluator as m;"
        "assert 'torch' not in sys.modules, sorted(k for k in sys.modules if 'torch' in k)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
