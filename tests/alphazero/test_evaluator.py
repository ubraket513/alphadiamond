from __future__ import annotations

import math

import pytest
import torch

from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.evaluator.base import EvalRequest
from diamond.alphazero.evaluator.dummy import DummyEvaluator
from diamond.alphazero.evaluator.torch import TorchEvaluator
from diamond.alphazero.network import SooModel


def request(legal: tuple[int, ...] = (7, 19, 100)) -> EvalRequest:
    return EvalRequest(
        node_features=tuple((0.0, 0.0, 0.0, 0.0) for _ in range(73)),
        legal_action_ids=legal,
        canonical_player_ids=(1, 2),
    )


def test_dummy_evaluator_is_deterministic_and_legal_only() -> None:
    evaluator = DummyEvaluator(value=0.25)

    first = evaluator.evaluate((request(),))[0]
    second = evaluator.evaluate((request(),))[0]

    assert first == second
    assert first.priors == {7: pytest.approx(1 / 3), 19: pytest.approx(1 / 3), 100: pytest.approx(1 / 3)}
    assert first.value == 0.25


def test_torch_evaluator_masks_and_normalizes_legal_actions() -> None:
    model = SooModel(NetworkConfig(width=16, residual_blocks=1))
    evaluator = TorchEvaluator(model, value_size=1, device="cpu")

    result = evaluator.evaluate((request(),))[0]

    assert set(result.priors) == {7, 19, 100}
    assert sum(result.priors.values()) == pytest.approx(1.0)
    assert all(math.isfinite(probability) and probability > 0 for probability in result.priors.values())
    assert isinstance(result.value, float)
    assert not any(parameter.grad is not None for parameter in model.parameters())
    assert not model.training


def test_evaluators_reject_requests_without_legal_actions() -> None:
    empty = request(())
    with pytest.raises(ValueError, match="legal"):
        DummyEvaluator().evaluate((empty,))
    with pytest.raises(ValueError, match="legal"):
        TorchEvaluator(
            SooModel(NetworkConfig(width=16, residual_blocks=1)),
            value_size=1,
        ).evaluate((empty,))


def test_torch_evaluator_validates_model_value_shape() -> None:
    class BadModel(torch.nn.Module):
        def forward(self, features):
            return torch.zeros(features.shape[0], 5329), torch.zeros(features.shape[0], 2)

    with pytest.raises(ValueError, match="value shape"):
        TorchEvaluator(BadModel(), value_size=1).evaluate((request(),))


def _ragged_requests() -> tuple[EvalRequest, ...]:
    """A spread of legal-action counts, including the single-action edge case."""
    legal_sets = ((7,), (7, 19), (0, 1, 2, 3), tuple(range(100, 140)), (5328,), (0, 5328))
    return tuple(
        EvalRequest(
            node_features=tuple(
                tuple(float((index + row) % 3) for _ in range(4)) for row in range(73)
            ),
            legal_action_ids=legal,
            canonical_player_ids=(1, 2),
        )
        for index, legal in enumerate(legal_sets)
    )


def _cpu_evaluator() -> TorchEvaluator:
    torch.manual_seed(4321)
    return TorchEvaluator(
        SooModel(NetworkConfig(width=16, residual_blocks=1)), value_size=1, device="cpu"
    )


def test_torch_evaluator_batches_ragged_legal_action_sets_like_single_requests() -> None:
    """Padding a batch to its widest legal set must not perturb the narrow rows."""
    evaluator = _cpu_evaluator()
    requests = _ragged_requests()

    batched = evaluator.evaluate(requests)
    one_at_a_time = tuple(evaluator.evaluate((request,))[0] for request in requests)

    for expected, actual in zip(one_at_a_time, batched, strict=True):
        assert set(expected.priors) == set(actual.priors)
        for action, probability in expected.priors.items():
            assert actual.priors[action] == pytest.approx(probability, rel=1e-5, abs=1e-6)
        assert actual.value == pytest.approx(expected.value, rel=1e-5, abs=1e-6)
        assert sum(actual.priors.values()) == pytest.approx(1.0)


def test_torch_evaluator_rejects_an_out_of_range_action_in_any_row() -> None:
    """The offending row is not the first one; a per-row check must not be the guard."""
    evaluator = _cpu_evaluator()
    requests = (request((7, 19)), request((7, 999_999)), request((3,)))

    with pytest.raises(ValueError, match="outside the model policy space"):
        evaluator.evaluate(requests)

    with pytest.raises(ValueError, match="outside the model policy space"):
        evaluator.evaluate((request((7, 19)), request((-1, 7))))


def test_torch_evaluator_rejects_non_finite_policy_logits() -> None:
    class NanPolicyModel(torch.nn.Module):
        def forward(self, features):
            logits = torch.zeros(features.shape[0], 5329)
            logits[-1, 19] = float("nan")
            return logits, torch.zeros(features.shape[0], 1)

    with pytest.raises(ValueError, match="invalid legal policy probabilities"):
        TorchEvaluator(NanPolicyModel(), value_size=1).evaluate(
            (request((7, 19)), request((7, 19)))
        )


def test_torch_evaluator_rejects_a_non_finite_value() -> None:
    class NanValueModel(torch.nn.Module):
        def forward(self, features):
            values = torch.zeros(features.shape[0], 1)
            values[-1, 0] = float("inf")
            return torch.zeros(features.shape[0], 5329), values

    with pytest.raises(ValueError, match="non-finite value"):
        TorchEvaluator(NanValueModel(), value_size=1).evaluate(
            (request((7, 19)), request((7, 19)))
        )


def test_torch_evaluator_device_transfers_do_not_scale_with_batch_size(monkeypatch) -> None:
    """The per-row sync loop measured at 69% of a batch-30 GPU evaluation.

    Counted structurally rather than timed: a host transfer per row is the defect,
    so the transfer count must be constant in the batch size, not proportional.
    """
    evaluator = _cpu_evaluator()
    transfers = {"count": 0}
    original = torch.Tensor.cpu

    def counting_cpu(self, *args, **kwargs):
        transfers["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)

    def transfers_for(count: int) -> int:
        transfers["count"] = 0
        evaluator.evaluate(tuple(request((7, 19, 100)) for _ in range(count)))
        return transfers["count"]

    small, large = transfers_for(2), transfers_for(24)

    assert small == large, f"transfers scale with batch size: {small} -> {large}"
    assert large <= 4
