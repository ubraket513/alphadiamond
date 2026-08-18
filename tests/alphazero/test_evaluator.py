from __future__ import annotations

import math

import pytest
import torch

from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.evaluator.base import EvalRequest
from diamond.alphazero.evaluator.dummy import DummyEvaluator
from diamond.alphazero.evaluator.torch import TorchEvaluator
from diamond.alphazero.network.model_2p import DiamondModel2P


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
    model = DiamondModel2P(NetworkConfig(width=16, residual_blocks=1))
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
            DiamondModel2P(NetworkConfig(width=16, residual_blocks=1)),
            value_size=1,
        ).evaluate((empty,))


def test_torch_evaluator_validates_model_value_shape() -> None:
    class BadModel(torch.nn.Module):
        def forward(self, features):
            return torch.zeros(features.shape[0], 5329), torch.zeros(features.shape[0], 2)

    with pytest.raises(ValueError, match="value shape"):
        TorchEvaluator(BadModel(), value_size=1).evaluate((request(),))
