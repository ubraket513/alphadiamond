from __future__ import annotations

import pytest
import torch

from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.network.model_2p import DiamondModel2P
from diamond.alphazero.network.model_3p import DiamondModel3P


@pytest.mark.parametrize("batch_size", [1, 4])
def test_two_player_model_shapes_and_gradients(batch_size: int) -> None:
    model = DiamondModel2P(NetworkConfig(width=32, residual_blocks=2))
    features = torch.randn(batch_size, 73, 4)

    policy, value = model(features)
    (policy.mean() + value.mean()).backward()

    assert policy.shape == (batch_size, 5329)
    assert value.shape == (batch_size, 1)
    assert torch.isfinite(policy).all()
    assert torch.isfinite(value).all()
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("batch_size", [1, 3])
def test_three_player_model_shapes_and_gradients(batch_size: int) -> None:
    model = DiamondModel3P(NetworkConfig(width=32, residual_blocks=2))
    features = torch.randn(batch_size, 73, 6)

    policy, value = model(features)
    (policy.mean() + value.mean()).backward()

    assert policy.shape == (batch_size, 5329)
    assert value.shape == (batch_size, 3)
    assert torch.isfinite(policy).all()
    assert torch.isfinite(value).all()
    assert torch.all(value.abs() <= 1.0)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_model_rejects_wrong_board_or_feature_shape() -> None:
    model = DiamondModel2P(NetworkConfig(width=16, residual_blocks=1))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 72, 4))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 73, 6))

