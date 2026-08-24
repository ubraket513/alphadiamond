from __future__ import annotations

import pytest
import torch

from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.identity import ModelIdentity
from diamond.alphazero.network import MinModel, SooModel
from diamond.alphazero.network.trunk import (
    DirectionalResidualBlock,
    directional_adjacency,
)
from diamond.contract.board import standard_board
from diamond.contract.coordinates import NUM_DIRECTIONS


@pytest.mark.parametrize("batch_size", [1, 4])
def test_two_player_model_shapes_and_gradients(batch_size: int) -> None:
    model = SooModel(NetworkConfig(width=32, residual_blocks=2))
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
    model = MinModel(NetworkConfig(width=32, residual_blocks=2))
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
    model = SooModel(NetworkConfig(width=16, residual_blocks=1))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 72, 4))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 73, 6))


def test_models_expose_distinct_checkpoint_identities() -> None:
    soo = SooModel(model_version="0.3.0")
    min_model = MinModel(model_version="2.1.0")

    assert soo.identity == ModelIdentity.soo("0.3.0")
    assert min_model.identity == ModelIdentity.min("2.1.0")


def _loop_reference(block, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    """The per-direction Python loop the block's forward replaced.

    Kept here rather than in the module so the shipped forward carries no dead
    code, while the equivalence it claims stays pinned by a test.
    """
    message = block.self_projection(nodes)
    for direction, projection in enumerate(block.direction_projections):
        neighbours = torch.matmul(adjacency[direction], nodes)
        message = message + projection(neighbours)
    return nodes + block.activation(block.norm(message))


@pytest.mark.parametrize("batch_size", [1, 5, 17])
def test_directional_block_matches_the_per_direction_loop(batch_size: int) -> None:
    """The einsum contraction is a reassociation, not a different computation.

    The forward was vectorized because self-play inference is CPU
    kernel-launch bound; that optimization is only legitimate if it computes
    exactly what the loop did, so the loop is retained here as the oracle.
    """
    torch.manual_seed(7)
    board = standard_board()
    adjacency = directional_adjacency(board)
    block = DirectionalResidualBlock(width=16).eval()
    nodes = torch.randn(batch_size, len(board), 16)

    with torch.no_grad():
        expected = _loop_reference(block, nodes, adjacency)
        actual = block(nodes, adjacency)

    # FP32 summation order differs, so this is a tolerance, not equality.
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_directional_block_keeps_its_per_direction_parameters() -> None:
    """Vectorizing must not change the state_dict; checkpoints must stay loadable."""
    block = DirectionalResidualBlock(width=8)

    keys = set(block.state_dict())

    assert keys == {
        "self_projection.weight",
        "self_projection.bias",
        "norm.weight",
        "norm.bias",
        *{f"direction_projections.{index}.weight" for index in range(NUM_DIRECTIONS)},
    }


def test_directional_block_still_propagates_gradients_to_every_direction() -> None:
    """The stacked contraction must not detach any direction's weight."""
    board = standard_board()
    adjacency = directional_adjacency(board)
    block = DirectionalResidualBlock(width=8)
    nodes = torch.randn(2, len(board), 8)

    block(nodes, adjacency).mean().backward()

    assert all(
        projection.weight.grad is not None and torch.isfinite(projection.weight.grad).all()
        for projection in block.direction_projections
    )
