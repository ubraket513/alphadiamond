"""Fixed-topology directional graph trunk using standard PyTorch operations."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import NetworkConfig
from ...game.board import Board, standard_board
from ...game.coordinates import NUM_DIRECTIONS


def directional_adjacency(board: Board) -> Tensor:
    """Return ``[direction, node, neighbour]`` connectivity."""
    adjacency = torch.zeros(NUM_DIRECTIONS, len(board), len(board), dtype=torch.float32)
    for node in range(len(board)):
        for direction, neighbour in enumerate(board.neighbours(node)):
            if neighbour is not None:
                adjacency[direction, node, neighbour] = 1.0
    return adjacency


class DirectionalResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_projection = nn.Linear(width, width)
        self.direction_projections = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in range(NUM_DIRECTIONS)
        )
        self.norm = nn.LayerNorm(width)
        self.activation = nn.GELU()

    def forward(self, nodes: Tensor, adjacency: Tensor) -> Tensor:
        message = self.self_projection(nodes)
        for direction, projection in enumerate(self.direction_projections):
            neighbours = torch.matmul(adjacency[direction], nodes)
            message = message + projection(neighbours)
        return nodes + self.activation(self.norm(message))


class DiamondGraphTrunk(nn.Module):
    def __init__(
        self,
        input_features: int,
        config: NetworkConfig,
        board: Board | None = None,
    ) -> None:
        super().__init__()
        if input_features <= 0:
            raise ValueError("input_features must be positive")
        if config.width <= 0 or config.residual_blocks <= 0:
            raise ValueError("network width and residual_blocks must be positive")
        self.board = board or standard_board()
        self.input_features = input_features
        self.width = config.width
        self.input_projection = nn.Linear(input_features, config.width)
        self.blocks = nn.ModuleList(
            DirectionalResidualBlock(config.width)
            for _ in range(config.residual_blocks)
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.register_buffer("adjacency", directional_adjacency(self.board))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, board, features]")
        if features.shape[1:] != (len(self.board), self.input_features):
            raise ValueError(
                f"expected features [B,{len(self.board)},{self.input_features}], "
                f"got {tuple(features.shape)}"
            )
        nodes = self.input_projection(features)
        for block in self.blocks:
            nodes = block(nodes, self.adjacency)
        return self.output_norm(nodes)


__all__ = ["DiamondGraphTrunk", "DirectionalResidualBlock", "directional_adjacency"]
