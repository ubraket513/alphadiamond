"""Fixed-topology directional graph trunk using standard PyTorch operations."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...game.board import Board, standard_board
from ...game.coordinates import NUM_DIRECTIONS
from ..config import NetworkConfig


def directional_adjacency(board: Board) -> Tensor:
    """Return ``[direction, node, neighbour]`` connectivity."""
    adjacency = torch.zeros(NUM_DIRECTIONS, len(board), len(board), dtype=torch.float32)
    for node in range(len(board)):
        for direction, neighbour in enumerate(board.neighbours(node)):
            if neighbour is not None:
                adjacency[direction, node, neighbour] = 1.0
    return adjacency


class DirectionalResidualBlock(nn.Module):
    """``nodes + GELU(LayerNorm(message))``, optionally behind a scalar gate.

    ``gated`` adds a single zero-initialised ``alpha`` after the activation, so
    the block starts as an exact identity and opens as training decides.  This
    is ReZero's ``x + alpha * F(x)``, and the placement is the point: zeroing
    the *LayerNorm* scale instead also yields an identity, but that parameter
    sits **inside** ``F`` and before a nonlinearity, so it rescales the branch's
    representation rather than its amplitude -- ``GELU(gamma * z)`` is not
    ``gamma * GELU(z)``.  A gate that reads as "how much of this branch do I
    want" has to be outside the activation, and only then does its sign and
    magnitude mean what one would expect.

    Off by default: the parameter is absent entirely when ``gated`` is false, so
    every existing checkpoint keeps its exact ``state_dict``.
    """

    def __init__(self, width: int, *, gated: bool = False) -> None:
        super().__init__()
        self.self_projection = nn.Linear(width, width)
        self.direction_projections = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in range(NUM_DIRECTIONS)
        )
        self.norm = nn.LayerNorm(width)
        self.activation = nn.GELU()
        if gated:
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("alpha", None)

    def forward(self, nodes: Tensor, adjacency: Tensor) -> Tensor:
        # Contract over directions in two einsums rather than looping in Python.
        #
        # The quantity is unchanged --
        #     message = self_projection(nodes) + sum_d (adjacency[d] @ nodes) @ W_d^T
        # -- this only reassociates the sum, so it is exact up to FP32 ordering
        # (measured worst-case 1.8e-06 against the loop on the trained weights).
        # The parameters are untouched, so the state_dict and every checkpoint
        # stay byte-compatible.
        #
        # Why it matters: self-play inference on this pipeline is CPU
        # kernel-launch bound, not GPU bound.  The forward costs the same for
        # batch 1 as for batch 256, and after 16 queued forwards the GPU drains
        # in 22 us -- the parent thread is spending its time issuing operations,
        # not waiting for them.  The loop issued an index, a matmul, a Linear and
        # an add per direction, ~24 operations per block and ~144 across the
        # trunk.  See docs/rtx5060_bottleneck_findings.md.
        neighbours = torch.einsum("dij,bjw->bdiw", adjacency, nodes)
        weights = torch.stack([projection.weight for projection in self.direction_projections])
        message = self.self_projection(nodes) + torch.einsum("bdiw,dvw->biv", neighbours, weights)
        branch = self.activation(self.norm(message))
        if self.alpha is not None:
            branch = self.alpha * branch
        return nodes + branch


class DiamondGraphTrunk(nn.Module):
    def __init__(
        self,
        input_features: int,
        config: NetworkConfig,
        board: Board | None = None,
        *,
        gate_blocks_from: int | None = None,
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
        # Blocks at or after `gate_blocks_from` carry a residual gate.  The
        # index, rather than a bool, is what a depth transplant needs: the
        # inherited blocks must keep their exact parameter set or the parent's
        # weights no longer load.
        if gate_blocks_from is not None and not 0 <= gate_blocks_from <= config.residual_blocks:
            raise ValueError("gate_blocks_from must index a block, or equal the block count")
        self.gate_blocks_from = gate_blocks_from
        self.blocks = nn.ModuleList(
            DirectionalResidualBlock(
                config.width,
                gated=gate_blocks_from is not None and index >= gate_blocks_from,
            )
            for index in range(config.residual_blocks)
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
