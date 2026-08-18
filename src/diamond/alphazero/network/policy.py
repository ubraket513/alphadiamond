"""Factorized source/destination policy head."""

from __future__ import annotations

import math

from torch import Tensor, nn


class SourceDestinationPolicyHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.source = nn.Linear(width, width)
        self.destination = nn.Linear(width, width)
        self.scale = math.sqrt(width)

    def forward(self, nodes: Tensor) -> Tensor:
        source = self.source(nodes)
        destination = self.destination(nodes)
        logits = source @ destination.transpose(-1, -2) / self.scale
        return logits.flatten(start_dim=1)


__all__ = ["SourceDestinationPolicyHead"]
