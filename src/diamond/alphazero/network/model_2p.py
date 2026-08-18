"""Two-player model with scalar current-player value semantics."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .policy import SourceDestinationPolicyHead
from .trunk import DiamondGraphTrunk
from ..config import NetworkConfig


class DiamondModel2P(nn.Module):
    input_features = 4
    value_size = 1

    def __init__(self, config: NetworkConfig = NetworkConfig()) -> None:
        super().__init__()
        self.config = config
        self.trunk = DiamondGraphTrunk(self.input_features, config)
        self.policy_head = SourceDestinationPolicyHead(config.width)
        self.value_head = nn.Sequential(
            nn.Linear(config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, 1),
            nn.Tanh(),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        nodes = self.trunk(features)
        return self.policy_head(nodes), self.value_head(nodes.mean(dim=1))


__all__ = ["DiamondModel2P"]
