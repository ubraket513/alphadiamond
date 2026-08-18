"""Graph policy/value models for two- and three-player Diamond."""

from .model_2p import DiamondModel2P
from .model_3p import DiamondModel3P

__all__ = ["DiamondModel2P", "DiamondModel3P"]
