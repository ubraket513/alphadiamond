"""Backward-compatible import for the model now officially named Soo."""

from .soo import SooModel

DiamondModel2P = SooModel

__all__ = ["DiamondModel2P", "SooModel"]
