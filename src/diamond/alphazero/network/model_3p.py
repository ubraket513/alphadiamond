"""Backward-compatible import for the model now officially named Min."""

from .min import MinModel

DiamondModel3P = MinModel

__all__ = ["DiamondModel3P", "MinModel"]
