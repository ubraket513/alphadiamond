"""Versioned source-to-final-destination action identity.

Intermediate jump positions are deliberately absent.  The authoritative game
engine resolves the canonical path for a selected source and destination.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActionSpaceSpec:
    board_size: int
    version: str

    def __post_init__(self) -> None:
        if self.board_size <= 0:
            raise ValueError("board_size must be positive")
        if not self.version:
            raise ValueError("action-space version must not be empty")

    @property
    def action_size(self) -> int:
        return self.board_size * self.board_size

    @classmethod
    def diamond73(cls) -> "ActionSpaceSpec":
        return cls(board_size=73, version="diamond73-srcdst-v1")


class ActionCodec:
    """Bijective mapping between ``(source, destination)`` and an integer."""

    def __init__(self, spec: ActionSpaceSpec) -> None:
        self.spec = spec

    @property
    def board_size(self) -> int:
        return self.spec.board_size

    @property
    def action_size(self) -> int:
        return self.spec.action_size

    def encode(self, source: int, destination: int) -> int:
        self._validate_position(source, "source")
        self._validate_position(destination, "destination")
        return source * self.board_size + destination

    def decode(self, action_id: int) -> tuple[int, int]:
        if not isinstance(action_id, int) or isinstance(action_id, bool):
            raise TypeError("action_id must be an integer")
        if not 0 <= action_id < self.action_size:
            raise ValueError(f"action_id must be in [0, {self.action_size})")
        return divmod(action_id, self.board_size)

    def _validate_position(self, position_id: int, name: str) -> None:
        if not isinstance(position_id, int) or isinstance(position_id, bool):
            raise TypeError(f"{name} must be an integer")
        if not 0 <= position_id < self.board_size:
            raise ValueError(f"{name} must be in [0, {self.board_size})")


__all__ = ["ActionCodec", "ActionSpaceSpec"]
