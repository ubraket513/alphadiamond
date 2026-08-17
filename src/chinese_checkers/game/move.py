"""Move representation shared by the engine, the agents and the UI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MoveKind(Enum):
    STEP = "step"
    """A slide into an adjacent empty hole."""

    JUMP = "jump"
    """One or more chained jumps over occupied holes."""


@dataclass(frozen=True, slots=True)
class Move:
    """A single complete turn for one player.

    ``path`` always includes both endpoints:

    * single step   -> ``(12, 13)``
    * multi-hop     -> ``(37, 49, 63, 77, 91)``
    """

    player_id: int
    source: int
    destination: int
    path: tuple[int, ...]
    kind: MoveKind = MoveKind.STEP

    def __post_init__(self) -> None:
        if len(self.path) < 2:
            raise ValueError("a move path needs at least two holes")
        if self.path[0] != self.source or self.path[-1] != self.destination:
            raise ValueError("path endpoints must match source/destination")

    @property
    def hop_count(self) -> int:
        return len(self.path) - 1

    @property
    def is_multi_hop(self) -> bool:
        return self.hop_count > 1

    def path_text(self, separator: str = " → ") -> str:
        return separator.join(str(p) for p in self.path)

    def short_text(self) -> str:
        return f"{self.source} → {self.destination}"

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "source": self.source,
            "destination": self.destination,
            "path": list(self.path),
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Move":
        return cls(
            player_id=int(data["player_id"]),
            source=int(data["source"]),
            destination=int(data["destination"]),
            path=tuple(int(p) for p in data["path"]),
            kind=MoveKind(data.get("kind", MoveKind.STEP.value)),
        )
