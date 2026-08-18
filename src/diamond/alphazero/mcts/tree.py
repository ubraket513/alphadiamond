"""Small search-tree records with explicit value semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ScalarEdge:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


@dataclass(slots=True)
class VectorEdge:
    prior: float
    player_ids: tuple[int, ...]
    visit_count: int = 0
    value_sum: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.value_sum:
            self.value_sum = {player_id: 0.0 for player_id in self.player_ids}

    def q(self, player_id: int) -> float:
        if player_id not in self.value_sum:
            raise ValueError(f"value vector has no component for player {player_id}")
        return self.value_sum[player_id] / self.visit_count if self.visit_count else 0.0

    def q_vector(self) -> dict[int, float]:
        return {player_id: self.q(player_id) for player_id in self.player_ids}


@dataclass(slots=True)
class ScalarNode:
    state: Any
    player_id: int
    edges: dict[int, ScalarEdge] = field(default_factory=dict)
    children: dict[int, "ScalarNode"] = field(default_factory=dict)
    expanded: bool = False


@dataclass(slots=True)
class VectorNode:
    state: Any
    player_id: int
    player_ids: tuple[int, ...]
    edges: dict[int, VectorEdge] = field(default_factory=dict)
    children: dict[int, "VectorNode"] = field(default_factory=dict)
    expanded: bool = False


__all__ = ["ScalarEdge", "ScalarNode", "VectorEdge", "VectorNode"]
