"""Shared episode records; value semantics stay in the named runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..deadline import MAX_GAME_TIME_EXCEEDED
from ..evaluator.base import EvalRequest
from ..replay import TrainingSample


class SelfPlayGame(Protocol):
    def initial_state(self) -> Any: ...
    def current_player_id(self, state: Any) -> int: ...
    def legal_action_ids(self, state: Any) -> tuple[int, ...]: ...
    def apply_action(self, state: Any, action_id: int) -> Any: ...
    def is_terminal(self, state: Any) -> bool: ...
    def evaluation_request(self, state: Any) -> EvalRequest: ...
    def final_order(self, state: Any) -> tuple[int, ...]: ...


@dataclass(frozen=True, slots=True)
class PendingSample:
    request: EvalRequest
    sparse_policy: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class SelfPlayEpisode:
    samples: tuple[TrainingSample, ...]
    final_order: tuple[int, ...] | None
    move_count: int
    completed: bool
    aborted_reason: str | None = None


def sparse_policy(policy: dict[int, float]) -> tuple[tuple[int, float], ...]:
    return tuple(sorted((action, probability) for action, probability in policy.items() if probability > 0))


MAX_GAME_MOVES_EXCEEDED = "max_game_moves_exceeded"
"""The episode ran out of moves, as distinct from running out of time."""


__all__ = [
    "MAX_GAME_MOVES_EXCEEDED",
    "MAX_GAME_TIME_EXCEEDED",
    "PendingSample",
    "SelfPlayEpisode",
    "SelfPlayGame",
    "sparse_policy",
]
