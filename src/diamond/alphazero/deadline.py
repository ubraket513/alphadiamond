"""Monotonic wall-clock budget for one self-play game.

Standard library only, deliberately: spawn workers import this module, and the
worker import path is asserted to stay free of Torch.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic

MAX_GAME_TIME_EXCEEDED = "max_game_time_exceeded"
"""Abort reason for a game that outran its wall-clock budget.

Deliberately distinct from ``max_game_moves_exceeded``: one means the game was
too slow, the other that it was too long.  They call for different responses.
"""


@dataclass(frozen=True, slots=True)
class Deadline:
    """An elapsed-time budget measured against an injectable monotonic clock.

    The clock is a constructor argument so timeout behaviour can be tested
    deterministically instead of by waiting.
    """

    started_at: float
    budget_s: float
    clock: Callable[[], float] = field(default=monotonic)

    @classmethod
    def start(
        cls,
        budget_s: float | None,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> "Deadline | None":
        """Begin a budget now, or return ``None`` when none is configured.

        Returning ``None`` rather than an always-unexpired sentinel keeps the
        no-budget path free of any per-simulation clock reads.
        """
        if budget_s is None:
            return None
        if (
            not isinstance(budget_s, (int, float))
            or isinstance(budget_s, bool)
            or not math.isfinite(budget_s)
            or budget_s <= 0
        ):
            raise ValueError("deadline budget must be a positive finite number of seconds")
        return cls(started_at=clock(), budget_s=float(budget_s), clock=clock)

    @property
    def elapsed_s(self) -> float:
        return self.clock() - self.started_at

    @property
    def expired(self) -> bool:
        """True once the budget is reached; the boundary itself counts as expired."""
        return self.elapsed_s >= self.budget_s

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.budget_s - self.elapsed_s)


__all__ = ["MAX_GAME_TIME_EXCEEDED", "Deadline"]
