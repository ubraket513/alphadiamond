"""Move history with full state snapshots, so undo is exact rather than inverse."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .move import Move
from .state import GameState


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class MoveRecord:
    """One committed move plus the state it produced.

    ``state_after`` makes undo a pop-and-restore rather than a move inversion,
    which keeps it correct even as rules grow.
    """

    turn_number: int
    player_id: int
    move: Move
    timestamp: str
    state_after: GameState
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "turn_number": self.turn_number,
            "player_id": self.player_id,
            "move": self.move.to_dict(),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
