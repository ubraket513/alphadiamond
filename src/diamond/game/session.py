"""Headless authoritative game: state + history + commit/undo + save/load.

This is the Python reference game used by training and native parity checks. It
imports no UI toolkit, so the entire rule set is testable without a display.
The production native controller implements the matching contract in C++.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .board import Board, Camp, standard_board
from .history import MoveRecord, utc_now_iso
from .move import Move
from .rules import legal_moves, moves_from, update_ranking, validate_move
from .state import (
    DEFAULT_PLAYERS,
    GameState,
    GameStatus,
    PlayerKind,
    PlayerSpec,
    initial_state,
    next_player_id,
    player_by_id,
)

SCHEMA_VERSION = 2
"""2 added ``finish_order`` and made the seat list variable (2- or 3-player)."""


class GameSession:
    """Owns the authoritative :class:`GameState` and its history."""

    def __init__(
        self,
        players: tuple[PlayerSpec, ...] = DEFAULT_PLAYERS,
        board: Board | None = None,
        initial: GameState | None = None,
    ) -> None:
        self._board = board or standard_board()
        self._players = players
        # `initial` starts the session from a set-up position instead of the
        # standard opening; New Game returns here, not to the standard opening.
        self._initial = initial if initial is not None else initial_state(players, self._board)
        self._state = self._initial
        self._history: list[MoveRecord] = []

    # -- accessors -------------------------------------------------------
    @property
    def board(self) -> Board:
        return self._board

    @property
    def players(self) -> tuple[PlayerSpec, ...]:
        return self._players

    @property
    def state(self) -> GameState:
        return self._state

    @property
    def history(self) -> tuple[MoveRecord, ...]:
        return tuple(self._history)

    @property
    def is_over(self) -> bool:
        return self._state.status is GameStatus.FINISHED

    def current_player(self) -> PlayerSpec:
        return player_by_id(self._players, self._state.current_player_id)

    def legal_moves(self, player_id: int | None = None) -> tuple[Move, ...]:
        return legal_moves(self._board, self._state, player_id)

    def moves_from(self, source: int) -> dict[int, Move]:
        return moves_from(
            self._board, self._state, source, player_id=self._state.current_player_id
        )

    # -- mutation --------------------------------------------------------
    def reset(self) -> None:
        self._state = self._initial
        self._history.clear()

    def commit(self, move: Move, metadata: dict[str, Any] | None = None) -> MoveRecord:
        """Validate and apply ``move``; this is the only way state advances."""
        if self.is_over:
            raise RuntimeError("the game is already over")
        validate_move(self._board, self._state, move)

        turn_number = self._state.turn_number
        # Rank first, then hand over: a player who just got home drops out of
        # the rotation, so the next seat must be chosen against the new podium.
        new_state = self._state.apply(
            move,
            next_player_id=move.player_id,  # placeholder, corrected below
            advance_turn=True,
        )
        new_state = update_ranking(self._board, new_state, self._players)
        new_state = replace(
            new_state,
            current_player_id=next_player_id(
                self._players, move.player_id, skip=new_state.finish_order
            ),
        )

        record = MoveRecord(
            turn_number=turn_number,
            player_id=move.player_id,
            move=move,
            timestamp=utc_now_iso(),
            state_after=new_state,
            metadata=dict(metadata or {}),
        )
        self._state = new_state
        self._history.append(record)
        return record

    def can_undo(self) -> bool:
        return bool(self._history)

    def undo(self) -> MoveRecord | None:
        """Roll back the last committed move by restoring the prior snapshot."""
        if not self._history:
            return None
        removed = self._history.pop()
        self._state = self._history[-1].state_after if self._history else self._initial
        return removed

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "kind": p.kind.value,
                    "camp": p.camp.value,
                    "target_camp": p.target_camp.value,
                    "color": p.color,
                }
                for p in self._players
            ],
            "occupancy": list(self._state.occupancy),
            "current_player_id": self._state.current_player_id,
            "turn_number": self._state.turn_number,
            "status": self._state.status.value,
            "finish_order": list(self._state.finish_order),
            "history": [r.to_dict() for r in self._history],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def load_dict(self, data: dict) -> None:
        """Restore state by replaying the saved history, then cross-checking.

        Replaying rebuilds the per-move snapshots that undo needs, and the final
        cross-check against the saved occupancy catches a corrupted file.
        """
        version = int(data.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported save schema version: {version}")

        # The seat list is part of the save: a 2-player match must come back as
        # one, in its original turn order, whatever the session was set up with.
        self._players = tuple(
            PlayerSpec(
                id=int(entry["id"]),
                name=entry["name"],
                kind=PlayerKind(entry["kind"]),
                camp=Camp(entry["camp"]),
                target_camp=Camp(entry["target_camp"]),
                color=entry["color"],
            )
            for entry in data["players"]
        )

        # A save file always encodes a match played from the standard opening.
        self._initial = initial_state(self._players, self._board)
        self.reset()
        for entry in data.get("history", []):
            move = Move.from_dict(entry["move"])
            self.commit(move, metadata=entry.get("metadata") or {})
            # keep the original timestamp rather than the replay time
            last = self._history[-1]
            self._history[-1] = MoveRecord(
                turn_number=last.turn_number,
                player_id=last.player_id,
                move=last.move,
                timestamp=entry.get("timestamp", last.timestamp),
                state_after=last.state_after,
                metadata=last.metadata,
            )

        expected = tuple(int(v) for v in data["occupancy"])
        if self._state.occupancy != expected:
            raise ValueError("saved board does not match the replayed move history")
        if self._state.current_player_id != int(data["current_player_id"]):
            raise ValueError("saved current player does not match the replayed move history")
        if list(self._state.finish_order) != [int(v) for v in data.get("finish_order", [])]:
            raise ValueError("saved finishing order does not match the replayed move history")

    def load(self, path: str | Path) -> None:
        self.load_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = ["GameSession", "SCHEMA_VERSION"]
