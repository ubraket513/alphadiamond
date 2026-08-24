"""The agent boundary the GUI talks to.

Everything above this line (controller, QML) knows only :class:`Agent`,
:class:`MoveRequest` and :class:`MoveProposal`.  Swapping ``RandomAgent`` for a
future ``AlphaZeroAgent`` (MCTS + OpenVINO) means registering a different
object here and changing nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..contract.board import Board
from ..contract.move import Move, MoveKind
from ..contract.state import GameState


@dataclass(frozen=True, slots=True)
class MoveRequest:
    """Everything an agent needs to pick a move, and nothing more.

    ``legal_moves`` is precomputed by the engine so agents never re-implement
    the rules.  ``avoid`` carries moves the controller would rather not see
    again (used by "Think Again"); an agent may ignore it when it has no choice.
    """

    board: Board
    state: GameState
    legal_moves: tuple[Move, ...]
    avoid: tuple[Move, ...] = ()
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class MoveProposal:
    """An agent's suggestion.  Never applied to game state by itself.

    ``metadata`` is free-form and purely informational.  The UI must render
    correctly when it is empty, and must never invent values that are not in
    here.  A future AlphaZero agent can add ``search_time_ms``,
    ``simulation_count``, ``value``, ``visit_count``, ``policy_probability``
    without any change to this class.
    """

    player_id: int
    source: int
    destination: int
    path: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_move(self, kind: MoveKind | None = None) -> Move:
        resolved = kind if kind is not None else (
            MoveKind.STEP if len(self.path) == 2 else MoveKind.JUMP
        )
        return Move(
            player_id=self.player_id,
            source=self.source,
            destination=self.destination,
            path=self.path,
            kind=resolved,
        )

    @classmethod
    def from_move(cls, move: Move, metadata: dict[str, Any] | None = None) -> MoveProposal:
        return cls(
            player_id=move.player_id,
            source=move.source,
            destination=move.destination,
            path=move.path,
            metadata=dict(metadata or {}),
        )


@runtime_checkable
class Agent(Protocol):
    """A move-choosing strategy.  Implementations must be thread-safe enough to
    run on a worker thread and must not touch Qt or any GUI object."""

    @property
    def name(self) -> str:
        """Short display name, e.g. ``"RandomAgent"``."""
        ...

    def choose_move(self, request: MoveRequest) -> MoveProposal | None:
        """Return a proposal, or ``None`` when no legal move exists."""
        ...


class NoLegalMoveError(RuntimeError):
    """Raised when an agent is asked to move with no legal moves available."""
