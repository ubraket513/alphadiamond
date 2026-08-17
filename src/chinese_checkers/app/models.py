"""Qt list models exposed to QML.

These are pure view models: they hold no rules and compute no legality.  The
controller pushes already-decided data into them; QML only renders it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import (
    QAbstractListModel,
    QByteArray,
    QModelIndex,
    QObject,
    Qt,
    Signal,
    Slot,
)

from ..game.board import Board, Camp
from ..game.move import Move
from ..game.state import EMPTY, PlayerKind, PlayerSpec

_UNSET = QModelIndex()


def _roles(names: list[str], start: int = Qt.UserRole + 1) -> dict[int, QByteArray]:
    return {start + i: QByteArray(name.encode()) for i, name in enumerate(names)}


# --------------------------------------------------------------------------
# Board holes
# --------------------------------------------------------------------------

HOLE_ROLE_NAMES = [
    "positionId",
    "unitX",
    "unitY",
    "campKey",
    "occupant",
    "isSelected",
    "isLegalStep",
    "isLegalJump",
    "isPathNode",
    "pathIndex",
    "isLastMoveSource",
    "isLastMoveDest",
    "isProposalSource",
    "isProposalDest",
]


@dataclass
class _Hole:
    positionId: int
    unitX: float
    unitY: float
    campKey: str
    occupant: int = EMPTY
    isSelected: bool = False
    isLegalStep: bool = False
    isLegalJump: bool = False
    isPathNode: bool = False
    pathIndex: int = -1
    isLastMoveSource: bool = False
    isLastMoveDest: bool = False
    isProposalSource: bool = False
    isProposalDest: bool = False


class BoardModel(QAbstractListModel):
    """One row per playable hole (121 rows), with all render-time flags."""

    _ROLES = _roles(HOLE_ROLE_NAMES)

    def __init__(self, board: Board, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._holes: list[_Hole] = [
            _Hole(
                positionId=p.id,
                unitX=p.unit_xy()[0],
                unitY=p.unit_xy()[1],
                campKey=p.camp.value if p.camp else "",
            )
            for p in board.positions
        ]

    def rowCount(self, parent: QModelIndex = _UNSET) -> int:
        return 0 if parent.isValid() else len(self._holes)

    def roleNames(self) -> dict[int, QByteArray]:
        return dict(self._ROLES)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        name = self._ROLES.get(role)
        if name is None:
            return None
        return getattr(self._holes[index.row()], bytes(name).decode())

    # -- controller-facing updates ---------------------------------------
    def _refresh(self) -> None:
        self.dataChanged.emit(self.index(0, 0), self.index(len(self._holes) - 1, 0))

    def set_occupancy(self, occupancy: tuple[int, ...]) -> None:
        for hole, value in zip(self._holes, occupancy):
            hole.occupant = value
        self._refresh()

    def clear_interaction(self) -> None:
        for hole in self._holes:
            hole.isSelected = False
            hole.isLegalStep = False
            hole.isLegalJump = False
            hole.isPathNode = False
            hole.pathIndex = -1
            hole.isProposalSource = False
            hole.isProposalDest = False
        self._refresh()

    def set_selection(self, selected: int | None, steps: set[int], jumps: set[int]) -> None:
        for hole in self._holes:
            hole.isSelected = hole.positionId == selected
            hole.isLegalStep = hole.positionId in steps
            hole.isLegalJump = hole.positionId in jumps
        self._refresh()

    def set_proposal(self, move: Move | None) -> None:
        for hole in self._holes:
            hole.isPathNode = False
            hole.pathIndex = -1
            hole.isProposalSource = False
            hole.isProposalDest = False
        if move is not None:
            for order, pid in enumerate(move.path):
                hole = self._holes[pid]
                hole.isPathNode = True
                hole.pathIndex = order
            self._holes[move.source].isProposalSource = True
            self._holes[move.destination].isProposalDest = True
        self._refresh()

    def set_last_move(self, move: Move | None) -> None:
        for hole in self._holes:
            hole.isLastMoveSource = False
            hole.isLastMoveDest = False
        if move is not None:
            self._holes[move.source].isLastMoveSource = True
            self._holes[move.destination].isLastMoveDest = True
        self._refresh()


# --------------------------------------------------------------------------
# Pieces (separate from holes so they can animate independently of state)
# --------------------------------------------------------------------------

PIECE_ROLE_NAMES = ["pieceId", "playerId", "positionId", "unitX", "unitY", "color", "isMoving"]


@dataclass
class _Piece:
    pieceId: int
    playerId: int
    positionId: int
    unitX: float
    unitY: float
    color: str
    isMoving: bool = False


class PieceModel(QAbstractListModel):
    """One row per piece.  Rows keep their identity across moves so QML can
    animate a piece between holes instead of re-creating it."""

    _ROLES = _roles(PIECE_ROLE_NAMES)

    def __init__(
        self,
        board: Board,
        players: tuple[PlayerSpec, ...],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._board = board
        self._colors = {p.id: p.color for p in players}
        self._pieces: list[_Piece] = []

    def rowCount(self, parent: QModelIndex = _UNSET) -> int:
        return 0 if parent.isValid() else len(self._pieces)

    def roleNames(self) -> dict[int, QByteArray]:
        return dict(self._ROLES)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        name = self._ROLES.get(role)
        if name is None:
            return None
        return getattr(self._pieces[index.row()], bytes(name).decode())

    def row_at(self, position_id: int) -> int:
        for row, piece in enumerate(self._pieces):
            if piece.positionId == position_id:
                return row
        return -1

    def _place(self, piece: _Piece, position_id: int) -> None:
        x, y = self._board.position(position_id).unit_xy()
        piece.positionId = position_id
        piece.unitX = x
        piece.unitY = y

    def move_piece(self, row: int, position_id: int, *, moving: bool) -> None:
        if not 0 <= row < len(self._pieces):
            return
        piece = self._pieces[row]
        self._place(piece, position_id)
        piece.isMoving = moving
        idx = self.index(row, 0)
        self.dataChanged.emit(idx, idx)

    def rebuild(self, occupancy: tuple[int, ...]) -> None:
        """Resync every piece to ``occupancy``, reusing rows where possible.

        Rows that already sit on a still-occupied hole keep their identity, so
        an undo animates the one piece that actually moved instead of snapping
        the whole board.
        """
        by_player: dict[int, list[int]] = {}
        for pid, owner in enumerate(occupancy):
            if owner != EMPTY:
                by_player.setdefault(owner, []).append(pid)

        keep: list[_Piece] = []
        for piece in self._pieces:
            targets = by_player.get(piece.playerId)
            if targets and piece.positionId in targets:
                targets.remove(piece.positionId)
                keep.append(piece)

        leftovers = [p for p in self._pieces if p not in keep]
        for piece in leftovers:
            targets = by_player.get(piece.playerId) or []
            if targets:
                self._place(piece, targets.pop(0))
                piece.isMoving = False
                keep.append(piece)

        remaining = [(owner, pid) for owner, pids in by_player.items() for pid in pids]
        if remaining or len(keep) != len(self._pieces):
            self.beginResetModel()
            pieces = sorted(keep, key=lambda p: p.pieceId)
            next_id = (max((p.pieceId for p in pieces), default=-1)) + 1
            for owner, pid in remaining:
                x, y = self._board.position(pid).unit_xy()
                pieces.append(
                    _Piece(next_id, owner, pid, x, y, self._colors.get(owner, "#888888"))
                )
                next_id += 1
            self._pieces = pieces
            self.endResetModel()
        else:
            self._pieces = keep
            self._refresh()

    def _refresh(self) -> None:
        if self._pieces:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self._pieces) - 1, 0))


# --------------------------------------------------------------------------
# Move history
# --------------------------------------------------------------------------

HISTORY_ROLE_NAMES = [
    "turnNumber",
    "playerId",
    "playerLabel",
    "playerColor",
    "moveText",
    "pathText",
    "hopCount",
    "isAi",
]


class MoveHistoryModel(QAbstractListModel):
    _ROLES = _roles(HISTORY_ROLE_NAMES)

    def __init__(self, players: tuple[PlayerSpec, ...], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._players = {p.id: p for p in players}
        self._rows: list[dict[str, Any]] = []

    def rowCount(self, parent: QModelIndex = _UNSET) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def roleNames(self) -> dict[int, QByteArray]:
        return dict(self._ROLES)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        name = self._ROLES.get(role)
        return None if name is None else self._rows[index.row()].get(bytes(name).decode())

    def set_records(self, records: tuple) -> None:
        self.beginResetModel()
        self._rows = []
        for record in records:
            spec = self._players.get(record.player_id)
            is_ai = spec is not None and spec.kind is PlayerKind.AI
            self._rows.append(
                {
                    "turnNumber": record.turn_number,
                    "playerId": record.player_id,
                    "playerLabel": ("AI" if is_ai else f"P{record.player_id}"),
                    "playerColor": spec.color if spec else "#888888",
                    "moveText": record.move.short_text(),
                    "pathText": record.move.path_text(),
                    "hopCount": record.move.hop_count,
                    "isAi": is_ai,
                }
            )
        self.endResetModel()


# --------------------------------------------------------------------------
# Players
# --------------------------------------------------------------------------

PLAYER_ROLE_NAMES = [
    "playerId",
    "name",
    "kindLabel",
    "color",
    "isCurrent",
    "isAi",
    "homeCount",
    "hasFinished",
]


class PlayerModel(QAbstractListModel):
    _ROLES = _roles(PLAYER_ROLE_NAMES)

    def __init__(
        self,
        board: Board,
        players: tuple[PlayerSpec, ...],
        agent_names: dict[int, str],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._board = board
        self._specs = players
        self._agent_names = agent_names
        self._rows: list[dict[str, Any]] = [
            {
                "playerId": p.id,
                "name": p.name,
                "kindLabel": (
                    agent_names.get(p.id, "Agent") if p.kind is PlayerKind.AI else "Human"
                ),
                "color": p.color,
                "isCurrent": False,
                "isAi": p.kind is PlayerKind.AI,
                "homeCount": 0,
                "hasFinished": False,
            }
            for p in players
        ]

    def rowCount(self, parent: QModelIndex = _UNSET) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def roleNames(self) -> dict[int, QByteArray]:
        return dict(self._ROLES)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        name = self._ROLES.get(role)
        return None if name is None else self._rows[index.row()].get(bytes(name).decode())

    def update(self, occupancy: tuple[int, ...], current_player_id: int) -> None:
        for row, spec in zip(self._rows, self._specs):
            target = self._board.camp_positions(spec.target_camp)
            home = sum(1 for pid in target if occupancy[pid] == spec.id)
            row["isCurrent"] = spec.id == current_player_id
            row["homeCount"] = home
            row["hasFinished"] = home == len(target)
        if self._rows:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self._rows) - 1, 0))


# --------------------------------------------------------------------------
# Static board geometry for the renderer
# --------------------------------------------------------------------------


class BoardGeometry(QObject):
    """Constant lattice geometry in abstract units, computed from the engine.

    QML fits ``bounds`` into the available space and scales; no pixel position
    is ever hard-coded, and the board stays sharp at any resolution.
    """

    changed = Signal()

    def __init__(
        self,
        board: Board,
        players: tuple[PlayerSpec, ...],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        min_x, min_y, max_x, max_y = board.unit_bounds()
        self._bounds = {"minX": min_x, "minY": min_y, "maxX": max_x, "maxY": max_y}
        self._edges = [
            {
                "x1": board.position(a).unit_xy()[0],
                "y1": board.position(a).unit_xy()[1],
                "x2": board.position(b).unit_xy()[0],
                "y2": board.position(b).unit_xy()[1],
            }
            for a, b in board.edges
        ]

        colors: dict[Camp, str] = {}
        for spec in players:
            colors[spec.camp] = spec.color
            colors[spec.target_camp] = spec.color

        self._camps = []
        for camp in Camp:
            corners = board.camp_corners(camp)
            self._camps.append(
                {
                    "key": camp.value,
                    "color": colors.get(camp, "#C9CCD1"),
                    "points": [
                        {"x": board.position(c).unit_xy()[0], "y": board.position(c).unit_xy()[1]}
                        for c in corners
                    ],
                }
            )

        self._holes = [
            {"id": p.id, "x": p.unit_xy()[0], "y": p.unit_xy()[1]} for p in board.positions
        ]

    @Slot(result="QVariantMap")
    def bounds(self) -> dict:
        return self._bounds

    @Slot(result="QVariantList")
    def holes(self) -> list:
        return self._holes

    @Slot(result="QVariantList")
    def edges(self) -> list:
        return self._edges

    @Slot(result="QVariantList")
    def camps(self) -> list:
        return self._camps
