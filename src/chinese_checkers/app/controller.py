"""GameController — the turn state machine and the only bridge QML may use.

Layering::

    QML  ->  GameController  ->  GameSession / rules
                     ^
                     |
                 AiWorker -> Agent

QML never computes legality and never applies a move; it calls slots here and
renders the properties and models this object publishes.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from PySide6.QtCore import Property, QObject, QTimer, Signal, Slot

from ..agents.base import Agent, MoveProposal, MoveRequest
from ..agents.random_agent import RandomAgent
from ..game.move import Move, MoveKind
from ..game.rules import IllegalMoveError, validate_move
from ..game.session import GameSession
from ..game.state import (
    DEFAULT_PLAYERS,
    EMPTY,
    GameState,
    PlayerKind,
    PlayerSpec,
    player_by_id,
)
from .ai_worker import AiWorker
from .models import BoardGeometry, BoardModel, MoveHistoryModel, PieceModel, PlayerModel

HOP_DURATION_MS = 140
"""Wall time per lattice hop while animating a committed move."""

DEFAULT_SAVE_DIR = Path.home() / ".alphadiamond" / "saves"


class Phase(StrEnum):
    """UI-relevant turn states.  Every enabled/disabled control derives from this."""

    WAITING_FOR_HUMAN = "WAITING_FOR_HUMAN_INPUT"
    HUMAN_MOVE_PROPOSED = "HUMAN_MOVE_PROPOSED"
    AI_THINKING = "AI_THINKING"
    AI_MOVE_PROPOSED = "AI_MOVE_PROPOSED"
    ANIMATING = "ANIMATING_MOVE"
    GAME_OVER = "GAME_OVER"


def _path_to_local(path_or_url: str) -> Path:
    if path_or_url.startswith("file:"):
        return Path(unquote(urlparse(path_or_url).path))
    return Path(path_or_url)


class GameController(QObject):
    changed = Signal()
    """Coarse notify signal; QML bindings re-evaluate on any controller change."""

    errorRaised = Signal(str)
    gameFinished = Signal(int)

    def __init__(
        self,
        players: tuple[PlayerSpec, ...] = DEFAULT_PLAYERS,
        agents: dict[int, Agent] | None = None,
        parent: QObject | None = None,
        thinking_delay_ms: int | None = None,
        animate: bool = True,
        initial_state: GameState | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = GameSession(players, initial=initial_state)
        self._players = players
        self._agents: dict[int, Agent] = agents or {
            spec.id: RandomAgent() for spec in players if spec.kind is PlayerKind.AI
        }
        self._animate = animate

        board = self._session.board
        agent_names = {pid: agent.name for pid, agent in self._agents.items()}
        self._board_model = BoardModel(board, self)
        self._piece_model = PieceModel(board, players, self)
        self._history_model = MoveHistoryModel(players, self)
        self._player_model = PlayerModel(board, players, agent_names, self)
        self._geometry = BoardGeometry(board, players, self)

        self._worker = AiWorker(self) if thinking_delay_ms is None else AiWorker(
            self, delay_ms=thinking_delay_ms
        )
        self._worker.proposalReady.connect(self._on_proposal_ready)
        self._worker.proposalFailed.connect(self._on_proposal_failed)

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(HOP_DURATION_MS)
        self._anim_timer.timeout.connect(self._on_animation_tick)
        self._anim_row = -1
        self._anim_path: tuple[int, ...] = ()
        self._anim_index = 0

        self._game_number = 1
        self._generation = 0
        self._phase = Phase.WAITING_FOR_HUMAN
        self._selected: int | None = None
        self._legal: dict[int, Move] = {}
        self._proposal: Move | None = None
        self._proposal_metadata: dict[str, Any] = {}
        self._proposal_is_ai = False
        self._ai_rejected: list[Move] = []
        self._ai_status = "Ready"
        self._status_message = ""
        self._error_message = ""
        self._last_move: Move | None = None

        self._sync_models()
        self._enter_turn()

    @property
    def session(self) -> GameSession:
        """The authoritative game. Read-only for callers; mutate via slots."""
        return self._session

    # ------------------------------------------------------------------
    # exposed models
    # ------------------------------------------------------------------
    @Property(QObject, constant=True)
    def boardModel(self) -> QObject:
        return self._board_model

    @Property(QObject, constant=True)
    def pieceModel(self) -> QObject:
        return self._piece_model

    @Property(QObject, constant=True)
    def historyModel(self) -> QObject:
        return self._history_model

    @Property(QObject, constant=True)
    def playerModel(self) -> QObject:
        return self._player_model

    @Property(QObject, constant=True)
    def geometry(self) -> QObject:
        return self._geometry

    # ------------------------------------------------------------------
    # scalar properties
    # ------------------------------------------------------------------
    def _get_phase(self) -> str:
        return str(self._phase)

    def _get_game_label(self) -> str:
        return f"Game #{self._game_number:03d}"

    def _get_turn_number(self) -> int:
        return self._session.state.turn_number

    def _get_current_player_id(self) -> int:
        return self._session.state.current_player_id

    def _current_spec(self) -> PlayerSpec:
        return player_by_id(self._players, self._session.state.current_player_id)

    def _get_current_player_name(self) -> str:
        return self._current_spec().name

    def _get_current_player_color(self) -> str:
        return self._current_spec().color

    def _get_is_current_ai(self) -> bool:
        return self._current_spec().kind is PlayerKind.AI

    def _get_status_message(self) -> str:
        return self._status_message

    def _get_error_message(self) -> str:
        return self._error_message

    def _get_is_game_over(self) -> bool:
        return self._phase is Phase.GAME_OVER

    def _get_winner_id(self) -> int:
        return self._session.state.winner_id or 0

    def _get_winner_name(self) -> str:
        winner = self._session.state.winner_id
        return player_by_id(self._players, winner).name if winner else ""

    def _get_can_undo(self) -> bool:
        return self._session.can_undo() and self._phase is not Phase.ANIMATING

    def _get_has_proposal(self) -> bool:
        return self._proposal is not None

    def _get_can_confirm(self) -> bool:
        return self._proposal is not None and self._phase in (
            Phase.HUMAN_MOVE_PROPOSED,
            Phase.AI_MOVE_PROPOSED,
        )

    def _get_can_cancel(self) -> bool:
        return self._phase is Phase.HUMAN_MOVE_PROPOSED or self._selected is not None

    def _get_can_select(self) -> bool:
        return self._phase is Phase.WAITING_FOR_HUMAN

    def _get_proposal_is_ai(self) -> bool:
        return self._proposal_is_ai

    def _get_proposal_summary(self) -> str:
        return self._proposal.short_text() if self._proposal else ""

    def _get_proposal_path(self) -> str:
        return self._proposal.path_text() if self._proposal else ""

    def _get_proposal_path_ids(self) -> list:
        return list(self._proposal.path) if self._proposal else []

    def _get_proposal_hops(self) -> int:
        return self._proposal.hop_count if self._proposal else 0

    def _get_proposal_is_multi_hop(self) -> bool:
        return bool(self._proposal and self._proposal.is_multi_hop)

    def _get_ai_status(self) -> str:
        return self._ai_status

    def _get_ai_agent_name(self) -> str:
        agent = self._agents.get(self._ai_player_id() or -1)
        return agent.name if agent else "—"

    def _get_ai_details(self) -> list:
        """Only metadata that actually exists — never fabricated values."""
        if not (self._proposal_is_ai and self._proposal_metadata):
            return []
        labels = {
            "agent": "Agent",
            "seed": "Seed",
            "legal_move_count": "Legal moves",
            "search_time_ms": "Search time (ms)",
            "simulation_count": "Simulations",
            "value": "Value",
            "visit_count": "Visits",
            "policy_probability": "Policy",
        }
        return [
            {"label": labels.get(key, key), "value": str(value)}
            for key, value in self._proposal_metadata.items()
            if value is not None
        ]

    def _get_last_move_text(self) -> str:
        if self._last_move is None:
            return "—"
        spec = player_by_id(self._players, self._last_move.player_id)
        tag = "AI" if spec.kind is PlayerKind.AI else f"P{spec.id}"
        return f"{tag}  {self._last_move.short_text()}"

    def _get_selected_position(self) -> int:
        return -1 if self._selected is None else self._selected

    def _get_default_save_dir(self) -> str:
        DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        return DEFAULT_SAVE_DIR.as_uri()

    phase = Property(str, _get_phase, notify=changed)
    gameLabel = Property(str, _get_game_label, notify=changed)
    turnNumber = Property(int, _get_turn_number, notify=changed)
    currentPlayerId = Property(int, _get_current_player_id, notify=changed)
    currentPlayerName = Property(str, _get_current_player_name, notify=changed)
    currentPlayerColor = Property(str, _get_current_player_color, notify=changed)
    isCurrentPlayerAi = Property(bool, _get_is_current_ai, notify=changed)
    statusMessage = Property(str, _get_status_message, notify=changed)
    errorMessage = Property(str, _get_error_message, notify=changed)
    isGameOver = Property(bool, _get_is_game_over, notify=changed)
    winnerId = Property(int, _get_winner_id, notify=changed)
    winnerName = Property(str, _get_winner_name, notify=changed)
    canUndo = Property(bool, _get_can_undo, notify=changed)
    hasProposal = Property(bool, _get_has_proposal, notify=changed)
    canConfirm = Property(bool, _get_can_confirm, notify=changed)
    canCancel = Property(bool, _get_can_cancel, notify=changed)
    canSelect = Property(bool, _get_can_select, notify=changed)
    proposalIsAi = Property(bool, _get_proposal_is_ai, notify=changed)
    proposalSummary = Property(str, _get_proposal_summary, notify=changed)
    proposalPath = Property(str, _get_proposal_path, notify=changed)
    proposalPathIds = Property("QVariantList", _get_proposal_path_ids, notify=changed)
    proposalHopCount = Property(int, _get_proposal_hops, notify=changed)
    proposalIsMultiHop = Property(bool, _get_proposal_is_multi_hop, notify=changed)
    aiStatus = Property(str, _get_ai_status, notify=changed)
    aiAgentName = Property(str, _get_ai_agent_name, notify=changed)
    aiDetails = Property("QVariantList", _get_ai_details, notify=changed)
    lastMoveText = Property(str, _get_last_move_text, notify=changed)
    selectedPosition = Property(int, _get_selected_position, notify=changed)
    defaultSaveDir = Property(str, _get_default_save_dir, constant=True)

    # ------------------------------------------------------------------
    # slots called from QML
    # ------------------------------------------------------------------
    @Slot(int)
    def selectPosition(self, position_id: int) -> None:
        """Handle a click on a hole.  All legality comes from the engine."""
        if self._phase is Phase.HUMAN_MOVE_PROPOSED:
            self._fail("Confirm or cancel the proposed move first.")
            return
        if self._phase is not Phase.WAITING_FOR_HUMAN:
            self._fail("The board is locked right now.")
            return
        if not 0 <= position_id < len(self._session.board):
            return

        state = self._session.state
        occupant = state.occupant(position_id)
        current = state.current_player_id

        if occupant == current:
            self._select(position_id)
            return
        if self._selected is not None and position_id in self._legal:
            self._propose(self._legal[position_id], is_ai=False, metadata={})
            return
        if occupant != EMPTY:
            spec = player_by_id(self._players, occupant)
            self._fail(f"{spec.name}'s piece — it is {self._current_spec().name}'s turn.")
            return
        if self._selected is not None:
            # keep the selection so a misclick does not force a re-select
            self._fail("Not a legal destination for the selected piece.")
            return
        self._clear_selection()
        self._emit()

    @Slot()
    def cancelProposal(self) -> None:
        if self._phase is Phase.HUMAN_MOVE_PROPOSED:
            self._proposal = None
            self._proposal_metadata = {}
            self._proposal_is_ai = False
            self._board_model.set_proposal(None)
            self._phase = Phase.WAITING_FOR_HUMAN
            self._selected = None
            self._legal = {}
            self._board_model.set_selection(None, set(), set())
            self._status_message = "Proposal cancelled."
            self._emit()
        elif self._selected is not None:
            self._clear_selection()
            self._status_message = "Selection cleared."
            self._emit()

    @Slot()
    def confirmProposal(self) -> None:
        if not self._get_can_confirm() or self._proposal is None:
            return
        self._commit(self._proposal, self._proposal_metadata)

    @Slot()
    def thinkAgain(self) -> None:
        if self._phase is not Phase.AI_MOVE_PROPOSED or self._proposal is None:
            return
        self._ai_rejected.append(self._proposal)
        self._proposal = None
        self._proposal_metadata = {}
        self._board_model.set_proposal(None)
        self._start_ai_turn()

    @Slot()
    def undoLastMove(self) -> None:
        if not self._session.can_undo():
            self._fail("Nothing to undo.")
            return
        self._stop_animation()
        self._generation += 1  # invalidates any AI result still in flight
        self._session.undo()
        self._reset_transient()
        self._sync_models()
        self._last_move = self._session.history[-1].move if self._session.history else None
        self._board_model.set_last_move(self._last_move)
        self._status_message = "Last move undone."
        self._enter_turn()

    @Slot()
    def newGame(self) -> None:
        self._stop_animation()
        self._generation += 1
        self._session.reset()
        self._game_number += 1
        for agent in self._agents.values():
            if isinstance(agent, RandomAgent):
                agent.reset()
        self._reset_transient()
        self._last_move = None
        self._board_model.set_last_move(None)
        self._sync_models()
        self._status_message = "New game started."
        self._enter_turn()

    @Slot(str, result=bool)
    def saveGame(self, path_or_url: str) -> bool:
        try:
            target = _path_to_local(path_or_url)
            if target.suffix == "":
                target = target.with_suffix(".json")
            self._session.save(target)
        except Exception as exc:
            self._fail(f"Save failed: {exc}")
            return False
        self._status_message = f"Saved to {target}"
        self._emit()
        return True

    @Slot(str, result=bool)
    def loadGame(self, path_or_url: str) -> bool:
        self._stop_animation()
        self._generation += 1
        try:
            self._session.load(_path_to_local(path_or_url))
        except Exception as exc:
            self._fail(f"Load failed: {exc}")
            return False
        self._reset_transient()
        self._last_move = self._session.history[-1].move if self._session.history else None
        self._sync_models()
        self._board_model.set_last_move(self._last_move)
        self._status_message = "Game loaded."
        self._enter_turn()
        return True

    @Slot()
    def requestAiMove(self) -> None:
        if self._phase is Phase.WAITING_FOR_HUMAN and self._get_is_current_ai():
            self._start_ai_turn()

    @Slot()
    def shutdown(self) -> None:
        self._stop_animation()
        self._generation += 1
        self._worker.shutdown()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _emit(self) -> None:
        self.changed.emit()

    def _fail(self, message: str) -> None:
        self._error_message = message
        self.errorRaised.emit(message)
        self._emit()

    def _reset_transient(self) -> None:
        self._selected = None
        self._legal = {}
        self._proposal = None
        self._proposal_metadata = {}
        self._proposal_is_ai = False
        self._ai_rejected.clear()
        self._error_message = ""
        self._board_model.clear_interaction()

    def _sync_models(self) -> None:
        state = self._session.state
        self._board_model.set_occupancy(state.occupancy)
        self._piece_model.rebuild(state.occupancy)
        self._history_model.set_records(self._session.history)
        self._player_model.update(state.occupancy, state.current_player_id)

    def _ai_player_id(self) -> int | None:
        for spec in self._players:
            if spec.kind is PlayerKind.AI:
                return spec.id
        return None

    def _select(self, position_id: int) -> None:
        self._selected = position_id
        self._legal = self._session.moves_from(position_id)
        steps = {d for d, m in self._legal.items() if m.kind is MoveKind.STEP}
        jumps = {d for d, m in self._legal.items() if m.kind is MoveKind.JUMP}
        self._board_model.set_selection(position_id, steps, jumps)
        self._board_model.set_proposal(None)
        self._status_message = (
            f"{len(self._legal)} legal destination(s)." if self._legal else "No legal move for that piece."
        )
        self._error_message = ""
        self._emit()

    def _clear_selection(self) -> None:
        self._selected = None
        self._legal = {}
        self._board_model.set_selection(None, set(), set())
        self._board_model.set_proposal(None)

    def _propose(self, move: Move, *, is_ai: bool, metadata: dict[str, Any]) -> None:
        self._proposal = move
        self._proposal_metadata = dict(metadata)
        self._proposal_is_ai = is_ai
        self._board_model.set_proposal(move)
        self._phase = Phase.AI_MOVE_PROPOSED if is_ai else Phase.HUMAN_MOVE_PROPOSED
        self._error_message = ""
        self._status_message = (
            "AI move proposed — move the piece on the physical board, then confirm."
            if is_ai
            else "Move proposed — confirm or cancel."
        )
        if is_ai:
            self._ai_status = "Move proposed"
        self._emit()

    def _commit(self, move: Move, metadata: dict[str, Any]) -> None:
        row = self._piece_model.row_at(move.source)
        try:
            self._session.commit(move, metadata)
        except (IllegalMoveError, RuntimeError) as exc:
            self._fail(str(exc))
            self._reset_transient()
            self._enter_turn()
            return

        self._generation += 1
        self._last_move = move
        self._reset_transient()
        self._board_model.set_occupancy(self._session.state.occupancy)
        self._board_model.set_last_move(move)
        self._history_model.set_records(self._session.history)
        self._player_model.update(
            self._session.state.occupancy, self._session.state.current_player_id
        )
        self._ai_status = "Ready"
        self._status_message = ""
        self._start_animation(row, move)

    # -- animation -------------------------------------------------------
    def _start_animation(self, row: int, move: Move) -> None:
        if not self._animate or row < 0:
            self._piece_model.rebuild(self._session.state.occupancy)
            self._finish_move()
            return
        self._anim_row = row
        self._anim_path = move.path
        self._anim_index = 0
        self._phase = Phase.ANIMATING
        self._emit()
        self._anim_timer.start()

    def _on_animation_tick(self) -> None:
        self._anim_index += 1
        if self._anim_index >= len(self._anim_path):
            self._stop_animation()
            self._piece_model.rebuild(self._session.state.occupancy)
            self._finish_move()
            return
        last_hop = self._anim_index == len(self._anim_path) - 1
        self._piece_model.move_piece(
            self._anim_row, self._anim_path[self._anim_index], moving=not last_hop
        )

    def _stop_animation(self) -> None:
        self._anim_timer.stop()
        self._anim_row = -1
        self._anim_path = ()
        self._anim_index = 0

    def _finish_move(self) -> None:
        self._enter_turn()

    # -- turn machine ----------------------------------------------------
    def _enter_turn(self) -> None:
        if self._session.is_over:
            self._phase = Phase.GAME_OVER
            self._ai_status = "Idle"
            winner = self._session.state.winner_id or 0
            self._status_message = f"Game over — {self._get_winner_name()} wins."
            self._emit()
            self.gameFinished.emit(winner)
            return

        self._phase = Phase.WAITING_FOR_HUMAN
        if self._get_is_current_ai():
            self._ai_rejected.clear()
            self._start_ai_turn()
        else:
            self._ai_status = "Ready"
            self._emit()

    def _start_ai_turn(self) -> None:
        player_id = self._session.state.current_player_id
        agent = self._agents.get(player_id)
        if agent is None:
            self._fail(f"No agent registered for player {player_id}.")
            return
        moves = self._session.legal_moves(player_id)
        if not moves:
            self._ai_status = "No legal move"
            self._fail(f"{self._current_spec().name} has no legal move.")
            return

        self._phase = Phase.AI_THINKING
        self._ai_status = "Thinking…"
        self._status_message = f"{self._current_spec().name} is thinking…"
        self._emit()
        request = MoveRequest(
            board=self._session.board,
            state=self._session.state,
            legal_moves=moves,
            avoid=tuple(self._ai_rejected),
        )
        self._worker.submit(agent, request, self._generation)

    def _on_proposal_ready(self, generation: int, proposal: MoveProposal | None) -> None:
        if generation != self._generation:
            return  # stale: computed against a board that no longer exists
        if self._phase is not Phase.AI_THINKING:
            return
        if proposal is None:
            self._ai_status = "No legal move"
            self._fail("Agent returned no move.")
            return
        move = proposal.to_move()
        try:
            validate_move(self._session.board, self._session.state, move)
        except IllegalMoveError as exc:
            self._ai_status = "Invalid proposal"
            self._fail(f"Agent proposed an illegal move: {exc}")
            self._phase = Phase.WAITING_FOR_HUMAN
            self._emit()
            return
        self._propose(move, is_ai=True, metadata=dict(proposal.metadata))

    def _on_proposal_failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self._ai_status = "Error"
        self._phase = Phase.WAITING_FOR_HUMAN
        self._fail(f"Agent failed: {message}")
