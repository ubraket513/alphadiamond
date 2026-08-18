"""Thin AlphaZero adapter over the authoritative Diamond rules/session."""

from __future__ import annotations

from .action_codec import ActionCodec, ActionSpaceSpec
from ..game.board import Board, standard_board
from ..game.move import Move
from ..game.rules import IllegalMoveError, find_legal_move, legal_moves
from ..game.session import GameSession
from ..game.state import GameState, GameStatus, PlayerSpec, initial_state


class AlphaZeroGameAdapter:
    """Expose search-friendly operations without duplicating game rules."""

    def __init__(self, players: tuple[PlayerSpec, ...], board: Board | None = None) -> None:
        if len(players) not in (2, 3):
            raise ValueError("AlphaZero supports exactly 2 or 3 players")
        self.players = tuple(players)
        self.board = board or standard_board()
        self.codec = ActionCodec(
            ActionSpaceSpec(
                board_size=len(self.board),
                version=f"diamond{len(self.board)}-srcdst-v1",
            )
        )

    def initial_state(self) -> GameState:
        return initial_state(self.players, self.board)

    def legal_moves(self, state: GameState) -> tuple[Move, ...]:
        return legal_moves(self.board, state)

    def legal_action_ids(self, state: GameState) -> tuple[int, ...]:
        return tuple(
            self.codec.encode(move.source, move.destination)
            for move in self.legal_moves(state)
        )

    def resolve_action(self, state: GameState, action_id: int) -> Move:
        source, destination = self.codec.decode(action_id)
        move = find_legal_move(
            self.board,
            state,
            source,
            destination,
            player_id=state.current_player_id,
        )
        if move is None:
            raise IllegalMoveError(
                f"action {action_id} ({source} → {destination}) is not legal"
            )
        return move

    def apply_action(self, state: GameState, action_id: int) -> GameState:
        """Apply through ``GameSession.commit`` so ranking/turn rules stay authoritative."""
        session = GameSession(self.players, board=self.board, initial=state)
        session.commit(self.resolve_action(state, action_id))
        return session.state

    def is_terminal(self, state: GameState) -> bool:
        return state.status is GameStatus.FINISHED

    def final_order(self, state: GameState) -> tuple[int, ...]:
        if not self.is_terminal(state) or len(state.finish_order) != len(self.players):
            raise ValueError("final order is only available for a completed match")
        return state.finish_order


__all__ = ["AlphaZeroGameAdapter"]
