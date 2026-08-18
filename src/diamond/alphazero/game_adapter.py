"""Thin AlphaZero adapter over the authoritative Diamond rules/session."""

from __future__ import annotations

from .action_codec import ActionCodec, ActionSpaceSpec
from .encoder import CanonicalEncoder
from .evaluator.base import EvalRequest
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
        self.encoder = CanonicalEncoder(self.board, self.codec)

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


class DiamondSearchAdapter:
    """Canonical-action view of :class:`AlphaZeroGameAdapter` for MCTS."""

    def __init__(self, game: AlphaZeroGameAdapter) -> None:
        self.game = game

    @property
    def players(self) -> tuple[PlayerSpec, ...]:
        return self.game.players

    def current_player_id(self, state: GameState) -> int:
        return state.current_player_id

    def legal_action_ids(self, state: GameState) -> tuple[int, ...]:
        return tuple(
            self.game.encoder.to_canonical_action(
                physical,
                self.players,
                state.current_player_id,
            )
            for physical in self.game.legal_action_ids(state)
        )

    def apply_action(self, state: GameState, canonical_action_id: int) -> GameState:
        physical = self.game.encoder.to_physical_action(
            canonical_action_id,
            self.players,
            state.current_player_id,
        )
        return self.game.apply_action(state, physical)

    def is_terminal(self, state: GameState) -> bool:
        return self.game.is_terminal(state)

    def evaluation_request(self, state: GameState) -> EvalRequest:
        encoded = self.game.encoder.encode(state, self.players)
        return EvalRequest(
            node_features=encoded.node_features,
            legal_action_ids=self.legal_action_ids(state),
            canonical_player_ids=encoded.canonical_player_ids,
        )

    def terminal_scalar_value(self, state: GameState, player_id: int) -> float:
        if len(self.players) != 2:
            raise ValueError("scalar terminal values are only defined for 2P")
        order = self.game.final_order(state)
        return 1.0 if player_id == order[0] else -1.0

    def terminal_vector_value(self, state: GameState) -> dict[int, float]:
        if len(self.players) != 3:
            raise ValueError("placement utility vectors are only defined for 3P")
        order = self.game.final_order(state)
        return dict(zip(order, (1.0, 0.0, -1.0)))


__all__ = ["AlphaZeroGameAdapter", "DiamondSearchAdapter"]
