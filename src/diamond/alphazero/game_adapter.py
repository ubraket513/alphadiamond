"""AlphaZero adapter over the authoritative Diamond rules.

The rules are the C++ core's. This holds the Python-side shape of a position --
``GameState``, ``PlayerSpec``, the action codec and the canonical encoder -- and
asks the native ``Game`` for everything that *applies* rules: which actions are
legal, and what a position becomes when one is played.

**The state type stays Python on purpose.** A ``GameState`` crosses process
boundaries in ``SelfPlayJob``, is stored in run state and is what the Qt agent
and the arena already speak. Replacing it with the native ``State`` is Phase B
of the migration (see docs/architecture/decisions.md); replacing the *rules*
underneath it is Phase A, and is what this file does. The conversion either way
is a field copy.

Parity is not assumed: ``tests/native/test_game_adapter_parity.py`` runs the
whole fixture corpus -- 1,327 positions, every legal action of each -- through
both this adapter and the Python engine and requires identical successors.
"""

from __future__ import annotations

from ..contract.board import Board, standard_board
from ..contract.move import IllegalMoveError
from ..contract.state import GameState, GameStatus, PlayerSpec, initial_state
from .action_codec import ActionCodec, ActionSpaceSpec
from .encoder import CanonicalEncoder
from .evaluator.base import EvalRequest

_NATIVE_FINISHED = 1


class AlphaZeroGameAdapter:
    """Expose search-friendly operations without duplicating game rules."""

    def __init__(
        self,
        players: tuple[PlayerSpec, ...],
        board: Board | None = None,
        initial: GameState | None = None,
    ) -> None:
        if len(players) not in (2, 3):
            raise ValueError("AlphaZero supports exactly 2 or 3 players")
        self.players = tuple(players)
        self.board = board or standard_board()
        self._initial = initial if initial is not None else initial_state(self.players, self.board)
        if len(self._initial.occupancy) != len(self.board):
            raise ValueError("initial state does not match board topology")
        # Imported here rather than at module scope: ``native`` reaches the
        # bootstrap heuristic for its topology tables, and that package imports
        # this one back.
        from .native import native_game, require_native

        self._module = require_native()
        # No board-size check: ``Board()`` takes no arguments and there is one
        # board. If a reduced board ever exists, the core will reject it and
        # that rejection is the right answer, not a fallback.
        self._native = native_game(self.players)
        self.codec = ActionCodec(
            ActionSpaceSpec(
                board_size=len(self.board),
                version=f"diamond{len(self.board)}-srcdst-v1",
            )
        )
        self.encoder = CanonicalEncoder(self.board, self.codec)

    def initial_state(self) -> GameState:
        return self._initial

    def _to_native(self, state: GameState):
        return self._module.State(
            occupancy=list(state.occupancy),
            current_player=state.current_player_id,
            turn_number=state.turn_number,
            status=_NATIVE_FINISHED if state.status is GameStatus.FINISHED else 0,
            finish_order=list(state.finish_order),
        )

    def _from_native(self, state) -> GameState:
        return GameState(
            occupancy=tuple(state.occupancy),
            current_player_id=state.current_player_id,
            turn_number=state.turn_number,
            status=(
                GameStatus.FINISHED
                if int(state.status) == _NATIVE_FINISHED
                else GameStatus.IN_PROGRESS
            ),
            finish_order=tuple(state.finish_order),
        )

    def legal_action_ids(self, state: GameState) -> tuple[int, ...]:
        return tuple(self._native.legal_action_ids(self._to_native(state)))

    def apply_action(self, state: GameState, action_id: int) -> GameState:
        """Play ``action_id``; the core settles turn order and the podium."""
        try:
            successor = self._native.apply_action(self._to_native(state), action_id)
        except (ValueError, IndexError) as error:
            # The adapter's contract is one exception for "you cannot play
            # that", whatever the core called it.
            raise IllegalMoveError(f"action {action_id} is not legal: {error}") from error
        return self._from_native(successor)

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

    def initial_state(self) -> GameState:
        return self.game.initial_state()

    def current_player_id(self, state: GameState) -> int:
        if (
            len(self.players) == 2
            and self.game.is_terminal(state)
            and len(state.finish_order) == 2
        ):
            # The authoritative session keeps the last mover in
            # ``current_player_id`` when everybody is skipped at game end.
            # Soo's terminal leaf still needs the would-be opponent
            # perspective so scalar backup flips exactly once across the edge.
            return state.finish_order[1]
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

    def final_order(self, state: GameState) -> tuple[int, ...]:
        return self.game.final_order(state)


__all__ = ["AlphaZeroGameAdapter", "DiamondSearchAdapter"]
