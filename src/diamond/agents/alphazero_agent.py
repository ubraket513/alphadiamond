"""An :class:`Agent` that picks moves with the AlphaZero MCTS stack.

This is the GUI-facing seam the agent boundary was designed for: the controller
still knows only ``Agent``/``MoveRequest``/``MoveProposal``, while everything
underneath -- canonical encoding, PUCT search, the evaluator chain -- is the
same code self-play uses.

Torch is deliberately optional.  With no checkpoint the agent runs on the
deterministic dummy evaluator wrapped in a bootstrap prior, which is exactly the
configuration the bootstrap probe measured: strong enough to walk pieces home
and finish games, and available on a machine with no trained network at all.
"""

from __future__ import annotations

from ..alphazero.bootstrap.evaluator import bootstrap_evaluator
from ..alphazero.config import (
    CANONICAL_TARGET_VACANCY_DISTANCE_V2,
    MCTSConfig,
)
from ..alphazero.evaluator.base import Evaluator
from ..alphazero.evaluator.dummy import DummyEvaluator
from ..alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from ..alphazero.mcts.search_2p import MCTS2P
from ..alphazero.mcts.search_3p import MCTS3P
from ..game.state import PlayerSpec
from .base import Agent, MoveProposal, MoveRequest

DEFAULT_SIMULATIONS = 128
"""Measured floor for finishing games, not a comfort setting.

At 32 simulations with ``temperature=0`` the bootstrap prior packs eight pieces
into the target camp and then shuffles forever -- the same greedy plateau the v1
probe documented.  128 walks a two-player game home in ~84 moves at roughly
0.16s per move on CPU, which is still well inside the GUI thinking delay.
"""


class AlphaZeroAgent(Agent):
    """Runs one MCTS search per requested move.

    The search tree is rebuilt each turn.  That is wasteful compared with
    reusing the subtree, but the GUI hands us an authoritative state that may
    have been undone or loaded from disk, so a fresh root is the only one
    guaranteed to match the board on screen.
    """

    def __init__(
        self,
        players: tuple[PlayerSpec, ...],
        *,
        evaluator: Evaluator | None = None,
        simulations: int = DEFAULT_SIMULATIONS,
        bootstrap_prior: str = CANONICAL_TARGET_VACANCY_DISTANCE_V2,
        seed: int = 0,
        temperature: float = 0.0,
    ) -> None:
        if len(players) not in (2, 3):
            raise ValueError("AlphaZero supports exactly 2 or 3 players")
        if simulations <= 0:
            raise ValueError("simulations must be positive")
        self._players = tuple(players)
        self._bootstrap_prior = bootstrap_prior
        self._simulations = int(simulations)
        self._seed = int(seed)
        self._temperature = float(temperature)
        base = DummyEvaluator(self._neutral_value()) if evaluator is None else evaluator
        self._evaluator = bootstrap_evaluator(base, bootstrap_prior)
        self._untrained = evaluator is None

    def _neutral_value(self):
        return 0.0 if len(self._players) == 2 else (0.0, 0.0, 0.0)

    @property
    def name(self) -> str:
        return "AlphaZero" if not self._untrained else "AlphaZero (bootstrap)"

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def bootstrap_prior(self) -> str:
        return self._bootstrap_prior

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = int(seed)

    def choose_move(self, request: MoveRequest) -> MoveProposal | None:
        if not request.legal_moves:
            return None

        state = request.state
        game = DiamondSearchAdapter(
            AlphaZeroGameAdapter(self._players, board=request.board, initial=state)
        )
        if game.is_terminal(state):
            return None

        seed = self._seed if request.seed is None else int(request.seed)
        config = MCTSConfig(simulations=self._simulations, seed=seed)
        search_cls = MCTS2P if len(self._players) == 2 else MCTS3P
        result = search_cls(game, self._evaluator, config).run(
            state, temperature=self._temperature
        )

        action = self._respecting_avoid(result, game, state, request)
        move = game.game.resolve_action(
            state,
            game.game.encoder.to_physical_action(
                action, self._players, state.current_player_id
            ),
        )
        return MoveProposal.from_move(
            move,
            metadata={
                "agent": self.name,
                "seed": seed,
                "simulations": self._simulations,
                "bootstrap_prior": self._bootstrap_prior,
                "visit_count": result.visit_counts.get(action, 0),
                "policy_probability": round(result.policy.get(action, 0.0), 4),
                "legal_move_count": len(request.legal_moves),
            },
        )

    def _respecting_avoid(self, result, game, state, request: MoveRequest) -> int:
        """Best action that is not in ``avoid``, falling back to the best overall.

        "Think Again" asks for a different suggestion; when every candidate is
        avoided we still owe the caller a legal move.
        """
        if not request.avoid:
            return result.selected_action
        avoided = {(m.source, m.destination) for m in request.avoid}

        def physical(action: int) -> tuple[int, int]:
            return game.game.codec.decode(
                game.game.encoder.to_physical_action(
                    action, self._players, state.current_player_id
                )
            )

        ranked = sorted(
            result.visit_counts.items(), key=lambda item: (-item[1], item[0])
        )
        for action, _ in ranked:
            if physical(action) not in avoided:
                return action
        return result.selected_action


__all__ = ["DEFAULT_SIMULATIONS", "AlphaZeroAgent"]
