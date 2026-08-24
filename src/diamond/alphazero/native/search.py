"""Native two-player search, answered by a Python evaluator.

The C++ core owns the tree: selection, expansion, backup, the Dirichlet mixing
and the temperature draw all happen there, and Python only answers the
positions the search asks about. That is the same division the self-play pool
uses; this module is the single-search shape the arena needs, where two
different networks alternate moves within one game.

The interface deliberately mirrors :class:`diamond.alphazero.mcts.MCTS2P` --
same constructor arguments, same ``run`` signature, same result fields -- so a
caller can switch engines without knowing which one it got.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..evaluator.base import EvalRequest, Evaluator
from ..search_result import SearchResult2P, SearchResult3P
from . import native_game, require_native

VALUE_ONLY = "value_only"
POLICY_VALUE = "policy_value"


@dataclass(frozen=True, slots=True)
class _Config:
    simulations: int
    c_puct: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
    seed: int


def _native_state(module: Any, state: Any) -> Any:
    return module.State(
        occupancy=list(state.occupancy),
        current_player=state.current_player_id,
        turn_number=state.turn_number,
        status=1 if getattr(state.status, "value", state.status) == "finished" else 0,
        finish_order=list(state.finish_order),
    )


def _budget_seconds(deadline: Any) -> float:
    """Seconds left on a game's wall-clock budget, or 0 for unlimited.

    The native side measures with steady_clock rather than the Deadline's
    injectable one, so what crosses the boundary is what is *left* at the moment
    the search starts. An already-spent budget is passed as a tiny positive
    number, not as zero: zero means unlimited, and the two must not collide.
    """
    if deadline is None:
        return 0.0
    remaining = getattr(deadline, "remaining_s", None)
    if remaining is None:
        return 0.0
    return max(float(remaining), 1e-9)


class NativeSearch2P:
    """A drop-in for :class:`MCTS2P` whose tree lives in C++.

    ``game`` is the Python search adapter, used only to convert states and to
    read the seat list; no rule, encoding or selection decision is taken on the
    Python side.
    """

    @staticmethod
    def can_drive(game: Any) -> bool:
        """Whether this game is one the native core can play.

        The C++ core is compiled around the 73-hole board and a two-seat match.
        Reduced boards and the arena's test doubles are neither, and they must
        keep working on the Python search rather than crashing here.
        """
        players = getattr(game, "players", None)
        if players is None or len(players) != 2:
            return False
        # The search adapter wraps the game adapter, which knows the board it
        # plays. A test double knows nothing about a board, which is the point:
        # it is not a game this can play.
        size = getattr(game, "board_size", None) or getattr(
            getattr(game, "game", None), "board_size", None
        )
        return size == 73

    def __init__(
        self, game: Any, evaluator: Evaluator, config: Any, *, deadline: Any = None
    ) -> None:
        if not self.can_drive(game):
            raise ValueError("the native search needs a two-seat game on the 73-hole board")
        self.deadline = deadline
        self.module = require_native()
        self.game = game
        self.evaluator = evaluator
        self.native = native_game(tuple(game.players))
        self.config = _Config(
            simulations=int(config.simulations),
            c_puct=float(config.c_puct),
            dirichlet_alpha=float(getattr(config, "dirichlet_alpha", 0.3)),
            dirichlet_epsilon=float(getattr(config, "dirichlet_epsilon", 0.0)),
            seed=int(getattr(config, "seed", 0)),
        )

    def _callback(self, features: Any, actions: Any, offsets: Any) -> tuple[Any, Any]:
        """Answer one batch of positions from the Python evaluator.

        The batch is a single position -- an arena move is one forward pass, and
        pretending otherwise would only add a scheduler that has nothing to
        schedule. Kept in the batched shape anyway so the ABI matches the pool's.
        """
        import numpy as np

        rows = features.shape[0]
        requests = []
        for row in range(rows):
            begin = int(offsets[row])
            end = int(offsets[row + 1])
            requests.append(
                EvalRequest(
                    node_features=tuple(tuple(float(v) for v in node) for node in features[row]),
                    legal_action_ids=tuple(int(a) for a in actions[begin:end]),
                    canonical_player_ids=self.canonical_player_ids,
                )
            )
        results = self.evaluator.evaluate(tuple(requests))

        priors = np.empty(int(offsets[rows]), dtype=np.float32)
        values = np.empty(rows, dtype=np.float32)
        for row, (request, result) in enumerate(zip(requests, results)):
            begin = int(offsets[row])
            for index, action in enumerate(request.legal_action_ids):
                priors[begin + index] = result.priors[action]
            value = result.value
            values[row] = float(value[0] if isinstance(value, tuple) else value)
        return priors, values

    def run(self, state: Any, temperature: float = 0.0) -> SearchResult2P:
        module = self.module
        # Canonical ids are a property of the seat list and the player to act,
        # which the search never changes mid-run for the root's perspective;
        # the encoder rotates per node and the native side already did that.
        encoded = self.game.evaluation_request(state)
        self.canonical_player_ids = tuple(encoded.canonical_player_ids)

        config = module.MCTSConfig(
            simulations=self.config.simulations,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
            seed=self.config.seed,
        )
        result = self.native.search_with_callback(
            _native_state(module, state),
            config,
            temperature=float(temperature),
            trace=False,
            callback=self._callback,
            mode=POLICY_VALUE,
            budget_seconds=_budget_seconds(self.deadline),
        )

        actions = [int(action) for action in result["root_actions"]]
        return SearchResult2P(
            selected_action=int(result["selected_action"]),
            visit_counts=dict(zip(actions, (int(v) for v in result["visit_counts"]))),
            policy=dict(zip(actions, (float(v) for v in result["policy"]))),
            q_values=dict(zip(actions, (float(v) for v in result["q_values"]))),
        )


class NativeSearch3P:
    """A drop-in for :class:`MCTS3P` whose tree lives in C++.

    Min's search differs from Soo's in what a value is, not in how the tree is
    walked: the evaluator returns one component per seat, and that vector is
    backed through every ancestor unchanged rather than negated once per edge.
    The C++ side stores it by seat and hands back q vectors in the same order.
    """

    @staticmethod
    def can_drive(game: Any) -> bool:
        players = getattr(game, "players", None)
        if players is None or len(players) != 3:
            return False
        size = getattr(game, "board_size", None) or getattr(
            getattr(game, "game", None), "board_size", None
        )
        return size == 73

    def __init__(
        self, game: Any, evaluator: Evaluator, config: Any, *, deadline: Any = None
    ) -> None:
        if not self.can_drive(game):
            raise ValueError("the native 3P search needs a three-seat game on the 73-hole board")
        self.deadline = deadline
        self.module = require_native()
        self.game = game
        self.evaluator = evaluator
        self.native = native_game(tuple(game.players))
        self.config = _Config(
            simulations=int(config.simulations),
            c_puct=float(config.c_puct),
            dirichlet_alpha=float(getattr(config, "dirichlet_alpha", 0.3)),
            dirichlet_epsilon=float(getattr(config, "dirichlet_epsilon", 0.0)),
            seed=int(getattr(config, "seed", 0)),
        )

    def _callback(self, features: Any, actions: Any, canonical: Any) -> tuple[Any, Any]:
        import numpy as np

        request = EvalRequest(
            node_features=tuple(tuple(float(v) for v in node) for node in features[0]),
            legal_action_ids=tuple(int(a) for a in actions),
            # Per node, not per search: the encoder rotates the seat order to
            # the player to act, and the value components come back in that
            # order. Sending the root's would mislabel every deeper node.
            canonical_player_ids=tuple(int(p) for p in canonical),
        )
        result = self.evaluator.evaluate((request,))[0]
        priors = np.array(
            [result.priors[action] for action in request.legal_action_ids], dtype=np.float32
        )
        values = np.array(result.value, dtype=np.float32)
        return priors, values

    def run(self, state: Any, temperature: float = 0.0) -> SearchResult3P:
        module = self.module
        config = module.MCTSConfig(
            simulations=self.config.simulations,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
            seed=self.config.seed,
        )
        result = self.native.search3p_with_callback(
            _native_state(module, state),
            config,
            temperature=float(temperature),
            callback=self._callback,
            budget_seconds=_budget_seconds(self.deadline),
        )

        actions = [int(action) for action in result["root_actions"]]
        seats = [int(player) for player in result["player_ids"]]
        return SearchResult3P(
            selected_action=int(result["selected_action"]),
            visit_counts=dict(zip(actions, (int(v) for v in result["visit_counts"]))),
            policy=dict(zip(actions, (float(v) for v in result["policy"]))),
            q_values={
                action: {seat: float(vector[index]) for index, seat in enumerate(seats)}
                for action, vector in zip(actions, result["q_vectors"])
            },
        )


__all__ = ["NativeSearch2P", "NativeSearch3P"]
