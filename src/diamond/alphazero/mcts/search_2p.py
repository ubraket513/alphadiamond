"""Scalar two-player PUCT with explicit perspective alternation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol

from .puct import add_dirichlet_noise, exploration_bonus, select_from_visits
from .tree import ScalarEdge, ScalarNode
from ..config import MCTSConfig
from ..evaluator.base import EvalRequest, Evaluator


class TwoPlayerSearchGame(Protocol):
    def current_player_id(self, state: Any) -> int: ...
    def legal_action_ids(self, state: Any) -> tuple[int, ...]: ...
    def apply_action(self, state: Any, action_id: int) -> Any: ...
    def is_terminal(self, state: Any) -> bool: ...
    def terminal_scalar_value(self, state: Any, player_id: int) -> float: ...
    def evaluation_request(self, state: Any) -> EvalRequest: ...


@dataclass(frozen=True, slots=True)
class SearchResult2P:
    selected_action: int
    visit_counts: dict[int, int]
    policy: dict[int, float]
    q_values: dict[int, float]


class MCTS2P:
    def __init__(self, game: TwoPlayerSearchGame, evaluator: Evaluator, config: MCTSConfig) -> None:
        if config.simulations <= 0:
            raise ValueError("simulations must be positive")
        self.game = game
        self.evaluator = evaluator
        self.config = config
        self.rng = random.Random(config.seed)

    def run(self, state: Any, *, temperature: float = 0.0) -> SearchResult2P:
        if self.game.is_terminal(state):
            raise ValueError("cannot search a terminal state")
        root = ScalarNode(state=state, player_id=self.game.current_player_id(state))
        self._expand(root, root_noise=True)

        for _ in range(self.config.simulations):
            node = root
            path: list[ScalarEdge] = []
            while node.expanded and not self.game.is_terminal(node.state):
                action, edge = self._select(node)
                path.append(edge)
                child = node.children.get(action)
                if child is None:
                    child_state = self.game.apply_action(node.state, action)
                    child = ScalarNode(
                        state=child_state,
                        player_id=self.game.current_player_id(child_state),
                    )
                    node.children[action] = child
                node = child
                if not node.expanded:
                    break

            if self.game.is_terminal(node.state):
                value = self.game.terminal_scalar_value(node.state, node.player_id)
            else:
                value = self._expand(node)

            # Evaluator/terminal value is from the leaf player-to-act's
            # perspective. Each traversed 2P edge crosses exactly one turn, so
            # child -> parent perspective changes sign once per edge.
            for edge in reversed(path):
                value = -value
                edge.visit_count += 1
                edge.value_sum += value

        visits = {action: edge.visit_count for action, edge in root.edges.items()}
        total = sum(visits.values())
        policy = {action: count / total for action, count in visits.items()}
        selected = select_from_visits(visits, temperature=temperature, rng=self.rng)
        return SearchResult2P(
            selected_action=selected,
            visit_counts=visits,
            policy=policy,
            q_values={action: edge.q for action, edge in root.edges.items()},
        )

    def _expand(self, node: ScalarNode, *, root_noise: bool = False) -> float:
        request = self.game.evaluation_request(node.state)
        result = self.evaluator.evaluate((request,))[0]
        if not isinstance(result.value, float):
            raise ValueError("2P evaluator must return a scalar float value")
        legal = set(self.game.legal_action_ids(node.state))
        if set(result.priors) != legal:
            raise ValueError("evaluator priors must match authoritative legal actions")
        priors = result.priors
        if root_noise:
            priors = add_dirichlet_noise(
                priors,
                alpha=self.config.dirichlet_alpha,
                epsilon=self.config.dirichlet_epsilon,
                rng=self.rng,
            )
        node.edges = {action: ScalarEdge(prior) for action, prior in priors.items()}
        node.expanded = True
        return result.value

    def _select(self, node: ScalarNode) -> tuple[int, ScalarEdge]:
        parent_visits = sum(edge.visit_count for edge in node.edges.values())
        action = min(
            node.edges,
            key=lambda candidate: (
                -(
                    node.edges[candidate].q
                    + exploration_bonus(
                        node.edges[candidate].prior,
                        parent_visits,
                        node.edges[candidate].visit_count,
                        self.config.c_puct,
                    )
                ),
                candidate,
            ),
        )
        return action, node.edges[action]


__all__ = ["MCTS2P", "SearchResult2P", "TwoPlayerSearchGame"]
