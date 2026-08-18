"""Vector-valued three-player PUCT without scalar negation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol

from .puct import add_dirichlet_noise, exploration_bonus, select_from_visits
from .tree import VectorEdge, VectorNode
from ..config import MCTSConfig
from ..evaluator.base import EvalRequest, Evaluator


class ThreePlayerSearchGame(Protocol):
    def current_player_id(self, state: Any) -> int: ...
    def legal_action_ids(self, state: Any) -> tuple[int, ...]: ...
    def apply_action(self, state: Any, action_id: int) -> Any: ...
    def is_terminal(self, state: Any) -> bool: ...
    def terminal_vector_value(self, state: Any) -> dict[int, float]: ...
    def evaluation_request(self, state: Any) -> EvalRequest: ...


@dataclass(frozen=True, slots=True)
class SearchResult3P:
    selected_action: int
    visit_counts: dict[int, int]
    policy: dict[int, float]
    q_values: dict[int, dict[int, float]]


class MCTS3P:
    def __init__(self, game: ThreePlayerSearchGame, evaluator: Evaluator, config: MCTSConfig) -> None:
        if config.simulations <= 0:
            raise ValueError("simulations must be positive")
        self.game = game
        self.evaluator = evaluator
        self.config = config
        self.rng = random.Random(config.seed)

    def run(self, state: Any, *, temperature: float = 0.0) -> SearchResult3P:
        if self.game.is_terminal(state):
            raise ValueError("cannot search a terminal state")
        root_request = self.game.evaluation_request(state)
        player_ids = root_request.canonical_player_ids
        if len(player_ids) != 3 or len(set(player_ids)) != 3:
            raise ValueError("3P search requires three stable global player ids")
        root = VectorNode(
            state=state,
            player_id=self.game.current_player_id(state),
            player_ids=player_ids,
        )
        self._expand(root, root_request, root_noise=True)

        for _ in range(self.config.simulations):
            node = root
            path: list[VectorEdge] = []
            while node.expanded and not self.game.is_terminal(node.state):
                action, edge = self._select(node)
                path.append(edge)
                child = node.children.get(action)
                if child is None:
                    child_state = self.game.apply_action(node.state, action)
                    child = VectorNode(
                        state=child_state,
                        player_id=self.game.current_player_id(child_state),
                        player_ids=root.player_ids,
                    )
                    node.children[action] = child
                node = child
                if not node.expanded:
                    break

            if self.game.is_terminal(node.state):
                value = self.game.terminal_vector_value(node.state)
            else:
                value = self._expand(node, self.game.evaluation_request(node.state))
            if set(value) != set(root.player_ids):
                raise ValueError("3P value vector does not match stable player identity")

            # The same global-player utility vector is backed through every
            # ancestor. It is never negated or rotated.
            for edge in reversed(path):
                edge.visit_count += 1
                for player_id in root.player_ids:
                    edge.value_sum[player_id] += value[player_id]

        visits = {action: edge.visit_count for action, edge in root.edges.items()}
        total = sum(visits.values())
        selected = select_from_visits(visits, temperature=temperature, rng=self.rng)
        return SearchResult3P(
            selected_action=selected,
            visit_counts=visits,
            policy={action: count / total for action, count in visits.items()},
            q_values={action: edge.q_vector() for action, edge in root.edges.items()},
        )

    def _expand(
        self,
        node: VectorNode,
        request: EvalRequest,
        *,
        root_noise: bool = False,
    ) -> dict[int, float]:
        result = self.evaluator.evaluate((request,))[0]
        if not isinstance(result.value, tuple) or len(result.value) != 3:
            raise ValueError("3P evaluator must return a three-component value tuple")
        if len(request.canonical_player_ids) != 3:
            raise ValueError("3P evaluator request must identify three canonical players")
        value = dict(zip(request.canonical_player_ids, result.value))
        if set(value) != set(node.player_ids):
            raise ValueError("canonical evaluator value cannot map to global players")
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
        node.edges = {
            action: VectorEdge(prior, node.player_ids)
            for action, prior in priors.items()
        }
        node.expanded = True
        return value

    def _select(self, node: VectorNode) -> tuple[int, VectorEdge]:
        parent_visits = sum(edge.visit_count for edge in node.edges.values())
        action = min(
            node.edges,
            key=lambda candidate: (
                -(
                    node.edges[candidate].q(node.player_id)
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


__all__ = ["MCTS3P", "SearchResult3P", "ThreePlayerSearchGame"]
