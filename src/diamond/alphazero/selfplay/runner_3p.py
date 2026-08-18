"""Single-process full-ranking self-play for the Min three-player model."""

from __future__ import annotations

from dataclasses import replace

from .common import PendingSample, SelfPlayEpisode, SelfPlayGame, sparse_policy
from ..config import MCTSConfig, SelfPlayConfig
from ..evaluator.base import Evaluator
from ..identity import CheckpointCompatibilitySpec, MIN_MODEL_NAME
from ..mcts.search_3p import MCTS3P
from ..replay import TrainingSample


class MinSelfPlayRunner:
    def __init__(
        self,
        game: SelfPlayGame,
        evaluator: Evaluator,
        mcts_config: MCTSConfig,
        selfplay_config: SelfPlayConfig,
        compatibility: CheckpointCompatibilitySpec,
    ) -> None:
        if compatibility.identity.model_name != MIN_MODEL_NAME:
            raise ValueError("MinSelfPlayRunner requires Min compatibility metadata")
        self.game = game
        self.evaluator = evaluator
        self.mcts_config = mcts_config
        self.selfplay_config = selfplay_config
        self.compatibility = compatibility

    def run(self) -> SelfPlayEpisode:
        state = self.game.initial_state()
        pending: list[PendingSample] = []
        move_count = 0
        while not self.game.is_terminal(state) and move_count < self.selfplay_config.max_moves:
            temperature = (
                self.selfplay_config.temperature
                if move_count < self.selfplay_config.temperature_moves
                else 0.0
            )
            search = MCTS3P(
                self.game,
                self.evaluator,
                replace(self.mcts_config, seed=self.selfplay_config.seed + move_count),
            )
            result = search.run(state, temperature=temperature)
            pending.append(
                PendingSample(self.game.evaluation_request(state), sparse_policy(result.policy))
            )
            state = self.game.apply_action(state, result.selected_action)
            move_count += 1

        if not self.game.is_terminal(state):
            return SelfPlayEpisode((), None, move_count, False, "max_game_moves_exceeded")

        final_order = self.game.final_order(state)
        global_utility = dict(zip(final_order, (1.0, 0.0, -1.0)))
        samples = tuple(
            TrainingSample(
                compatibility=self.compatibility,
                node_features=row.request.node_features,
                canonical_player_ids=row.request.canonical_player_ids,
                sparse_policy=row.sparse_policy,
                value_target=tuple(
                    global_utility[player_id]
                    for player_id in row.request.canonical_player_ids
                ),
            )
            for row in pending
        )
        return SelfPlayEpisode(samples, final_order, move_count, True)


__all__ = ["MinSelfPlayRunner"]
