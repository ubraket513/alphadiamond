"""Single-process full-ranking self-play for the Min three-player model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from time import monotonic

from .common import (
    MAX_GAME_MOVES_EXCEEDED,
    PendingSample,
    SelfPlayEpisode,
    SelfPlayGame,
    sparse_policy,
)
from ..config import MCTSConfig, SelfPlayConfig
from ..deadline import MAX_GAME_TIME_EXCEEDED, Deadline
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
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if compatibility.identity.model_name != MIN_MODEL_NAME:
            raise ValueError("MinSelfPlayRunner requires Min compatibility metadata")
        self.game = game
        self.evaluator = evaluator
        self.mcts_config = mcts_config
        self.selfplay_config = selfplay_config
        self.compatibility = compatibility
        self.clock = clock

    def run(self) -> SelfPlayEpisode:
        state = self.game.initial_state()
        pending: list[PendingSample] = []
        move_count = 0
        deadline = Deadline.start(self.selfplay_config.max_game_seconds, clock=self.clock)
        while not self.game.is_terminal(state) and move_count < self.selfplay_config.max_moves:
            # A game that outran its wall-clock budget contributes nothing, the
            # same as any other aborted game.  Checked before the move so the
            # abort is immediate rather than one full search late.
            if deadline is not None and deadline.expired:
                return SelfPlayEpisode((), None, move_count, False, MAX_GAME_TIME_EXCEEDED)
            temperature = (
                self.selfplay_config.temperature
                if move_count < self.selfplay_config.temperature_moves
                else 0.0
            )
            search = MCTS3P(
                self.game,
                self.evaluator,
                replace(self.mcts_config, seed=self.selfplay_config.seed + move_count),
                deadline=deadline,
            )
            result = search.run(state, temperature=temperature)
            pending.append(
                PendingSample(self.game.evaluation_request(state), sparse_policy(result.policy))
            )
            state = self.game.apply_action(state, result.selected_action)
            move_count += 1

        if not self.game.is_terminal(state):
            return SelfPlayEpisode((), None, move_count, False, MAX_GAME_MOVES_EXCEEDED)

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
