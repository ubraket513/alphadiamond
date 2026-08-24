"""Single-process self-play for the Soo two-player model.

The episode loop is control plane -- temperature schedule, sample construction,
the wall-clock abort -- and stays in Python. The *search* is not: it comes from
``search_factory``, which returns the native two-seat search. A caller with a
game the native core cannot play (a stub, a toy) injects its own factory rather
than getting a second engine by default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from time import monotonic

from ..config import MCTSConfig, SelfPlayConfig
from ..deadline import MAX_GAME_TIME_EXCEEDED, Deadline
from ..evaluator.base import Evaluator
from ..identity import SOO_MODEL_NAME, CheckpointCompatibilitySpec
from ..replay import TrainingSample
from ..search_factory import SearchFactory, two_player_search
from .common import (
    MAX_GAME_MOVES_EXCEEDED,
    PendingSample,
    SelfPlayEpisode,
    SelfPlayGame,
    sparse_policy,
)


class SooSelfPlayRunner:
    def __init__(
        self,
        game: SelfPlayGame,
        evaluator: Evaluator,
        mcts_config: MCTSConfig,
        selfplay_config: SelfPlayConfig,
        compatibility: CheckpointCompatibilitySpec,
        *,
        clock: Callable[[], float] = monotonic,
        search_factory: SearchFactory | None = None,
    ) -> None:
        if compatibility.identity.model_name != SOO_MODEL_NAME:
            raise ValueError("SooSelfPlayRunner requires Soo compatibility metadata")
        self.game = game
        self.evaluator = evaluator
        self.mcts_config = mcts_config
        self.selfplay_config = selfplay_config
        self.compatibility = compatibility
        self.clock = clock
        self.search_factory = search_factory or two_player_search()

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
            search = self.search_factory(
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
        winner = final_order[0]
        samples = tuple(
            TrainingSample(
                compatibility=self.compatibility,
                node_features=row.request.node_features,
                canonical_player_ids=row.request.canonical_player_ids,
                sparse_policy=row.sparse_policy,
                value_target=(1.0 if row.request.canonical_player_ids[0] == winner else -1.0,),
            )
            for row in pending
        )
        return SelfPlayEpisode(samples, final_order, move_count, True)


__all__ = ["SooSelfPlayRunner"]
