"""Deterministic balanced Soo/Min candidate-versus-baseline arenas.

The Soo arena runs its searches on the C++ core wherever the extension is
importable: the arena plays whole games, so it is one of the largest remaining
consumers of the Python engine. It is the same search either way -- the native
tree answers through the same Python evaluator, and
``tests/native/test_arena_search_parity.py`` holds the two to the same selected
action and the same visit counts.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from .config import ArenaConfig, MCTSConfig
from .evaluator.base import Evaluator
from .search_factory import SearchFactory, three_player_search, two_player_search

GameFactory = Callable[[tuple[int, ...]], Any]
def _balanced_matchups(
    player_ids: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Cross every fixed seat assignment with every possible turn order."""
    return tuple(
        (order, candidate_player)
        for order in itertools.permutations(player_ids)
        for candidate_player in player_ids
    )


@dataclass(frozen=True, slots=True)
class SooArenaResult:
    wins: int
    losses: int
    aborted_games: int
    win_rate: float
    promoted: bool


@dataclass(frozen=True, slots=True)
class MinArenaResult:
    first_places: int
    second_places: int
    third_places: int
    aborted_games: int
    mean_utility: float
    promoted: bool


def _validate_configs(mcts: MCTSConfig, arena: ArenaConfig) -> MCTSConfig:
    if arena.games <= 0 or arena.max_moves <= 0:
        raise ValueError("arena games and max_moves must be positive")
    if not 0 <= arena.promotion_threshold <= 1:
        raise ValueError("arena promotion_threshold must be in [0, 1]")
    # Evaluation never inherits training root noise.
    return replace(mcts, dirichlet_epsilon=0.0)


class SooArena:
    def __init__(
        self,
        *,
        candidate: Evaluator,
        baseline: Evaluator,
        mcts_config: MCTSConfig,
        arena_config: ArenaConfig,
        search_factory: SearchFactory | None = None,
    ) -> None:
        self.candidate = candidate
        self.baseline = baseline
        self.mcts_config = _validate_configs(mcts_config, arena_config)
        if arena_config.games % len(_balanced_matchups((1, 2))) != 0:
            raise ValueError("Soo arena games must be a multiple of 4")
        self.arena_config = arena_config
        self.search_factory = search_factory or two_player_search()

    def run(self, game_factory: GameFactory) -> SooArenaResult:
        wins = losses = aborted = 0
        matchups = _balanced_matchups((1, 2))
        for game_index in range(self.arena_config.games):
            order, candidate_player = matchups[game_index % len(matchups)]
            game = game_factory(order)
            state = game.initial_state()
            moves = 0
            while not game.is_terminal(state) and moves < self.arena_config.max_moves:
                evaluator = (
                    self.candidate
                    if game.current_player_id(state) == candidate_player
                    else self.baseline
                )
                config = replace(
                    self.mcts_config,
                    seed=self.arena_config.seed + game_index * self.arena_config.max_moves + moves,
                )
                action = (
                    self.search_factory(game, evaluator, config)
                    .run(state, temperature=0.0)
                    .selected_action
                )
                state = game.apply_action(state, action)
                moves += 1
            if not game.is_terminal(state):
                aborted += 1
                continue
            if game.final_order(state)[0] == candidate_player:
                wins += 1
            else:
                losses += 1
        completed = wins + losses
        win_rate = wins / completed if completed else 0.0
        return SooArenaResult(
            wins=wins,
            losses=losses,
            aborted_games=aborted,
            win_rate=win_rate,
            promoted=completed > 0 and win_rate >= self.arena_config.promotion_threshold,
        )


class MinArena:
    def __init__(
        self,
        *,
        candidate: Evaluator,
        baseline: Evaluator,
        mcts_config: MCTSConfig,
        arena_config: ArenaConfig,
        search_factory: SearchFactory | None = None,
    ) -> None:
        self.candidate = candidate
        self.baseline = baseline
        self.mcts_config = _validate_configs(mcts_config, arena_config)
        if arena_config.games % len(_balanced_matchups((1, 2, 3))) != 0:
            raise ValueError("Min arena games must be a multiple of 18")
        self.arena_config = arena_config
        self.search_factory = search_factory or three_player_search()

    def run(self, game_factory: GameFactory) -> MinArenaResult:
        matchups = _balanced_matchups((1, 2, 3))
        placements = [0, 0, 0]
        aborted = 0
        for game_index in range(self.arena_config.games):
            order, candidate_player = matchups[game_index % len(matchups)]
            game = game_factory(order)
            state = game.initial_state()
            moves = 0
            while not game.is_terminal(state) and moves < self.arena_config.max_moves:
                evaluator = (
                    self.candidate
                    if game.current_player_id(state) == candidate_player
                    else self.baseline
                )
                config = replace(
                    self.mcts_config,
                    seed=self.arena_config.seed + game_index * self.arena_config.max_moves + moves,
                )
                action = (
                    self.search_factory(game, evaluator, config)
                    .run(state, temperature=0.0)
                    .selected_action
                )
                state = game.apply_action(state, action)
                moves += 1
            if not game.is_terminal(state):
                aborted += 1
                continue
            place = game.final_order(state).index(candidate_player)
            placements[place] += 1
        completed = sum(placements)
        mean_utility = (
            (placements[0] - placements[2]) / completed if completed else 0.0
        )
        return MinArenaResult(
            first_places=placements[0],
            second_places=placements[1],
            third_places=placements[2],
            aborted_games=aborted,
            mean_utility=mean_utility,
            promoted=(
                completed > 0
                and mean_utility >= self.arena_config.promotion_threshold
            ),
        )


__all__ = ["MinArena", "MinArenaResult", "SooArena", "SooArenaResult"]
