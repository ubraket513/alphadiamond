"""Bootstrap self-play keeps terminal and training semantics unchanged."""

from __future__ import annotations

from dataclasses import replace

import pytest

from diamond.alphazero.bootstrap.evaluator import (
    BootstrapPriorEvaluator,
    VacancyPriorEvaluator,
)
from diamond.alphazero.bootstrap.heuristic import (
    BOOTSTRAP_PRIOR_NONE,
    CANONICAL_TARGET_DISTANCE_V1,
)
from diamond.alphazero.config import MCTSConfig, NetworkConfig, SelfPlayConfig
from diamond.alphazero.evaluator.dummy import DummyEvaluator
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.orchestration.selfplay_workers import EpisodeResult
from diamond.alphazero.selfplay.runner_2p import SooSelfPlayRunner
from diamond.alphazero.selfplay.runner_3p import MinSelfPlayRunner
from diamond.game.state import build_players


def soo_compatibility() -> CheckpointCompatibilitySpec:
    return CheckpointCompatibilitySpec.soo(
        model_version="0.1.0", network_config=NetworkConfig()
    )


def min_compatibility() -> CheckpointCompatibilitySpec:
    return CheckpointCompatibilitySpec.min(
        model_version="0.7.0", network_config=NetworkConfig()
    )


def soo_episode(evaluator, *, max_moves: int, seed: int = 0, dirichlet: float | None = None):
    """One Soo episode.  ``dirichlet=0.0`` disables root exploration noise."""
    mcts = MCTSConfig(simulations=1, seed=seed)
    if dirichlet is not None:
        mcts = replace(mcts, dirichlet_epsilon=dirichlet)
    return SooSelfPlayRunner(
        DiamondSearchAdapter(AlphaZeroGameAdapter(build_players(2))),
        evaluator,
        mcts,
        SelfPlayConfig(max_moves=max_moves, temperature_moves=0, seed=seed),
        soo_compatibility(),
    ).run()


def test_bootstrap_disabled_follows_the_existing_path() -> None:
    """An unwrapped evaluator behaves exactly as it does today."""
    plain = soo_episode(DummyEvaluator(0.0), max_moves=1, dirichlet=0.0)
    assert not plain.completed
    assert plain.aborted_reason == "max_game_moves_exceeded"
    assert plain.samples == ()


def test_aborted_episodes_still_produce_zero_samples() -> None:
    episode = soo_episode(
        BootstrapPriorEvaluator(DummyEvaluator(0.0)), max_moves=2, dirichlet=0.0
    )
    assert not episode.completed
    assert episode.aborted_reason == "max_game_moves_exceeded"
    assert episode.samples == ()


def test_bootstrap_soo_reaches_a_real_terminal_and_keeps_value_semantics() -> None:
    episode = soo_episode(VacancyPriorEvaluator(DummyEvaluator(0.0)), max_moves=400)
    assert episode.completed, "v2 is expected to finish Soo inside 400 moves"
    assert episode.final_order is not None
    winner = episode.final_order[0]
    assert episode.samples
    for sample in episode.samples:
        # Unchanged Soo semantics: current-player scalar win/loss.
        expected = 1.0 if sample.canonical_player_ids[0] == winner else -1.0
        assert sample.value_target == (expected,)


def test_bootstrap_min_keeps_placement_utility_semantics() -> None:
    episode = MinSelfPlayRunner(
        DiamondSearchAdapter(AlphaZeroGameAdapter(build_players(3))),
        VacancyPriorEvaluator(DummyEvaluator((0.0, 0.0, 0.0))),
        MCTSConfig(simulations=1, seed=0),
        SelfPlayConfig(max_moves=2000, temperature_moves=0, seed=0),
        min_compatibility(),
    ).run()
    assert episode.completed, "v2 is expected to finish Min inside 2000 moves"
    assert episode.samples
    for sample in episode.samples:
        assert sorted(sample.value_target) == [-1.0, 0.0, 1.0]


def test_every_authoritative_legal_action_keeps_a_prior() -> None:
    game = DiamondSearchAdapter(AlphaZeroGameAdapter(build_players(2)))
    state = game.initial_state()
    request = game.evaluation_request(state)
    result = BootstrapPriorEvaluator(DummyEvaluator(0.0)).evaluate((request,))
    assert tuple(sorted(result[0].priors)) == tuple(sorted(request.legal_action_ids))
    assert all(value > 0.0 for value in result[0].priors.values())


def test_episode_result_records_self_play_provenance() -> None:
    result = EpisodeResult(
        game_id="game-1",
        seed=0,
        retry_id="retry-1",
        model_key=None,  # type: ignore[arg-type]
        compatibility=soo_compatibility(),
        samples=(),
        final_order=None,
        move_count=1,
        completed=False,
        aborted_reason="max_game_moves_exceeded",
        bootstrap_prior=CANONICAL_TARGET_DISTANCE_V1,
    )
    assert result.bootstrap_prior == CANONICAL_TARGET_DISTANCE_V1


def test_episode_result_defaults_to_no_bootstrap_prior() -> None:
    result = EpisodeResult(
        game_id="game-1",
        seed=0,
        retry_id="retry-1",
        model_key=None,  # type: ignore[arg-type]
        compatibility=soo_compatibility(),
        samples=(),
        final_order=None,
        move_count=1,
        completed=False,
        aborted_reason="max_game_moves_exceeded",
    )
    assert result.bootstrap_prior == BOOTSTRAP_PRIOR_NONE


def test_selfplay_config_rejects_unknown_prior() -> None:
    with pytest.raises(ValueError):
        SelfPlayConfig(bootstrap_prior="target-distance-v2")


def test_arena_config_has_no_bootstrap_surface() -> None:
    """Arena, Elo and TrueSkill paths cannot opt into the heuristic."""
    from diamond.alphazero.config import ArenaConfig

    assert not hasattr(ArenaConfig(), "bootstrap_prior")
