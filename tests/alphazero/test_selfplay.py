from __future__ import annotations

from dataclasses import dataclass

from diamond.alphazero.config import MCTSConfig, NetworkConfig, SelfPlayConfig
from diamond.alphazero.evaluator.base import EvalRequest
from diamond.alphazero.evaluator.dummy import DummyEvaluator
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.selfplay.runner_2p import SooSelfPlayRunner
from diamond.alphazero.selfplay.runner_3p import MinSelfPlayRunner
from diamond.game.state import build_players


@dataclass(frozen=True)
class ToyState:
    step: int
    player_id: int


class ToySelfPlayGame:
    def __init__(self, player_count: int) -> None:
        self.player_count = player_count
        self.players = tuple(range(1, player_count + 1))

    def initial_state(self) -> ToyState:
        return ToyState(0, 1)

    def current_player_id(self, state: ToyState) -> int:
        return state.player_id

    def legal_action_ids(self, state: ToyState) -> tuple[int, ...]:
        return () if self.is_terminal(state) else (10 + state.step,)

    def apply_action(self, state: ToyState, action_id: int) -> ToyState:
        assert action_id == 10 + state.step
        next_player = state.player_id % self.player_count + 1
        return ToyState(state.step + 1, next_player)

    def is_terminal(self, state: ToyState) -> bool:
        return state.step >= self.player_count - 1

    def evaluation_request(self, state: ToyState) -> EvalRequest:
        start = self.players.index(state.player_id)
        canonical = tuple(
            self.players[(start + offset) % self.player_count]
            for offset in range(self.player_count)
        )
        features = tuple(
            tuple(float(state.step) for _ in range(self.player_count * 2))
            for _ in range(73)
        )
        return EvalRequest(features, self.legal_action_ids(state), canonical)

    def terminal_scalar_value(self, state: ToyState, player_id: int) -> float:
        return 1.0 if player_id == 1 else -1.0

    def terminal_vector_value(self, state: ToyState) -> dict[int, float]:
        return {1: 1.0, 2: 0.0, 3: -1.0}

    def final_order(self, state: ToyState) -> tuple[int, ...]:
        assert self.is_terminal(state)
        return self.players


def test_soo_selfplay_assigns_scalar_target_from_each_samples_perspective() -> None:
    compatibility = CheckpointCompatibilitySpec.soo(
        model_version="0.1.0", network_config=NetworkConfig(width=16, residual_blocks=1)
    )
    episode = SooSelfPlayRunner(
        ToySelfPlayGame(2),
        DummyEvaluator(0.0),
        MCTSConfig(simulations=4, dirichlet_epsilon=0.0),
        SelfPlayConfig(max_moves=5, temperature_moves=0),
        compatibility,
    ).run()

    assert episode.completed
    assert episode.final_order == (1, 2)
    assert len(episode.samples) == 1
    assert episode.samples[0].value_target == (1.0,)
    assert episode.samples[0].model_name == "Soo"


def test_min_selfplay_keeps_states_after_first_place_and_maps_final_utility() -> None:
    compatibility = CheckpointCompatibilitySpec.min(
        model_version="0.7.0", network_config=NetworkConfig(width=16, residual_blocks=1)
    )
    episode = MinSelfPlayRunner(
        ToySelfPlayGame(3),
        DummyEvaluator((0.0, 0.0, 0.0)),
        MCTSConfig(simulations=4, dirichlet_epsilon=0.0),
        SelfPlayConfig(max_moves=5, temperature_moves=0),
        compatibility,
    ).run()

    assert episode.completed
    assert len(episode.samples) == 2
    assert episode.samples[0].canonical_player_ids == (1, 2, 3)
    assert episode.samples[0].value_target == (1.0, 0.0, -1.0)
    assert episode.samples[1].canonical_player_ids == (2, 3, 1)
    assert episode.samples[1].value_target == (0.0, -1.0, 1.0)


def test_aborted_authoritative_game_discards_unlabelled_samples() -> None:
    game = DiamondSearchAdapter(AlphaZeroGameAdapter(build_players(2)))
    compatibility = CheckpointCompatibilitySpec.soo(
        model_version="0.1.0", network_config=NetworkConfig()
    )
    episode = SooSelfPlayRunner(
        game,
        DummyEvaluator(0.0),
        MCTSConfig(simulations=1, dirichlet_epsilon=0.0),
        SelfPlayConfig(max_moves=1, temperature_moves=0),
        compatibility,
    ).run()

    assert not episode.completed
    assert episode.aborted_reason == "max_game_moves_exceeded"
    assert episode.samples == ()
    assert episode.move_count == 1
