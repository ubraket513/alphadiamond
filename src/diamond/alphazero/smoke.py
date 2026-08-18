"""Executable CPU smoke checks for the AlphaZero correctness baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from tempfile import TemporaryDirectory

from .arena import MinArena, SooArena
from .checkpoint import load_checkpoint, save_checkpoint
from .config import ArenaConfig, MCTSConfig, NetworkConfig, SelfPlayConfig, TrainingConfig
from .evaluator.base import EvalRequest, EvalResult
from .evaluator.dummy import DummyEvaluator
from .game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from .identity import CheckpointCompatibilitySpec
from .network import MinModel, SooModel
from .replay import ReplayBatch
from .selfplay.runner_2p import SooSelfPlayRunner
from .selfplay.runner_3p import MinSelfPlayRunner
from .trainer import AlphaZeroTrainer
from ..game.board import standard_board
from ..game.rules import find_legal_move
from ..game.state import EMPTY, GameState, PlayerSpec, build_players


class _PreferredActionEvaluator:
    def __init__(
        self,
        preferred_by_player: dict[int, int],
        value: float | tuple[float, ...],
    ) -> None:
        self.preferred_by_player = preferred_by_player
        self.value = value

    def evaluate(self, requests: tuple[EvalRequest, ...]) -> tuple[EvalResult, ...]:
        results: list[EvalResult] = []
        for request in requests:
            preferred = self.preferred_by_player[request.canonical_player_ids[0]]
            if preferred not in request.legal_action_ids:
                raise ValueError("authoritative smoke finishing action is not legal")
            priors = {action: 0.0 for action in request.legal_action_ids}
            priors[preferred] = 1.0
            results.append(EvalResult(priors, self.value))
        return tuple(results)


def _near_terminal_game(
    players: tuple[PlayerSpec, ...], finishers: int
) -> tuple[DiamondSearchAdapter, dict[int, int]]:
    board = standard_board()
    occupied = [EMPTY] * len(board)
    reserved_targets = {
        position
        for player in players[:finishers]
        for position in board.camp_positions(player.target_camp)
    }
    physical_actions: dict[int, int] = {}
    used_entries: set[int] = set()
    destinations: dict[int, int] = {}
    entries: dict[int, int] = {}

    for player in players[:finishers]:
        target = board.camp_positions(player.target_camp)
        choice: tuple[int, int] | None = None
        for destination in target:
            for neighbour in board.neighbours(destination):
                if (
                    neighbour is not None
                    and neighbour not in reserved_targets
                    and neighbour not in used_entries
                ):
                    choice = (neighbour, destination)
                    break
            if choice is not None:
                break
        if choice is None:
            raise RuntimeError("could not construct near-terminal Diamond smoke state")
        entry, destination = choice
        used_entries.add(entry)
        destinations[player.id] = destination
        entries[player.id] = entry
        for position in target:
            if position != destination:
                occupied[position] = player.id
        occupied[entry] = player.id

    state = GameState(
        occupancy=tuple(occupied),
        current_player_id=players[0].id,
        turn_number=40,
    )
    game = AlphaZeroGameAdapter(players, board=board, initial=state)
    for player in players[:finishers]:
        move = find_legal_move(
            board,
            state,
            entries[player.id],
            destinations[player.id],
            player_id=player.id,
        )
        if move is None:
            raise RuntimeError("constructed smoke finishing move is not authoritative")
        physical = game.codec.encode(move.source, move.destination)
        physical_actions[player.id] = game.encoder.to_canonical_action(
            physical, players, player.id
        )
    return DiamondSearchAdapter(game), physical_actions


def run_selfplay_smoke() -> dict[str, dict[str, object]]:
    mcts = MCTSConfig(simulations=1, dirichlet_epsilon=0.0, seed=7)
    selfplay = SelfPlayConfig(max_moves=2, temperature_moves=0, seed=7)

    soo_players = build_players(2)
    soo_game, soo_actions = _near_terminal_game(soo_players, finishers=1)
    soo = SooSelfPlayRunner(
        soo_game,
        _PreferredActionEvaluator(soo_actions, 0.0),
        mcts,
        selfplay,
        CheckpointCompatibilitySpec.soo(
            model_version="0.1.0", network_config=NetworkConfig()
        ),
    ).run()

    min_players = build_players(3)
    min_game, min_actions = _near_terminal_game(min_players, finishers=2)
    min_episode = MinSelfPlayRunner(
        min_game,
        _PreferredActionEvaluator(min_actions, (0.0, 0.0, 0.0)),
        mcts,
        selfplay,
        CheckpointCompatibilitySpec.min(
            model_version="0.1.0", network_config=NetworkConfig()
        ),
    ).run()
    return {
        "Soo": {
            "completed": soo.completed,
            "moves": soo.move_count,
            "final_order": list(soo.final_order or ()),
            "samples": len(soo.samples),
        },
        "Min": {
            "completed": min_episode.completed,
            "moves": min_episode.move_count,
            "final_order": list(min_episode.final_order or ()),
            "samples": len(min_episode.samples),
        },
    }


def _training_batch(player_count: int) -> ReplayBatch:
    features = player_count * 2
    policy = [0.0] * 5329
    policy[73] = 1.0
    value = (1.0,) if player_count == 2 else (1.0, 0.0, -1.0)
    return ReplayBatch(
        node_features=(tuple((0.0,) * features for _ in range(73)),),
        policy_targets=(tuple(policy),),
        value_targets=(value,),
    )


def _new_trainers() -> dict[str, AlphaZeroTrainer]:
    network = NetworkConfig(width=16, residual_blocks=1)
    training = TrainingConfig(batch_size=1, seed=13)
    soo_spec = CheckpointCompatibilitySpec.soo(
        model_version="0.1.0", network_config=network
    )
    min_spec = CheckpointCompatibilitySpec.min(
        model_version="0.1.0", network_config=network
    )
    return {
        "Soo": AlphaZeroTrainer(SooModel(network), soo_spec, training),
        "Min": AlphaZeroTrainer(MinModel(network), min_spec, training),
    }


def run_training_smoke() -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for name, trainer in _new_trainers().items():
        metrics = trainer.train_batch(
            _training_batch(trainer.compatibility.identity.player_count)
        )
        result[name] = {
            "training_step": trainer.training_step,
            "total_loss": metrics.total_loss,
        }
    return result


def run_checkpoint_smoke() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    with TemporaryDirectory() as directory:
        for name, trainer in _new_trainers().items():
            trainer.train_batch(
                _training_batch(trainer.compatibility.identity.player_count)
            )
            path = save_checkpoint(f"{directory}/{name}.pt", trainer)
            network = trainer.compatibility.network_config
            restored_model = (
                SooModel(network) if name == "Soo" else MinModel(network)
            )
            restored = AlphaZeroTrainer(
                restored_model,
                trainer.compatibility,
                TrainingConfig(batch_size=1),
            )
            info = load_checkpoint(path, restored, expected=trainer.compatibility)
            result[name] = {
                "model_version": info.metadata["model_version"],
                "training_step": info.training_step,
            }
    return result


@dataclass(frozen=True, slots=True)
class _ArenaState:
    terminal: bool
    player_id: int


class _ArenaGame:
    def __init__(self, order: tuple[int, ...], final_order: tuple[int, ...]) -> None:
        self.order = order
        self._final_order = final_order

    def initial_state(self) -> _ArenaState:
        return _ArenaState(False, self.order[0])

    def current_player_id(self, state: _ArenaState) -> int:
        return state.player_id

    def legal_action_ids(self, state: _ArenaState) -> tuple[int, ...]:
        return () if state.terminal else (5,)

    def apply_action(self, state: _ArenaState, action_id: int) -> _ArenaState:
        if action_id != 5:
            raise ValueError("unexpected arena smoke action")
        return _ArenaState(True, self.order[1])

    def is_terminal(self, state: _ArenaState) -> bool:
        return state.terminal

    def evaluation_request(self, state: _ArenaState) -> EvalRequest:
        start = self.order.index(state.player_id)
        canonical = tuple(
            self.order[(start + offset) % len(self.order)]
            for offset in range(len(self.order))
        )
        return EvalRequest(((0.0,),), self.legal_action_ids(state), canonical)

    def terminal_scalar_value(self, state: _ArenaState, player_id: int) -> float:
        return 1.0 if player_id == self._final_order[0] else -1.0

    def terminal_vector_value(self, state: _ArenaState) -> dict[int, float]:
        return dict(zip(self._final_order, (1.0, 0.0, -1.0)))

    def final_order(self, state: _ArenaState) -> tuple[int, ...]:
        if not state.terminal:
            raise ValueError("arena smoke game is not terminal")
        return self._final_order


def run_arena_smoke() -> dict[str, dict[str, int]]:
    search = MCTSConfig(simulations=2, dirichlet_epsilon=0.0)
    soo = SooArena(
        candidate=DummyEvaluator(0.0),
        baseline=DummyEvaluator(0.0),
        mcts_config=search,
        arena_config=ArenaConfig(games=2),
    ).run(lambda order: _ArenaGame(order, (1, 2)))
    min_result = MinArena(
        candidate=DummyEvaluator((0.0, 0.0, 0.0)),
        baseline=DummyEvaluator((0.0, 0.0, 0.0)),
        mcts_config=search,
        arena_config=ArenaConfig(games=6),
    ).run(lambda order: _ArenaGame(order, (1, 2, 3)))
    return {
        "Soo": {
            "wins": soo.wins,
            "losses": soo.losses,
            "aborted": soo.aborted_games,
        },
        "Min": {
            "first": min_result.first_places,
            "second": min_result.second_places,
            "third": min_result.third_places,
            "aborted": min_result.aborted_games,
        },
    }


def main() -> int:
    result = {
        "selfplay": run_selfplay_smoke(),
        "training": run_training_smoke(),
        "checkpoint": run_checkpoint_smoke(),
        "arena": run_arena_smoke(),
    }
    print(json.dumps(result, indent=2))
    return 0 if all(
        entry["completed"] for entry in result["selfplay"].values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "run_arena_smoke",
    "run_checkpoint_smoke",
    "run_selfplay_smoke",
    "run_training_smoke",
]
