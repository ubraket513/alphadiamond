"""Tiny, deterministic, headless Milestone 2 production smoke runner."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from .arena import MinArena, SooArena
from .checkpoint import load_inference_checkpoint, save_checkpoint
from .config import ArenaConfig, MCTSConfig, NetworkConfig, SelfPlayConfig, TrainingConfig
from .evaluator.base import EvalRequest, EvalResult
from .identity import CheckpointCompatibilitySpec, MIN_MODEL_NAME, SOO_MODEL_NAME
from .inference.coordinator import InferenceConfig, InferenceCoordinator
from .inference.protocol import InferenceRequest, InferenceResponse, ModelKey
from .network import MinModel, SooModel
from .orchestration.coordinator import (
    CandidateArtifact,
    PersistenceConfig,
    PromotionArtifact,
    TrainingCoordinator,
    TrainingLoopConfig,
    TrainingStepArtifact,
    WorkerConfig,
)
from .orchestration.replay_store import PersistentReplayStore
from .orchestration.run_state import RunStage, RunStateStore, TrainingRunState
from .orchestration.selfplay_workers import EpisodeResult, SelfPlayJob, SelfPlayWorkerPool
from .rating.events import SooRatingEvent
from .rating.participants import CheckpointParticipant
from .rating.protocol import BenchmarkProtocol, EloConfig, TrueSkillConfig
from .rating.registry import RatingRegistry
from .trainer import AlphaZeroTrainer
from ..game.board import standard_board
from ..game.rules import find_legal_move
from ..game.state import EMPTY, GameState, PlayerSpec, build_players
from .game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter

_NETWORK = NetworkConfig(width=8, residual_blocks=1)
_TRAINING = TrainingConfig(batch_size=1, learning_rate=1e-3, weight_decay=0.0, seed=17)


def _near_terminal_state(
    players: tuple[PlayerSpec, ...],
    *,
    finishers: int,
) -> tuple[GameState, tuple[tuple[int, int], ...]]:
    """Build an authoritative state one move from each required placement."""
    board = standard_board()
    occupied = [EMPTY] * len(board)
    reserved_targets = {
        position
        for player in players[:finishers]
        for position in board.camp_positions(player.target_camp)
    }
    used_entries: set[int] = set()
    entries: dict[int, int] = {}
    destinations: dict[int, int] = {}
    for player in players[:finishers]:
        choice: tuple[int, int] | None = None
        for destination in board.camp_positions(player.target_camp):
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
            raise RuntimeError("unable to create a near-terminal authoritative game")
        entry, destination = choice
        used_entries.add(entry)
        entries[player.id] = entry
        destinations[player.id] = destination
        for position in board.camp_positions(player.target_camp):
            if position != destination:
                occupied[position] = player.id
        occupied[entry] = player.id

    state = GameState(tuple(occupied), players[0].id, 40)
    adapter = AlphaZeroGameAdapter(players, board=board, initial=state)
    actions: list[tuple[int, int]] = []
    for player in players[:finishers]:
        move = find_legal_move(
            board,
            state,
            entries[player.id],
            destinations[player.id],
            player_id=player.id,
        )
        if move is None:
            raise RuntimeError("near-terminal move was rejected by authoritative rules")
        physical = adapter.codec.encode(move.source, move.destination)
        actions.append(
            (player.id, adapter.encoder.to_canonical_action(physical, players, player.id))
        )
    return state, tuple(actions)


class _PreferredBatchEvaluator:
    """A deterministic evaluator fixture behind the real central coordinator."""

    def __init__(self, actions: Mapping[tuple[str, int], int]) -> None:
        self.actions = dict(actions)

    def evaluate(self, requests: tuple[InferenceRequest, ...]) -> tuple[InferenceResponse, ...]:
        responses: list[InferenceResponse] = []
        for request in requests:
            preferred = self.actions.get(
                (request.model_key.model_name, request.canonical_player_ids[0]),
                request.legal_action_ids[0],
            )
            if preferred not in request.legal_action_ids:
                preferred = request.legal_action_ids[0]
            priors = {
                action: 1.0 if action == preferred else 0.0
                for action in request.legal_action_ids
            }
            value: float | tuple[float, ...]
            value = 0.0 if request.model_key.player_count == 2 else (0.0, 0.0, 0.0)
            responses.append(
                InferenceResponse.from_eval_result(request, EvalResult(priors, value))
            )
        return tuple(responses)


class _FinishingArenaEvaluator:
    """Tiny arena evaluator configured with authoritative finishing actions."""

    def __init__(self, value_size: int, actions: Mapping[tuple[int, ...], int]) -> None:
        self.value_size = value_size
        self.actions = dict(actions)

    def evaluate(self, requests: tuple[EvalRequest, ...]) -> tuple[EvalResult, ...]:
        value: float | tuple[float, ...]
        value = 0.0 if self.value_size == 1 else (0.0, 0.0, 0.0)
        return tuple(
            EvalResult({
                action: 1.0
                if action == self.actions.get(request.canonical_player_ids, request.legal_action_ids[0])
                else 0.0
                for action in request.legal_action_ids
            }, value)
            for request in requests
        )


class _WorkerStage:
    def __init__(self, actions: Mapping[tuple[str, int], int], worker_count: int) -> None:
        self.actions = dict(actions)
        self.worker_count = worker_count
        self.artifacts: dict[str, tuple[EpisodeResult, ...]] = {}

    def load(self, operation_id: str) -> tuple[EpisodeResult, ...] | None:
        return self.artifacts.get(operation_id)

    def execute(
        self, operation_id: str, jobs: tuple[SelfPlayJob, ...]
    ) -> tuple[EpisodeResult, ...]:
        coordinator = InferenceCoordinator(
            _PreferredBatchEvaluator(self.actions),
            InferenceConfig(
                max_batch_size=2,
                max_wait_ms=1,
                request_queue_capacity=8,
                response_timeout_s=10.0,
            ),
        )
        coordinator.start()
        try:
            episodes = SelfPlayWorkerPool(
                coordinator,
                worker_count=self.worker_count,
                worker_timeout_s=20.0,
                join_timeout_s=2.0,
            ).run(jobs)
        finally:
            coordinator.stop()
        self.artifacts[operation_id] = episodes
        return episodes


class _TorchTrainingStage:
    def __init__(self, trainer: AlphaZeroTrainer) -> None:
        self.trainer = trainer
        self.artifacts: dict[str, TrainingStepArtifact] = {}

    def load(self, operation_id: str) -> TrainingStepArtifact | None:
        return self.artifacts.get(operation_id)

    def execute(
        self,
        operation_id: str,
        replay: PersistentReplayStore,
        batch_size: int,
        expected_training_step: int,
    ) -> TrainingStepArtifact:
        if self.trainer.training_step != expected_training_step:
            raise ValueError("trainer step does not match authoritative run state")
        samples = replay.sample(batch_size)
        metrics = self.trainer.train_batch(
            replay.load_buffer().collate(samples, action_size=73 * 73)
        )
        artifact = TrainingStepArtifact(
            operation_id=operation_id,
            compatibility_namespace=_compatibility_namespace(self.trainer.compatibility),
            input_training_step=expected_training_step,
            output_training_step=self.trainer.training_step,
            metrics=asdict(metrics),
        )
        self.artifacts[operation_id] = artifact
        return artifact


class _CheckpointStage:
    def __init__(self, trainer: AlphaZeroTrainer) -> None:
        self.trainer = trainer
        self.artifacts: dict[str, CandidateArtifact] = {}
        self.read_only_loads = 0

    def load(self, operation_id: str) -> CandidateArtifact | None:
        return self.artifacts.get(operation_id)

    def execute(
        self, operation_id: str, path: Path, training: TrainingStepArtifact
    ) -> CandidateArtifact:
        save_checkpoint(path, self.trainer)
        model = _new_model(self.trainer.compatibility)
        load_inference_checkpoint(path, model, expected=self.trainer.compatibility, device="cpu")
        self.read_only_loads += 1
        participant = CheckpointParticipant.from_checkpoint(path)
        artifact = CandidateArtifact(
            operation_id=operation_id,
            path=path,
            checkpoint_sha256=participant.checkpoint_sha256,
            training_step=training.output_training_step,
            compatibility_namespace=training.compatibility_namespace,
            participant=participant,
        )
        self.artifacts[operation_id] = artifact
        return artifact


class _ArenaStage:
    def __init__(self, compatibility: CheckpointCompatibilitySpec, protocol_id: str) -> None:
        self._compatibility = compatibility
        self._protocol_id = protocol_id
        self.artifacts: dict[str, PromotionArtifact] = {}

    @property
    def compatibility(self) -> CheckpointCompatibilitySpec:
        return self._compatibility

    @property
    def protocol_id(self) -> str:
        return self._protocol_id

    def load(self, operation_id: str) -> PromotionArtifact | None:
        return self.artifacts.get(operation_id)

    def execute(
        self,
        operation_id: str,
        candidate: CandidateArtifact,
        champion_checkpoint: str | None,
    ) -> PromotionArtifact:
        player_count = self.compatibility.identity.player_count
        evaluator = _FinishingArenaEvaluator(
            1 if player_count == 2 else 3, _arena_finishing_actions(player_count)
        )
        if player_count == 2:
            result = SooArena(
                candidate=evaluator,
                baseline=evaluator,
                mcts_config=MCTSConfig(simulations=1, dirichlet_epsilon=0.0),
                arena_config=ArenaConfig(games=4, seed=29, max_moves=2, promotion_threshold=0.0),
            ).run(_arena_game_factory(2))
        else:
            result = MinArena(
                candidate=evaluator,
                baseline=evaluator,
                mcts_config=MCTSConfig(simulations=1, dirichlet_epsilon=0.0),
                arena_config=ArenaConfig(games=18, seed=29, max_moves=2, promotion_threshold=0.0),
            ).run(_arena_game_factory(3))
        artifact = PromotionArtifact(
            operation_id=operation_id,
            promotion_protocol_id=self.protocol_id,
            candidate_sha256=candidate.checkpoint_sha256,
            champion_checkpoint=champion_checkpoint,
            promoted=result.promoted,
            result=asdict(result),
        )
        self.artifacts[operation_id] = artifact
        return artifact


class _RatingStage:
    def __init__(
        self,
        protocol: BenchmarkProtocol,
        champion: CheckpointParticipant,
    ) -> None:
        self.protocol = protocol
        self.champion = champion
        self.artifacts: dict[str, tuple[SooRatingEvent, ...]] = {}
        self.status = "eligible"

    @property
    def compatibility(self) -> CheckpointCompatibilitySpec:
        return self.protocol.compatibility

    @property
    def protocol_id(self) -> str:
        return self.protocol.protocol_id

    def load(self, operation_id: str) -> tuple[SooRatingEvent, ...] | None:
        return self.artifacts.get(operation_id)

    def execute(
        self,
        operation_id: str,
        candidate: CandidateArtifact,
        champion_checkpoint: str | None,
    ) -> tuple[SooRatingEvent, ...]:
        if self.compatibility.identity.model_name == MIN_MODEL_NAME:
            # A Min event needs three distinct artifact identities.  The tiny
            # fixture deliberately has only champion and candidate; never pad
            # the triple with a duplicate or invent a historical rating.
            self.status = "insufficient_history"
            self.artifacts[operation_id] = ()
            return ()
        event = SooRatingEvent(
            sequence_index=0,
            protocol_id=self.protocol.protocol_id,
            participant_ids=(candidate.participant.participant_id, self.champion.participant_id),
            seat_assignment=(1, 2),
            turn_order=(1, 2),
            opening_id="tiny-authoritative-opening-v1",
            completed=True,
            winner_id=candidate.participant.participant_id,
            loser_id=self.champion.participant_id,
        )
        self.artifacts[operation_id] = (event,)
        return (event,)


def _compatibility_namespace(compatibility: CheckpointCompatibilitySpec) -> str:
    payload = json.dumps(
        compatibility.to_metadata(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _new_model(compatibility: CheckpointCompatibilitySpec) -> SooModel | MinModel:
    if compatibility.identity.model_name == SOO_MODEL_NAME:
        return SooModel(
            compatibility.network_config, model_version=compatibility.identity.model_version
        )
    return MinModel(
        compatibility.network_config, model_version=compatibility.identity.model_version
    )


def _arena_game_factory(player_count: int):
    def factory(order: tuple[int, ...]) -> DiamondSearchAdapter:
        players = build_players(player_count, order=order)
        state, _actions = _near_terminal_state(
            players, finishers=player_count - 1
        )
        return DiamondSearchAdapter(AlphaZeroGameAdapter(players, initial=state))

    return factory


def _arena_finishing_actions(player_count: int) -> dict[tuple[int, ...], int]:
    actions: dict[tuple[int, ...], int] = {}
    for order in itertools.permutations(range(1, player_count + 1)):
        players = build_players(player_count, order=order)
        state, finishing = _near_terminal_state(players, finishers=player_count - 1)
        request = DiamondSearchAdapter(AlphaZeroGameAdapter(players, initial=state)).evaluation_request(state)
        actions[request.canonical_player_ids] = dict(finishing)[state.current_player_id]
    return actions


def _build_jobs(
    state: TrainingRunState,
    config: WorkerConfig,
    compatibility: CheckpointCompatibilitySpec,
    model_key: ModelKey,
) -> tuple[SelfPlayJob, ...]:
    players = build_players(compatibility.identity.player_count)
    initial, _actions = _near_terminal_state(
        players, finishers=compatibility.identity.player_count - 1
    )
    return tuple(
        SelfPlayJob(
            run_seed=state.run_seed,
            iteration=state.iteration,
            game_index=index,
            retry_id=config.retry_id,
            model_key=model_key,
            compatibility=compatibility,
            players=players,
            initial_state=initial,
            mcts_config=MCTSConfig(simulations=1, dirichlet_epsilon=0.0),
            selfplay_config=SelfPlayConfig(max_moves=2, temperature_moves=0),
        )
        for index in range(config.games_per_iteration)
    )


class TinyTrainingServices:
    """Production-stage assembly used by the smoke and headless CLI defaults."""

    def __init__(self, root: Path, model_name: str) -> None:
        self.root = Path(root)
        if model_name not in (SOO_MODEL_NAME, MIN_MODEL_NAME):
            raise ValueError("model_name must be Soo or Min")
        self.model_name = model_name
        self.compatibility = (
            CheckpointCompatibilitySpec.soo(model_version="2.0.0", network_config=_NETWORK)
            if model_name == SOO_MODEL_NAME
            else CheckpointCompatibilitySpec.min(model_version="2.0.0", network_config=_NETWORK)
        )
        self.worker_config = WorkerConfig(worker_count=2, games_per_iteration=2)
        self.promotion_protocol_id = f"tiny-{model_name.lower()}-promotion-v1"
        self.protocol = BenchmarkProtocol(
            compatibility=self.compatibility,
            simulations=1,
            c_puct=1.5,
            dirichlet_epsilon=0.0,
            decision_temperature=0.0,
            max_game_moves=2,
            opening_suite_version="tiny-authoritative-openings-v1",
            opening_suite_hash="sha256:tiny-authoritative-openings",
            rating_system_version=(
                EloConfig().rating_system_version
                if model_name == SOO_MODEL_NAME
                else TrueSkillConfig().rating_system_version
            ),
            rating_parameters=asdict(EloConfig() if model_name == SOO_MODEL_NAME else TrueSkillConfig()),
        )

    def _runtime(self, run_id: str, *, create: bool) -> tuple[TrainingCoordinator, RunStateStore, _RatingStage, _CheckpointStage]:
        run_root = self.root / self.model_name.lower() / run_id
        bootstrap = run_root / "bootstrap.pt"
        if create:
            if bootstrap.exists():
                raise ValueError(f"training run already exists: {run_id}")
            trainer = AlphaZeroTrainer(_new_model(self.compatibility), self.compatibility, _TRAINING)
            save_checkpoint(bootstrap, trainer)
            champion = CheckpointParticipant.from_checkpoint(bootstrap)
            registry = RatingRegistry(self.protocol)
            registry.add_participant(champion)
        else:
            if not bootstrap.exists():
                raise ValueError(f"training run does not exist: {run_id} ({self.model_name})")
            champion = CheckpointParticipant.from_checkpoint(bootstrap)
            registry_path = run_root / "ratings" / "registry.json"
            registry = RatingRegistry.load(registry_path) if registry_path.exists() else RatingRegistry(self.protocol)
            if not registry.participants:
                registry.add_participant(champion)

        players = build_players(self.compatibility.identity.player_count)
        _state, actions = _near_terminal_state(
            players, finishers=self.compatibility.identity.player_count - 1
        )
        actions_by_model = {(self.model_name, player_id): action for player_id, action in actions}
        model_key = ModelKey(
            self.model_name,
            self.compatibility.identity.model_version,
            champion.checkpoint_sha256,
        )
        state_store = RunStateStore(self.root)
        replay = PersistentReplayStore(
            run_root / "replay", self.compatibility, capacity=8, seed=17
        )
        trainer = AlphaZeroTrainer(_new_model(self.compatibility), self.compatibility, _TRAINING)
        worker = _WorkerStage(actions_by_model, self.worker_config.worker_count)
        checkpoints = _CheckpointStage(trainer)
        rating = _RatingStage(self.protocol, champion)
        coordinator = TrainingCoordinator(
            state_store=state_store,
            compatibility=self.compatibility,
            worker_config=self.worker_config,
            loop_config=TrainingLoopConfig(
                replay_batch_size=1,
                promotion_protocol_id=self.promotion_protocol_id,
                benchmark_protocol_id=self.protocol.protocol_id,
            ),
            persistence_config=PersistenceConfig(root=self.root),
            build_selfplay_jobs=lambda state, config: _build_jobs(
                state, config, self.compatibility, model_key
            ),
            self_play=worker,
            replay_store=replay,
            training=_TorchTrainingStage(trainer),
            checkpoints=checkpoints,
            promotion=_ArenaStage(self.compatibility, self.promotion_protocol_id),
            benchmark=rating,
            rating_registry=registry,
        )
        return coordinator, state_store, rating, checkpoints

    def train(self, *, model_name: str, run_id: str) -> Mapping[str, object]:
        self._assert_model(model_name)
        coordinator, state_store, rating, checkpoints = self._runtime(run_id, create=True)
        initial = state_store.initialize(
            run_id=run_id,
            compatibility=self.compatibility,
            run_seed=17,
            protocol_ids={
                "promotion": self.promotion_protocol_id,
                "rating": self.protocol.protocol_id,
            },
            champion_checkpoint=str(self.root / self.model_name.lower() / run_id / "bootstrap.pt"),
        )
        complete = coordinator.run_iteration(initial)
        return self._summary(complete, rating, checkpoints)

    def resume(self, *, model_name: str, run_id: str) -> Mapping[str, object]:
        self._assert_model(model_name)
        coordinator, _state_store, rating, checkpoints = self._runtime(run_id, create=False)
        complete = coordinator.resume(run_id, self.model_name)
        return self._summary(complete, rating, checkpoints)

    def benchmark(self, *, model_name: str, run_id: str) -> Mapping[str, object]:
        self._assert_model(model_name)
        registry = self._load_registry(run_id)
        return {
            "events": len(registry.events),
            "rating_status": "eligible" if registry.has_sufficient_min_history or self.model_name == SOO_MODEL_NAME else "insufficient_history",
        }

    def leaderboard(self, *, model_name: str, run_id: str) -> Mapping[str, object]:
        self._assert_model(model_name)
        registry = self._load_registry(run_id)
        entries: list[dict[str, object]] = []
        if self.model_name == SOO_MODEL_NAME:
            entries = [asdict(entry) for entry in registry.soo_leaderboard()]
        else:
            entries = [asdict(entry) for entry in registry.min_leaderboard()]
        return {"entries": entries}

    def _load_registry(self, run_id: str) -> RatingRegistry:
        path = self.root / self.model_name.lower() / run_id / "ratings" / "registry.json"
        if not path.exists():
            raise ValueError(f"rating registry does not exist: {run_id} ({self.model_name})")
        return RatingRegistry.load(path)

    def _assert_model(self, model_name: str) -> None:
        if model_name != self.model_name:
            raise ValueError("command model does not match configured services")

    def _summary(
        self,
        state: TrainingRunState,
        rating: _RatingStage,
        checkpoints: _CheckpointStage,
    ) -> dict[str, object]:
        return {
            "candidate_checkpoint": state.candidate_checkpoint,
            "rating_events": len(state.rating_records[-1]["event_ids"]),
            "rating_status": rating.status,
            "read_only_checkpoint_loads": checkpoints.read_only_loads,
            "stage": state.stage.value,
            "training_step": state.training_step,
        }


def build_tiny_services(root: Path, model_name: str) -> TinyTrainingServices:
    """Build the only small, production-backed runtime used by this milestone."""
    return TinyTrainingServices(root, model_name)


def _run_model(root: Path, model_name: str) -> dict[str, object]:
    services = build_tiny_services(root, model_name)
    run_id = f"smoke-{model_name.lower()}"
    result = dict(services.train(model_name=model_name, run_id=run_id))
    state_store = RunStateStore(root)
    persisted = state_store.load(run_id, model_name)
    resumed = services.resume(model_name=model_name, run_id=run_id)
    replay = PersistentReplayStore(
        root / model_name.lower() / run_id / "replay", services.compatibility, capacity=8, seed=17
    )
    registry = services._load_registry(run_id)
    return {
        "candidate_read_only_loaded": result["read_only_checkpoint_loads"] == 1,
        "participant_count": len(registry.participants),
        "rating_events": result["rating_events"],
        "rating_status": result["rating_status"],
        "replay_reloaded": len(replay.load_buffer()) > 0,
        "state_reloaded": persisted == state_store.load(run_id, model_name) and resumed["stage"] == RunStage.COMPLETE.value,
        "training_step": result["training_step"],
        "worker_games": len(persisted.completed_game_ids),
    }


def run_smoke(runtime_dir: str | Path | None = None) -> dict[str, object]:
    """Run tiny authoritative Soo and Min iterations in a temporary runtime."""
    if runtime_dir is None:
        with tempfile.TemporaryDirectory(prefix="alphadiamond-m2-") as temporary:
            return run_smoke(Path(temporary))
    root = Path(runtime_dir)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "models": {
            SOO_MODEL_NAME: _run_model(root, SOO_MODEL_NAME),
            MIN_MODEL_NAME: _run_model(root, MIN_MODEL_NAME),
        },
        "status": "ok",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m diamond.alphazero.milestone2_smoke")
    parser.add_argument("--runtime-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_smoke(args.runtime_dir), sort_keys=True, separators=(",", ":")))
    except ValueError as error:
        print(json.dumps({"error": str(error), "status": "error"}, sort_keys=True))
        return 3
    except Exception as error:
        print(json.dumps({"error": f"{type(error).__name__}: {error}", "status": "error"}, sort_keys=True))
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
