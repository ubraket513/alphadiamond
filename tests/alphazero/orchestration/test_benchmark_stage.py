from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from diamond.alphazero.checkpoint import save_checkpoint
from diamond.alphazero.config import NetworkConfig, TrainingConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.inference.coordinator import InferenceConfig
from diamond.alphazero.inference.model_pool import InferenceModelPool
from diamond.alphazero.network import MinModel, SooModel
from diamond.alphazero.orchestration.benchmark import ProductionBenchmarkStage
from diamond.alphazero.orchestration.coordinator import CandidateArtifact
from diamond.alphazero.rating.openings import OpeningSuite
from diamond.alphazero.rating.participants import CheckpointParticipant
from diamond.alphazero.rating.protocol import BenchmarkProtocol, EloConfig, TrueSkillConfig
from diamond.alphazero.rating.registry import RatingRegistry
from diamond.alphazero.rating.schedule import schedule_soo_pair
from diamond.alphazero.trainer import AlphaZeroTrainer


def _compatibility(model_name: str) -> CheckpointCompatibilitySpec:
    factory = (
        CheckpointCompatibilitySpec.soo
        if model_name == "Soo"
        else CheckpointCompatibilitySpec.min
    )
    return factory(
        model_version="2.0.0",
        network_config=NetworkConfig(width=8, residual_blocks=1),
    )


def _checkpoint(
    path: Path,
    compatibility: CheckpointCompatibilitySpec,
    *,
    training_step: int,
) -> Path:
    model = (
        SooModel(compatibility.network_config, model_version="2.0.0")
        if compatibility.identity.model_name == "Soo"
        else MinModel(compatibility.network_config, model_version="2.0.0")
    )
    trainer = AlphaZeroTrainer(
        model,
        compatibility,
        TrainingConfig(batch_size=1, weight_decay=0.0),
    )
    trainer.training_step = training_step
    save_checkpoint(path, trainer)
    return path


def _protocol(
    compatibility: CheckpointCompatibilitySpec, suite: OpeningSuite
) -> BenchmarkProtocol:
    rating = EloConfig() if compatibility.identity.model_name == "Soo" else TrueSkillConfig()
    return BenchmarkProtocol(
        compatibility=compatibility,
        simulations=1,
        c_puct=1.5,
        dirichlet_epsilon=0.0,
        decision_temperature=0.0,
        max_game_moves=2,
        opening_suite_version=suite.version,
        opening_suite_hash=suite.suite_hash,
        rating_system_version=rating.rating_system_version,
        rating_parameters=asdict(rating),
    )


def _candidate(path: Path, compatibility: CheckpointCompatibilitySpec) -> CandidateArtifact:
    participant = CheckpointParticipant.from_checkpoint(path)
    return CandidateArtifact(
        operation_id="candidate-operation",
        path=path,
        checkpoint_sha256=participant.checkpoint_sha256,
        training_step=participant.training_step,
        compatibility_namespace="sha256:test-compatibility",
        participant=participant,
    )


def test_soo_benchmark_executes_the_complete_independent_match_schedule(
    tmp_path: Path,
) -> None:
    compatibility = _compatibility("Soo")
    suite = OpeningSuite.generate(
        player_count=2, seed=7, opening_count=1, max_depth=0
    )
    protocol = _protocol(compatibility, suite)
    champion_path = _checkpoint(tmp_path / "champion.pt", compatibility, training_step=0)
    candidate_path = _checkpoint(tmp_path / "candidate.pt", compatibility, training_step=1)
    registry = RatingRegistry(protocol)
    completed_matches: list[str] = []

    def complete(match, opening, evaluators):
        completed_matches.append(match.match_id)
        assert opening.opening_id == match.opening_id
        assert set(evaluators) == set(match.participant_ids)
        return match.participant_ids

    stage = ProductionBenchmarkStage(
        protocol=protocol,
        opening_suite=suite,
        checkpoint_paths=lambda: (champion_path, candidate_path),
        artifacts_root=tmp_path / "benchmark-artifacts",
        registry=registry,
        registry_path=tmp_path / "ratings" / "registry.json",
        model_pool=InferenceModelPool(device="cpu"),
        inference_config=InferenceConfig(2, 2, 8),
        match_runner=complete,
    )

    events = stage.execute(
        "independent-rating-operation",
        _candidate(candidate_path, compatibility),
        str(champion_path),
    )

    assert len(completed_matches) == 4
    assert len(set(completed_matches)) == 4
    assert len(events) == 4
    assert registry.events == events
    assert all(event.opening_id == suite.openings[0].opening_id for event in events)
    assert (tmp_path / "ratings" / "registry.json").exists()


def test_min_benchmark_reports_insufficient_history_without_an_event(
    tmp_path: Path,
) -> None:
    compatibility = _compatibility("Min")
    suite = OpeningSuite.generate(
        player_count=3, seed=7, opening_count=1, max_depth=0
    )
    protocol = _protocol(compatibility, suite)
    first = _checkpoint(tmp_path / "first.pt", compatibility, training_step=0)
    second = _checkpoint(tmp_path / "second.pt", compatibility, training_step=1)
    registry = RatingRegistry(protocol)
    stage = ProductionBenchmarkStage(
        protocol=protocol,
        opening_suite=suite,
        checkpoint_paths=lambda: (first, second),
        artifacts_root=tmp_path / "benchmark-artifacts",
        registry=registry,
        registry_path=tmp_path / "ratings" / "registry.json",
        model_pool=InferenceModelPool(device="cpu"),
        inference_config=InferenceConfig(2, 2, 8),
        match_runner=lambda *_args: pytest.fail("no Min match may run"),
    )

    events = stage.execute(
        "insufficient-history-operation",
        _candidate(second, compatibility),
        str(first),
    )

    assert events == ()
    assert stage.status == "insufficient_history"
    assert registry.events == ()


def test_min_benchmark_runs_36_matches_per_opening_for_three_distinct_artifacts(
    tmp_path: Path,
) -> None:
    compatibility = _compatibility("Min")
    suite = OpeningSuite.generate(
        player_count=3, seed=7, opening_count=1, max_depth=0
    )
    protocol = _protocol(compatibility, suite)
    paths = tuple(
        _checkpoint(tmp_path / f"history-{step}.pt", compatibility, training_step=step)
        for step in range(3)
    )
    registry = RatingRegistry(protocol)
    stage = ProductionBenchmarkStage(
        protocol=protocol,
        opening_suite=suite,
        checkpoint_paths=lambda: paths,
        artifacts_root=tmp_path / "benchmark-artifacts",
        registry=registry,
        registry_path=tmp_path / "ratings" / "registry.json",
        model_pool=InferenceModelPool(device="cpu"),
        inference_config=InferenceConfig(4, 2, 16),
        match_runner=lambda match, _opening, _evaluators: match.participant_ids,
    )

    events = stage.execute(
        "min-complete-operation",
        _candidate(paths[-1], compatibility),
        str(paths[0]),
    )

    assert len(events) == 36
    assert len({event.event_id for event in events}) == 36
    assert all(len(set(event.participant_ids)) == 3 for event in events)
    assert registry.events == events


class _FirstLegalSearch:
    """Records who was asked to move, and plays the lowest legal action."""

    seats: list[int] = []

    def __init__(self, game, evaluator, config) -> None:
        self._game = game
        self._evaluator = evaluator

    def run(self, state, *, temperature: float):
        _FirstLegalSearch.seats.append(self._game.current_player_id(state))
        action = min(self._game.legal_action_ids(state))
        return SimpleNamespace(selected_action=action)


def test_the_default_match_runner_plays_through_the_injected_search(
    tmp_path: Path,
) -> None:
    """`_run_match` no longer names an engine: it asks the search factory.

    The seats it reports are the scheduled turn order, and the ranking it
    returns is in participant ids rather than seat ids -- which is the part a
    swapped engine could get wrong without any test noticing.
    """
    compatibility = _compatibility("Soo")
    suite = OpeningSuite.generate(player_count=2, seed=7, opening_count=1, max_depth=0)
    protocol = _protocol(compatibility, suite)
    registry = RatingRegistry(protocol)
    stage = ProductionBenchmarkStage(
        protocol=protocol,
        opening_suite=suite,
        checkpoint_paths=tuple,
        artifacts_root=tmp_path / "benchmark-artifacts",
        registry=registry,
        registry_path=tmp_path / "ratings" / "registry.json",
        model_pool=InferenceModelPool(device="cpu"),
        inference_config=InferenceConfig(2, 2, 8),
        search_factory=_FirstLegalSearch,
    )
    match = schedule_soo_pair(
        opening_suite=suite, participant_ids=("candidate", "champion")
    )[0]
    _FirstLegalSearch.seats = []

    ranking = stage._run_match(
        match, suite.openings[0], {"candidate": object(), "champion": object()}
    )

    # max_game_moves is 2, so the game is unfinished and no ranking is claimed.
    assert ranking is None
    assert _FirstLegalSearch.seats == list(match.turn_order)
