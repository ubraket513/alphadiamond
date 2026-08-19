from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from diamond.alphazero import milestone2_smoke
from diamond.alphazero.milestone2_smoke import TinyTrainingServices, run_smoke
from diamond.alphazero.orchestration.run_state import RunStage, RunStateStore
from diamond.alphazero.rating.registry import RatingRegistry


class _InterruptedAfterCandidate(RuntimeError):
    pass


class _InterruptAfterCandidateStore(RunStateStore):
    def transition(self, state, next_stage, *, completion_marker, **changes):
        committed = super().transition(
            state,
            next_stage,
            completion_marker=completion_marker,
            **changes,
        )
        if next_stage is RunStage.PROMOTION_ARENA:
            raise _InterruptedAfterCandidate("candidate persisted")
        return committed


def test_tiny_production_smoke_runs_soo_and_min_without_fabricating_min_rating(tmp_path) -> None:
    report = run_smoke(tmp_path)

    assert report["status"] == "ok"
    assert set(report["models"]) == {"Soo", "Min"}
    assert report["models"]["Soo"]["worker_games"] == 2
    assert report["models"]["Soo"]["training_step"] == 1
    assert report["models"]["Soo"]["rating_events"] == 4
    assert report["models"]["Soo"]["candidate_read_only_loaded"] is True
    assert report["models"]["Soo"]["state_reloaded"] is True
    assert report["models"]["Min"]["worker_games"] == 2
    assert report["models"]["Min"]["training_step"] == 1
    assert report["models"]["Min"]["rating_events"] == 0
    assert report["models"]["Min"]["candidate_read_only_loaded"] is True
    assert report["models"]["Min"]["rating_status"] == "insufficient_history"
    assert report["models"]["Min"]["participant_count"] == 2
    assert report["models"]["Min"]["state_reloaded"] is True


def test_fresh_min_resume_preserves_insufficient_history_summary(tmp_path: Path) -> None:
    run_id = "fresh-min-rating-resume"
    trained = TinyTrainingServices(tmp_path, "Min").train(
        model_name="Min", run_id=run_id
    )

    resumed = TinyTrainingServices(tmp_path, "Min").resume(
        model_name="Min", run_id=run_id
    )

    assert resumed["rating_status"] == "insufficient_history"
    assert resumed == trained


def test_fresh_services_resume_after_candidate_save_without_repeating_completed_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "fresh-resume"
    with monkeypatch.context() as patch:
        patch.setattr(milestone2_smoke, "RunStateStore", _InterruptAfterCandidateStore)
        interrupted = TinyTrainingServices(tmp_path, "Soo")
        with pytest.raises(_InterruptedAfterCandidate, match="candidate persisted"):
            interrupted.train(model_name="Soo", run_id=run_id)

    state_store = RunStateStore(tmp_path)
    unfinished = state_store.load(run_id, "Soo")
    assert unfinished.stage is RunStage.PROMOTION_ARENA
    assert unfinished.candidate_checkpoint is not None
    candidate = Path(unfinished.candidate_checkpoint)
    candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    candidate_mtime = candidate.stat().st_mtime_ns
    replay_manifest = Path(unfinished.replay_manifest or "")
    replay_before = replay_manifest.read_bytes()

    resumed = TinyTrainingServices(tmp_path, "Soo").resume(
        model_name="Soo", run_id=run_id
    )
    completed = state_store.load(run_id, "Soo")

    assert resumed["stage"] == RunStage.COMPLETE.value
    assert completed.stage is RunStage.COMPLETE
    assert completed.candidate_checkpoint == str(candidate)
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == candidate_hash
    assert candidate.stat().st_mtime_ns == candidate_mtime
    assert replay_manifest.read_bytes() == replay_before
    assert completed.completed_game_ids == unfinished.completed_game_ids

    reloaded = TinyTrainingServices(tmp_path, "Soo").resume(
        model_name="Soo", run_id=run_id
    )
    assert reloaded == resumed
    assert state_store.load(run_id, "Soo") == completed


def test_promotion_and_soo_ratings_come_from_exact_checkpoint_arena_results(
    tmp_path: Path,
) -> None:
    run_smoke(tmp_path)
    state = RunStateStore(tmp_path).load("smoke-soo", "Soo")
    promotion = state.promotion_records[-1]
    result = promotion["result"]
    candidate_path = Path(str(promotion["candidate_checkpoint"]))
    champion_path = tmp_path / "soo" / "smoke-soo" / "bootstrap.pt"
    candidate_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    champion_hash = hashlib.sha256(champion_path.read_bytes()).hexdigest()

    assert result["candidate_evaluator_sha256"] == candidate_hash
    assert result["champion_evaluator_sha256"] == champion_hash
    assert result["wins"] + result["losses"] > 0
    assert promotion["promoted"] is (
        result["win_rate"] >= 0.5 and result["wins"] + result["losses"] > 0
    )

    registry = RatingRegistry.load(
        tmp_path / "soo" / "smoke-soo" / "ratings" / "registry.json"
    )
    candidate_id = registry.participants[
        next(
            participant_id
            for participant_id, participant in registry.participants.items()
            if participant.checkpoint_sha256 == candidate_hash
        )
    ].participant_id
    champion_id = registry.participants[
        next(
            participant_id
            for participant_id, participant in registry.participants.items()
            if participant.checkpoint_sha256 == champion_hash
        )
    ].participant_id
    completed_events = tuple(event for event in registry.events if event.completed)

    assert len(completed_events) == result["wins"] + result["losses"]
    assert sum(event.winner_id == candidate_id for event in completed_events) == result["wins"]
    assert sum(event.winner_id == champion_id for event in completed_events) == result["losses"]
