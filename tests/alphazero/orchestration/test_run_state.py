from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.inference.protocol import ModelKey
from diamond.alphazero.orchestration.run_state import (
    RunStage,
    RunStateError,
    RunStateStore,
)


def _compatibility(kind: str) -> CheckpointCompatibilitySpec:
    factory = (
        CheckpointCompatibilitySpec.soo
        if kind == "soo"
        else CheckpointCompatibilitySpec.min
    )
    return factory(model_version="2.0.0", network_config=NetworkConfig(width=8, residual_blocks=1))


def _initialize(tmp_path, *, kind: str = "soo"):
    return RunStateStore(tmp_path).initialize(
        run_id="run-2026-08-19",
        compatibility=_compatibility(kind),
        run_seed=7401,
        protocol_ids={"promotion": "promotion-v1", "rating": f"{kind}-rating-v1"},
        champion_checkpoint="checkpoints/champion.json",
    )


def test_initialize_freezes_identity_and_json_state(tmp_path) -> None:
    state = _initialize(tmp_path)

    assert state.stage is RunStage.INITIALIZE
    assert state.generation == 0
    assert state.model_identity == {
        "model_name": "Soo",
        "model_version": "2.0.0",
        "player_count": 2,
        "value_semantics_version": "current-player-scalar-winloss-v1",
    }
    assert state.champion_checkpoint == "checkpoints/champion.json"
    assert state.compatibility_namespace.startswith("sha256:")
    assert state.candidate_checkpoint is None
    assert state.iteration == 0
    assert state.training_step == 0
    assert state.completed_game_ids == ()
    assert state.stage_completions == {}
    with pytest.raises(FrozenInstanceError):
        state.run_id = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        state.protocol_ids["rating"] = "changed"  # type: ignore[index]


def test_champion_checkpoint_model_key_round_trips_as_durable_identity(tmp_path) -> None:
    store = RunStateStore(tmp_path)
    champion_key = ModelKey("Soo", "2.0.0", "a" * 64)

    state = store.initialize(
        run_id="champion-bound",
        compatibility=_compatibility("soo"),
        run_seed=7401,
        protocol_ids={"promotion": "promotion-v1", "rating": "soo-rating-v1"},
        champion_checkpoint="checkpoints/champion.pt",
        champion_model_key=champion_key,
    )

    assert state.champion_model_key == champion_key
    assert store.load(state.run_id, "Soo").champion_model_key == champion_key


def test_same_run_id_has_independent_soo_and_min_state(tmp_path) -> None:
    store = RunStateStore(tmp_path)
    soo = _initialize(tmp_path, kind="soo")
    min_state = _initialize(tmp_path, kind="min")

    assert store.load(soo.run_id, "Soo") == soo
    assert store.load(min_state.run_id, "Min") == min_state
    assert store.state_path(soo.run_id, "Soo") != store.state_path(min_state.run_id, "Min")


def test_transitions_follow_the_exact_order_and_mark_completed_stages(tmp_path) -> None:
    store = RunStateStore(tmp_path)
    state = _initialize(tmp_path)
    order = (
        RunStage.INITIALIZE,
        RunStage.SELF_PLAY,
        RunStage.REPLAY_INGEST,
        RunStage.TRAIN,
        RunStage.SAVE_CANDIDATE,
        RunStage.PROMOTION_ARENA,
        RunStage.RATING_BENCHMARK,
        RunStage.PROMOTE_OR_REJECT,
        RunStage.PERSIST,
        RunStage.COMPLETE,
    )

    for current, next_stage in zip(order[:-1], order[1:], strict=True):
        assert state.stage is current
        state = store.transition(
            state,
            next_stage,
            completion_marker=f"iteration-0:{current.value}",
        )

    assert state.stage is RunStage.COMPLETE
    assert state.generation == 9
    assert state.stage_completions == {
        stage.value: f"iteration-0:{stage.value}" for stage in order[:-1]
    }
    with pytest.raises(RunStateError, match="COMPLETE is terminal"):
        store.transition(state, RunStage.INITIALIZE, completion_marker="restart")


def test_complete_run_atomically_starts_the_incremented_iteration(tmp_path) -> None:
    store = RunStateStore(tmp_path)
    state = _initialize(tmp_path)
    state = store.transition(state, RunStage.SELF_PLAY, completion_marker="initialize")
    state = store.save(
        replace(
            state,
            champion_checkpoint="checkpoints/champion-1.json",
            candidate_checkpoint="checkpoints/candidate-1.json",
            training_step=23,
            replay_manifest="replay/manifest.json",
            completed_game_ids=("game-1",),
            promotion_records=({"iteration": 0, "promoted": True},),
            rating_records=({"iteration": 0, "event_ids": ["event-1"]},),
        )
    )
    remaining = tuple(RunStage)[2:]
    for next_stage in remaining:
        changes = {"iteration": 1} if next_stage is RunStage.COMPLETE else {}
        state = store.transition(
            state,
            next_stage,
            completion_marker=f"complete-{state.stage.value}",
            **changes,
        )

    next_iteration = store.start_next_iteration(state)

    assert next_iteration.stage is RunStage.INITIALIZE
    assert next_iteration.iteration == 1
    assert next_iteration.generation == state.generation + 1
    assert next_iteration.model_identity == state.model_identity
    assert next_iteration.compatibility == state.compatibility
    assert next_iteration.protocol_ids == state.protocol_ids
    assert next_iteration.champion_checkpoint == "checkpoints/champion-1.json"
    assert next_iteration.training_step == 23
    assert next_iteration.replay_manifest == "replay/manifest.json"
    assert next_iteration.promotion_records == state.promotion_records
    assert next_iteration.rating_records == state.rating_records
    assert next_iteration.candidate_checkpoint is None
    assert next_iteration.completed_game_ids == ()
    assert next_iteration.stage_completions == {}
    assert store.load(state.run_id, "Soo") == next_iteration


def test_transition_rejects_skips_and_save_rejects_identity_changes(tmp_path) -> None:
    store = RunStateStore(tmp_path)
    state = _initialize(tmp_path)

    with pytest.raises(RunStateError, match="INITIALIZE -> SELF_PLAY"):
        store.transition(state, RunStage.TRAIN, completion_marker="skip")

    for altered in (
        replace(state, run_id="different-run"),
        replace(state, run_seed=99),
        replace(state, protocol_ids={"rating": "changed"}),
    ):
        with pytest.raises(RunStateError, match="immutable"):
            store.save(altered)
    with pytest.raises(RunStateError, match="model_identity does not match compatibility"):
        replace(state, compatibility={"model_name": "Soo"})


def test_progress_fields_round_trip_as_frozen_json(tmp_path) -> None:
    store = RunStateStore(tmp_path)
    state = _initialize(tmp_path)
    state = store.transition(state, RunStage.SELF_PLAY, completion_marker="initialized")
    state = replace(
        state,
        candidate_checkpoint="checkpoints/candidate-17.json",
        iteration=3,
        training_step=417,
        replay_manifest="replay/manifest.json",
        completed_game_ids=("game-a", "game-b"),
        promotion_records=({"candidate": "candidate-17", "promoted": True},),
        rating_records=({"event_id": "rating-9", "protocol_id": "soo-rating-v1"},),
    )

    saved = store.save(state)
    loaded = RunStateStore(tmp_path).load(state.run_id, "Soo")

    assert loaded == saved
    assert loaded.generation == 2
    assert loaded.candidate_checkpoint == "checkpoints/candidate-17.json"
    assert loaded.completed_game_ids == ("game-a", "game-b")
    assert loaded.promotion_records[0]["promoted"] is True
    with pytest.raises(TypeError):
        loaded.promotion_records[0]["promoted"] = False  # type: ignore[index]


def test_seed_derivation_is_stable_and_identity_scoped(tmp_path) -> None:
    soo = _initialize(tmp_path, kind="soo")
    min_state = _initialize(tmp_path, kind="min")

    seed = soo.derive_seed("self-play", 3, "game-a")

    assert seed == soo.derive_seed("self-play", 3, "game-a")
    assert 0 <= seed < 2**63
    assert seed != soo.derive_seed("self-play", 3, "game-b")
    assert seed != min_state.derive_seed("self-play", 3, "game-a")


def test_save_rejects_a_stale_generation_without_overwriting_newer_state(tmp_path) -> None:
    store = RunStateStore(tmp_path)
    stale = _initialize(tmp_path)
    current = store.save(replace(stale, training_step=1))

    with pytest.raises(RunStateError, match="stale training run generation"):
        store.save(replace(stale, training_step=999))

    assert store.load(stale.run_id, "Soo") == current


def test_failed_atomic_replace_preserves_authoritative_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStateStore(tmp_path)
    state = _initialize(tmp_path)
    path = store.state_path(state.run_id, "Soo")
    authoritative = path.read_bytes()

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(RunStateError, match="atomic write failed"):
        store.save(replace(state, candidate_checkpoint="checkpoints/candidate.json"))

    assert path.read_bytes() == authoritative
    assert list(path.parent.glob("*.tmp")) == []


def test_truncated_unreferenced_temporary_file_is_ignored(tmp_path) -> None:
    store = RunStateStore(tmp_path)
    state = _initialize(tmp_path)
    path = store.state_path(state.run_id, "Soo")
    path.with_name(f"{path.name}.crash.tmp").write_text('{"generation":', encoding="utf-8")

    assert RunStateStore(tmp_path).load(state.run_id, "Soo") == state


def test_corrupt_authoritative_state_fails_clearly(tmp_path) -> None:
    store = RunStateStore(tmp_path)
    state = _initialize(tmp_path)
    path = store.state_path(state.run_id, "Soo")
    path.write_text('{"schema_version":1', encoding="utf-8")

    with pytest.raises(RunStateError, match="corrupt training run state"):
        RunStateStore(tmp_path).load(state.run_id, "Soo")


def test_unknown_schema_version_is_rejected(tmp_path) -> None:
    store = RunStateStore(tmp_path)
    state = _initialize(tmp_path)
    path = store.state_path(state.run_id, "Soo")
    payload = state.to_payload()
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RunStateError, match="unsupported training run schema version"):
        RunStateStore(tmp_path).load(state.run_id, "Soo")
