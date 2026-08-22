from __future__ import annotations

import json
from pathlib import Path

import pytest

from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.inference.protocol import ModelKey
from diamond.alphazero.orchestration.replay_store import (
    PersistentReplayStore,
    ReplayStoreError,
)
from diamond.alphazero.orchestration.selfplay_workers import EpisodeResult
from diamond.alphazero.replay import TrainingSample


def _spec(kind: str = "soo") -> CheckpointCompatibilitySpec:
    factory = CheckpointCompatibilitySpec.soo if kind == "soo" else CheckpointCompatibilitySpec.min
    return factory(model_version="1.2.3", network_config=NetworkConfig(width=16, residual_blocks=1))


def _episode(
    spec: CheckpointCompatibilitySpec,
    game_id: str,
    *,
    markers: tuple[int, ...] = (1,),
    completed: bool = True,
) -> EpisodeResult:
    count = spec.identity.player_count
    samples = tuple(
        TrainingSample(
            compatibility=spec,
            node_features=((float(marker),) * (count * 2),) * 73,
            canonical_player_ids=tuple(range(1, count + 1)),
            sparse_policy=((marker, 0.75), (marker + 1, 0.25)),
            value_target=(1.0,) if count == 2 else (1.0, 0.0, -1.0),
        )
        for marker in markers
    )
    return EpisodeResult(
        game_id=game_id,
        seed=17,
        retry_id="attempt-0",
        model_key=ModelKey(spec.identity.model_name, "1.2.3", "a" * 64),
        compatibility=spec,
        samples=samples if completed else (),
        final_order=tuple(range(1, count + 1)) if completed else None,
        move_count=len(markers),
        completed=completed,
        aborted_reason=None if completed else "max_game_moves_exceeded",
    )


def test_completed_episode_is_immutable_json_chunk_and_manifest_is_authoritative(
    tmp_path: Path,
) -> None:
    store = PersistentReplayStore(tmp_path, _spec(), capacity=8, seed=3)

    assert store.ingest_episode(_episode(_spec(), "game-a", markers=(4, 8)))

    manifest = store.manifest
    assert manifest.game_ids == ("game-a",)
    assert len(store.load_buffer()) == 2
    chunk_path = store.chunk_path("game-a")
    payload = json.loads(chunk_path.read_text(encoding="utf-8"))
    assert payload["samples"][0]["sparse_policy"] == [[4, 0.75], [5, 0.25]]
    assert payload["sha256"] == store.manifest.chunks[0].sha256
    assert not list(store.namespace_path.glob("*.tmp"))


def test_completed_duplicate_with_same_content_is_a_noop(tmp_path: Path) -> None:
    store = PersistentReplayStore(tmp_path, _spec(), capacity=8)
    episode = _episode(_spec(), "game-a")

    assert store.ingest_episode(episode)
    assert not store.ingest_episode(episode)
    assert store.manifest.game_ids == ("game-a",)


def test_manifest_replace_failure_leaves_chunk_an_uningested_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = PersistentReplayStore(tmp_path, _spec(), capacity=8)
    original_replace = Path.replace

    def fail_manifest_replace(path: Path, target: str | Path) -> Path:
        if Path(target) == store.manifest_path:
            raise OSError("simulated manifest replacement interruption")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_manifest_replace)

    with pytest.raises(ReplayStoreError, match="atomic write"):
        store.ingest_episode(_episode(_spec(), "game-a"))

    assert store.manifest.game_ids == ()
    assert PersistentReplayStore(tmp_path, _spec(), capacity=8).manifest.game_ids == ()
    assert store.chunk_path("game-a").exists()


def test_sampling_manifest_failure_does_not_advance_the_live_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec()
    store = PersistentReplayStore(tmp_path / "store", spec, capacity=16, seed=7)
    baseline = PersistentReplayStore(tmp_path / "baseline", spec, capacity=16, seed=7)
    for marker in range(10):
        episode = _episode(spec, f"game-{marker}", markers=(marker,))
        store.ingest_episode(episode)
        baseline.ingest_episode(episode)
    expected = baseline.sample(3)
    original_replace = Path.replace

    def fail_manifest_replace(path: Path, target: str | Path) -> Path:
        if Path(target) == store.manifest_path:
            raise OSError("simulated manifest replacement interruption")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_manifest_replace)
    with pytest.raises(ReplayStoreError, match="atomic write"):
        store.sample(3)
    monkeypatch.undo()

    assert store.sample(3) == expected


def test_conflicting_duplicate_game_id_is_rejected(tmp_path: Path) -> None:
    store = PersistentReplayStore(tmp_path, _spec(), capacity=8)
    assert store.ingest_episode(_episode(_spec(), "game-a", markers=(1,)))

    with pytest.raises(ReplayStoreError, match="conflicting duplicate"):
        store.ingest_episode(_episode(_spec(), "game-a", markers=(2,)))


def test_aborted_episode_records_metrics_without_creating_samples(tmp_path: Path) -> None:
    store = PersistentReplayStore(tmp_path, _spec(), capacity=8)

    assert store.ingest_episode(_episode(_spec(), "game-aborted", completed=False))

    assert store.manifest.game_ids == ()
    assert store.manifest.aborted[0].game_id == "game-aborted"
    assert store.manifest.aborted[0].move_count == 1
    assert len(store.load_buffer()) == 0
    assert not list(store.chunks_path.glob("*.json"))


def test_soo_min_and_compatibility_are_strictly_isolated(tmp_path: Path) -> None:
    soo = _spec()
    min_spec = _spec("min")
    soo_store = PersistentReplayStore(tmp_path, soo, capacity=8)
    min_store = PersistentReplayStore(tmp_path, min_spec, capacity=8)

    assert soo_store.namespace_path != min_store.namespace_path
    with pytest.raises(ReplayStoreError, match="compatibility"):
        soo_store.ingest_episode(_episode(min_spec, "game-min"))


def test_bounded_reload_and_sampling_restart_are_deterministic(tmp_path: Path) -> None:
    spec = _spec()
    store = PersistentReplayStore(tmp_path, spec, capacity=3, seed=9)
    for marker in range(5):
        store.ingest_episode(_episode(spec, f"game-{marker}", markers=(marker,)))

    assert [row.sparse_policy[0][0] for row in store.load_buffer().samples] == [2, 3, 4]
    first = store.sample(2)
    restarted = PersistentReplayStore(tmp_path, spec, capacity=3, seed=9)
    assert restarted.sample(2) == store.sample(2)
    assert first != ()


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_missing_or_corrupt_referenced_chunk_fails_clearly(tmp_path: Path, damage: str) -> None:
    store = PersistentReplayStore(tmp_path, _spec(), capacity=8)
    store.ingest_episode(_episode(_spec(), "game-a"))
    chunk = store.chunk_path("game-a")
    if damage == "missing":
        chunk.unlink()
    else:
        chunk.write_text("not json", encoding="utf-8")

    with pytest.raises(ReplayStoreError, match=f"{damage}|chunk"):
        PersistentReplayStore(tmp_path, _spec(), capacity=8).load_buffer()


def test_orphan_chunk_before_manifest_is_ignored_and_recoverable(tmp_path: Path) -> None:
    store = PersistentReplayStore(tmp_path, _spec(), capacity=8)
    store.chunks_path.mkdir(parents=True)
    (store.chunks_path / "orphan.json").write_text('{"not":"ingested"}', encoding="utf-8")

    restarted = PersistentReplayStore(tmp_path, _spec(), capacity=8)
    assert restarted.manifest.game_ids == ()
    assert len(restarted.load_buffer()) == 0
    assert (restarted.chunks_path / "orphan.json").exists()


def test_bulk_ingest_writes_the_manifest_once(tmp_path: Path, monkeypatch) -> None:
    """The reason this method exists.

    ``ingest_episode`` rewrites the whole manifest per call.  At 768 games an
    iteration against a manifest that had grown to 1.4 MB, that is about a
    gigabyte of manifest writes to record what one write records.
    """
    spec = _spec()
    store = PersistentReplayStore(tmp_path, spec, capacity=100, seed=3)
    writes = {"n": 0}
    original = store._write_manifest

    def counted(manifest):
        writes["n"] += 1
        return original(manifest)

    monkeypatch.setattr(store, "_write_manifest", counted)
    ingested = store.ingest_episodes(
        _episode(spec, f"game-{index}", markers=(index + 1,)) for index in range(8)
    )
    assert ingested == 8
    assert writes["n"] == 1
    assert len(store.manifest.chunks) == 8


def test_bulk_ingest_matches_one_by_one_ingest(tmp_path: Path) -> None:
    """Same semantics, including duplicates and the resulting buffer."""
    spec = _spec()
    episodes = [_episode(spec, f"game-{index}", markers=(index + 1,)) for index in range(5)]

    single = PersistentReplayStore(tmp_path / "a", spec, capacity=100, seed=3)
    for episode in episodes:
        single.ingest_episode(episode)
    bulk = PersistentReplayStore(tmp_path / "b", spec, capacity=100, seed=3)
    bulk.ingest_episodes(episodes)

    assert single.manifest.chunks == bulk.manifest.chunks
    assert single.manifest.game_ids == bulk.manifest.game_ids
    assert len(single.load_buffer()) == len(bulk.load_buffer())
    # A repeat is a no-op on both paths.
    assert bulk.ingest_episodes(episodes) == 0


def test_bulk_ingest_rejects_a_conflicting_duplicate(tmp_path: Path) -> None:
    spec = _spec()
    store = PersistentReplayStore(tmp_path, spec, capacity=100, seed=3)
    store.ingest_episode(_episode(spec, "game-0", markers=(1,)))
    with pytest.raises(ReplayStoreError, match="conflicting duplicate"):
        store.ingest_episodes([_episode(spec, "game-0", markers=(9,))])


def test_pruning_drops_only_chunks_the_buffer_cannot_reach(tmp_path: Path) -> None:
    """The safety property: pruning must not change what ``load_buffer`` returns.

    ``load_buffer`` extends a bounded buffer in manifest order, so only the
    newest ``capacity`` samples survive.  Everything older is read and evicted
    on every iteration -- work and disk spent on data that cannot be sampled.
    """
    spec = _spec()
    store = PersistentReplayStore(tmp_path, spec, capacity=10, seed=3)
    # 12 episodes of 2 samples each: 24 samples against a capacity of 10.
    store.ingest_episodes(
        _episode(spec, f"game-{index}", markers=(index + 1, index + 2))
        for index in range(12)
    )
    before = [tuple(s.sparse_policy) for s in store.load_buffer().samples]
    assert len(store.manifest.chunks) == 12

    removed = store.prune_to_capacity()
    assert removed > 0
    assert len(store.manifest.chunks) == 12 - removed
    after = [tuple(s.sparse_policy) for s in store.load_buffer().samples]
    assert after == before, "pruning changed the sampled buffer"

    # The dropped files are actually gone, which is the point.
    assert len(list((tmp_path).rglob("chunks/*.json"))) == 12 - removed


def test_pruning_survives_a_reopen(tmp_path: Path) -> None:
    """A pruned store must still load: the manifest may not reference a file
    that was deleted."""
    spec = _spec()
    store = PersistentReplayStore(tmp_path, spec, capacity=10, seed=3)
    store.ingest_episodes(
        _episode(spec, f"game-{index}", markers=(index + 1, index + 2))
        for index in range(12)
    )
    expected = [tuple(s.sparse_policy) for s in store.load_buffer().samples]
    store.prune_to_capacity()

    reopened = PersistentReplayStore(tmp_path, spec, capacity=10, seed=3)
    assert [tuple(s.sparse_policy) for s in reopened.load_buffer().samples] == expected


def test_pruning_is_a_noop_below_capacity(tmp_path: Path) -> None:
    spec = _spec()
    store = PersistentReplayStore(tmp_path, spec, capacity=1000, seed=3)
    store.ingest_episodes(
        _episode(spec, f"game-{index}", markers=(index + 1,)) for index in range(4)
    )
    assert store.prune_to_capacity() == 0
    assert len(store.manifest.chunks) == 4
