"""Rolling one bad iteration out of a replay store, and the two ways it fails.

A training iteration was run with the wrong search budget and had to be undone.
The obvious repair -- drop the bad entries from `manifest.chunks` -- loads
correctly and then kills the *next* run, twice over, for two different reasons.
Both are reproduced here so the repair tool cannot regress into either.

The shape that matters is a `game_id` that appears with one outcome before the
rollback and the opposite outcome after it.  That is not exotic: the id is
derived from the run and the iteration, so replaying an iteration regenerates
the same ids, and a stronger actor finishes games the weaker one abandoned.
"""

from __future__ import annotations

import json
import subprocess
import sys
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

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools" / "replay_transaction.py"


def _spec() -> CheckpointCompatibilitySpec:
    return CheckpointCompatibilitySpec.soo(
        model_version="1.2.3", network_config=NetworkConfig(width=16, residual_blocks=1)
    )


def _episode(
    spec: CheckpointCompatibilitySpec,
    game_id: str,
    *,
    markers: tuple[int, ...] = (1,),
    completed: bool = True,
) -> EpisodeResult:
    samples = tuple(
        TrainingSample(
            compatibility=spec,
            node_features=((float(marker),) * 4,) * 73,
            canonical_player_ids=(1, 2),
            sparse_policy=((marker, 0.75), (marker + 1, 0.25)),
            value_target=(1.0,),
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
        final_order=(1, 2) if completed else None,
        move_count=len(markers),
        completed=completed,
        aborted_reason=None if completed else "max_game_moves_exceeded",
    )


def _store(root: Path, spec: CheckpointCompatibilitySpec) -> PersistentReplayStore:
    return PersistentReplayStore(root, spec, capacity=1000, seed=3)


def _namespace(root: Path) -> Path:
    return next(p.parent for p in root.rglob("manifest.json") if p.parent.name != "snapshots")


def _run_tool(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *arguments], capture_output=True, text=True, check=False
    )


def test_dropping_manifest_entries_leaves_a_chunk_that_blocks_reingestion(tmp_path):
    """The first failure: an orphan file conflicts with the regenerated game."""
    spec = _spec()
    root = tmp_path / "replay"
    store = _store(root, spec)
    store.ingest_episodes((_episode(spec, "game-" + "1" * 64, markers=(1, 2)),))

    # Hand-rollback of the kind that looks right: forget the entry, keep the file.
    namespace = _namespace(root)
    manifest = json.loads((namespace / "manifest.json").read_text())
    manifest["chunks"] = []
    manifest["game_ids"] = []
    (namespace / "manifest.json").write_text(json.dumps(manifest))

    replayed = _store(root, spec)
    assert len(replayed.load_buffer()) == 0, "the rollback should have emptied the buffer"
    # The manifest no longer knows the id, so this gets past the duplicate check
    # and dies deeper, in the chunk writer -- a different message for the same
    # cause, and the one a hand-rollback actually hits first.
    with pytest.raises(ReplayStoreError, match="conflicting immutable replay chunk"):
        # Same id, different content -- the actor changed, which is the point.
        replayed.ingest_episodes((_episode(spec, "game-" + "1" * 64, markers=(7, 8, 9)),))


def test_dropping_only_chunks_leaves_aborted_entries_that_block_completion(tmp_path):
    """The second failure: `aborted` is checked too, and is not bookkeeping."""
    spec = _spec()
    root = tmp_path / "replay"
    store = _store(root, spec)
    store.ingest_episodes((_episode(spec, "game-" + "2" * 64, completed=False),))

    namespace = _namespace(root)
    manifest = json.loads((namespace / "manifest.json").read_text())
    manifest["chunks"] = []
    manifest["game_ids"] = []
    # `aborted` deliberately left alone -- it holds no samples, so it reads as
    # a counter.  It is not: ingestion rejects a completed game whose id is here.
    (namespace / "manifest.json").write_text(json.dumps(manifest))

    replayed = _store(root, spec)
    with pytest.raises(ReplayStoreError, match="conflicting duplicate game_id"):
        replayed.ingest_episodes((_episode(spec, "game-" + "2" * 64, markers=(3, 4)),))


def test_snapshot_and_restore_survives_abort_then_completion(tmp_path):
    """The repair: the same sequence, rolled back as one transaction."""
    spec = _spec()
    root = tmp_path / "replay"
    store = _store(root, spec)
    store.ingest_episodes((_episode(spec, "game-" + "3" * 64, markers=(1, 2)),))

    assert _run_tool("snapshot", str(root), "--tag", "before").returncode == 0

    # The bad iteration: one new game completes, one aborts.
    store.ingest_episodes(
        (
            _episode(spec, "game-" + "4" * 64, markers=(5, 6)),
            _episode(spec, "game-" + "5" * 64, completed=False),
        )
    )

    restored = _run_tool("restore", str(root), "--tag", "before", "--delete")
    assert restored.returncode == 0, restored.stdout + restored.stderr
    assert _run_tool("verify", str(root)).returncode == 0

    replayed = _store(root, spec)
    assert len(replayed.load_buffer()) == 2, "only the pre-snapshot game should remain"
    # Both ids are free again, including the one that aborted and now completes.
    replayed.ingest_episodes(
        (
            _episode(spec, "game-" + "4" * 64, markers=(9,)),
            _episode(spec, "game-" + "5" * 64, markers=(9, 9, 9)),
        )
    )
    assert len(_store(root, spec).load_buffer()) == 6


def test_verify_reports_an_orphan_rather_than_waiting_for_the_next_ingest(tmp_path):
    spec = _spec()
    root = tmp_path / "replay"
    store = _store(root, spec)
    store.ingest_episodes((_episode(spec, "game-" + "6" * 64),))

    assert _run_tool("verify", str(root)).returncode == 0

    namespace = _namespace(root)
    manifest = json.loads((namespace / "manifest.json").read_text())
    manifest["chunks"] = []
    manifest["game_ids"] = []
    (namespace / "manifest.json").write_text(json.dumps(manifest))

    result = _run_tool("verify", str(root))
    assert result.returncode == 1
    assert "ORPHAN" in result.stdout
