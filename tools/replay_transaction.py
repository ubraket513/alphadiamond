"""Snapshot and roll back a persistent replay store as one unit.

A `PersistentReplayStore` keeps its authoritative state in `manifest.json` and
writes chunk files beside it.  That split is deliberate -- a crash between the
two leaves an orphan chunk that no manifest references, and the store is
designed to tolerate exactly that.  It is *not* designed for a hand-edited
manifest, and rolling one back by hand fails in two ways that only surface on
the next write:

**Orphan chunks collide.**  `ingest_episodes` hashes each completed episode and
compares it with any chunk already on disk for that `game_id`.  Removing an
entry from `manifest.chunks` but leaving its file means the next run regenerates
the same `game_id`, produces different content, and dies with `conflicting
duplicate game_id`.

**`aborted` is not bookkeeping.**  The same check rejects a `game_id` that
appears in `manifest.aborted`, so an iteration whose games aborted and were
rolled back still blocks those ids from ever completing.  Dropping only
`chunks` and `game_ids` looks correct, loads correctly, and fails on the next
ingest.

Both were paid for, in that order, rolling back one bad training iteration.

So a rollback is a transaction over the *whole* manifest -- `chunks`,
`game_ids`, `aborted` and `rng_state` together -- followed by disposing of every
chunk the restored manifest does not reference.  Adding a retry id to `game_id`
would sidestep the collision and is deliberately not done: the id is what makes
ingestion idempotent, and an id that changes per attempt cannot.

Usage::

    python tools/replay_transaction.py snapshot RUN/replay --tag before-i0150
    python tools/replay_transaction.py verify   RUN/replay
    python tools/replay_transaction.py restore  RUN/replay --tag before-i0150
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SNAPSHOT_DIR = "snapshots"


def _namespaces(root: Path) -> list[Path]:
    """Every `<model>/<compatibility-digest>` directory holding a manifest."""
    found = sorted(p.parent for p in root.rglob("manifest.json") if p.parent.name != SNAPSHOT_DIR)
    if not found:
        raise SystemExit(f"no replay namespace with a manifest under {root}")
    return found


def _chunk_name(game_id: str) -> str:
    return hashlib.sha256(game_id.encode("utf-8")).hexdigest() + ".json"


def _referenced(manifest: dict) -> set[str]:
    return {_chunk_name(entry["game_id"]) for entry in manifest["chunks"]}


def _on_disk(namespace: Path) -> set[str]:
    chunks = namespace / "chunks"
    return {p.name for p in chunks.iterdir() if p.suffix == ".json"} if chunks.is_dir() else set()


def _describe(manifest: dict) -> str:
    return (
        f"{len(manifest['chunks'])} chunks, "
        f"{len(manifest['game_ids'])} game_ids, "
        f"{len(manifest['aborted'])} aborted, "
        f"{sum(c['sample_count'] for c in manifest['chunks']):,} samples"
    )


def snapshot(namespace: Path, tag: str) -> None:
    target = namespace / SNAPSHOT_DIR
    target.mkdir(exist_ok=True)
    path = target / f"{tag}.json"
    if path.exists():
        raise SystemExit(f"snapshot {tag} already exists at {path}")
    manifest = json.loads((namespace / "manifest.json").read_text())
    path.write_text(json.dumps(manifest))
    print(f"  [snapshot] {tag}: {_describe(manifest)}")


def verify(namespace: Path) -> int:
    manifest = json.loads((namespace / "manifest.json").read_text())
    referenced, disk = _referenced(manifest), _on_disk(namespace)
    missing, orphans = referenced - disk, disk - referenced
    aborted_ids = {entry["game_id"] for entry in manifest["aborted"]}
    chunk_ids = {entry["game_id"] for entry in manifest["chunks"]}
    overlap = aborted_ids & chunk_ids
    print(f"  [verify] {_describe(manifest)}")
    problems = 0
    if missing:
        print(f"    MISSING {len(missing)} referenced chunk files -- the store cannot load")
        problems += 1
    if orphans:
        print(
            f"    ORPHAN  {len(orphans)} chunk files no manifest entry references; "
            f"the next ingest of those game_ids will conflict"
        )
        problems += 1
    if overlap:
        print(f"    OVERLAP {len(overlap)} game_ids are both completed and aborted")
        problems += 1
    if set(manifest["game_ids"]) != chunk_ids:
        print("    DESYNC  game_ids does not match the chunk entries")
        problems += 1
    if not problems:
        print("    consistent")
    return problems


def restore(namespace: Path, tag: str, *, delete: bool) -> None:
    path = namespace / SNAPSHOT_DIR / f"{tag}.json"
    if not path.exists():
        raise SystemExit(f"no snapshot {tag} at {path}")
    manifest = json.loads(path.read_text())
    current = json.loads((namespace / "manifest.json").read_text())
    print(f"  [restore] from {_describe(current)}")
    print(f"            to   {_describe(manifest)}")

    # Manifest first: it is the authoritative state, and writing it before
    # touching chunk files means an interruption leaves the store readable at
    # the restored point with orphans, which is the condition it tolerates.
    (namespace / "manifest.json").write_text(json.dumps(manifest))

    orphans = _on_disk(namespace) - _referenced(manifest)
    if not orphans:
        print("            no orphaned chunks")
        return
    if delete:
        for name in orphans:
            (namespace / "chunks" / name).unlink()
        print(f"            deleted {len(orphans)} orphaned chunks")
    else:
        quarantine = namespace / f"quarantine-{tag}-{int(time.time())}"
        quarantine.mkdir()
        for name in orphans:
            shutil.move(str(namespace / "chunks" / name), str(quarantine / name))
        print(f"            quarantined {len(orphans)} orphaned chunks in {quarantine.name}")
    print("            (an orphan left in place makes the next ingest of that game_id conflict)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "verify", "restore"))
    parser.add_argument("replay", type=Path, help="a run's replay directory")
    parser.add_argument("--tag", help="snapshot name; required for snapshot and restore")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete orphaned chunks rather than moving them into a quarantine directory.",
    )
    args = parser.parse_args()

    if args.action in {"snapshot", "restore"} and not args.tag:
        raise SystemExit(f"--tag is required for {args.action}")

    problems = 0
    for namespace in _namespaces(args.replay):
        print(f"{namespace.parent.name}/{namespace.name[:12]}...")
        if args.action == "snapshot":
            snapshot(namespace, args.tag)
        elif args.action == "verify":
            problems += verify(namespace)
        else:
            restore(namespace, args.tag, delete=args.delete)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
