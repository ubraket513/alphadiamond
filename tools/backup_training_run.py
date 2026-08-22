"""Copy a training run's durable state somewhere that outlives this instance.

This host reports ``workspace_is_volume: false``.  A stop/start is safe; a
recycle or destroy wipes the run.  The run is an experimental asset, so it needs
a durable copy -- and no rclone remote or Hugging Face token is configured here,
so the only durable target available is a GitHub Release on the project's own
repository, which already tracks a checkpoint of exactly this kind.

Three artefacts:

===================  ==================================================
``latest.pt``        the network
``replay.tar.gz``    manifest + retained chunks (357 MB -> ~15 MB)
``run-state.tar.gz`` config, ledger, loop_state, provenance
===================  ==================================================

Two properties make this a backup rather than a hopeful copy.

**It is validated before it is trusted.**  A checkpoint torn by a concurrent
write still has a plausible size; only loading it proves anything.  A replay
archive whose manifest references a chunk it does not contain is a backup of
nothing, because the manifest is the store's only authority.

**It does not pause the run.**  Stopping a healthy trainer to back it up is the
wrong trade, so the copy is retried until it is *provably* consistent.  That is
sound because of how the store writes: chunks are written before the manifest
that references them, and pruning rewrites the manifest before unlinking, so any
capture whose manifest is fully covered by its own chunks is a real point in
time.

Publishing is a separate, deliberate step.  Staging and validating is the
default; ``--publish`` is what uploads.  The repository is public, so uploading
should never be a side effect of taking a backup.

Usage::

    python tools/backup_training_run.py --stage-dir /path   # validate only
    python tools/backup_training_run.py --publish           # validate, then upload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_NOTES = (
    "Durable snapshot of the from-scratch Soo run: network, bounded replay "
    "store, run state and provenance. Taken because this host reports "
    "workspace_is_volume=false. Validated before upload: the checkpoint loads "
    "and every manifest-referenced chunk is present."
)
DEFAULT_TRAIN_ROOT = Path("/workspace/alphadiamond-training")
COPY_ATTEMPTS = 5


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _checkpoint_step(path: Path) -> int:
    """Load it.  A torn checkpoint has a plausible size and will not load."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not payload.get("model_state_dict"):
        raise SystemExit(f"{path} has no model_state_dict")
    return int(payload["training_step"])


def _replay_is_consistent(archive: Path) -> tuple[bool, str]:
    """Every chunk the manifest references must be inside the archive."""
    referenced = 0
    missing = 0
    with tarfile.open(archive) as tar:
        names = set(tar.getnames())
        manifests = [name for name in names if name.endswith("manifest.json")]
        if not manifests:
            return False, "archive contains no manifest"
        for name in manifests:
            member = tar.extractfile(name)
            if member is None:
                return False, f"{name} is unreadable"
            payload = json.load(member)
            base = name.rsplit("/", 1)[0]
            for chunk in payload["chunks"]:
                referenced += 1
                stem = hashlib.sha256(chunk["game_id"].encode("utf-8")).hexdigest()
                if f"{base}/chunks/{stem}.json" not in names:
                    missing += 1
    return missing == 0, f"{referenced} chunks referenced, {missing} missing"


def _capture_replay(run: Path, target: Path) -> None:
    for attempt in range(1, COPY_ATTEMPTS + 1):
        subprocess.run(
            ["tar", "--warning=no-file-changed", "-czf", str(target), "-C", str(run), "replay"],
            check=False,
        )
        ok, detail = _replay_is_consistent(target)
        print(f"  replay attempt {attempt}: {detail}")
        if ok:
            return
        if attempt == COPY_ATTEMPTS:
            raise SystemExit("could not capture a consistent replay snapshot")
        # The trainer writes in bursts between iterations; a short wait lands
        # the next attempt in a quiet window rather than the same one.
        time.sleep(10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--run-id", default="soo-scratch-20260822")
    parser.add_argument("--config", type=Path, default=ROOT / "runtime/configs/soo-rtx5090-native.json")
    parser.add_argument("--stage-dir", type=Path, default=None)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Upload to a GitHub Release. Without it, nothing leaves this host.",
    )
    parser.add_argument("--tag-suffix", default="")
    args = parser.parse_args()

    run = args.train_root / "runs" / "soo" / args.run_id
    checkpoint = run / "latest.pt"
    if not checkpoint.is_file():
        raise SystemExit(f"no checkpoint at {checkpoint}")

    stage = args.stage_dir or Path(tempfile.mkdtemp(prefix="az-backup-"))
    stage.mkdir(parents=True, exist_ok=True)
    state_dir = stage / "run-state"
    state_dir.mkdir(exist_ok=True)

    print(f"[backup] run={args.run_id}")
    # Copy the checkpoint first: it is the artefact that matters most and the
    # one written least often, so it is the least likely to be caught torn.
    (stage / "latest.pt").write_bytes(checkpoint.read_bytes())
    step = _checkpoint_step(stage / "latest.pt")
    print(f"  checkpoint loads, training_step={step}")

    _capture_replay(run, stage / "replay.tar.gz")

    for name in ("config.json", "ledger.jsonl", "loop_state.json"):
        source = run / name
        if source.is_file():
            (state_dir / name).write_bytes(source.read_bytes())
    if args.config.is_file():
        (state_dir / args.config.name).write_bytes(args.config.read_bytes())

    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    provenance = "\n".join(
        [
            f"run_id={args.run_id}",
            f"training_step={step}",
            f"captured_at={datetime.now(UTC).isoformat()}",
            f"git_head={head}",
            f"latest_pt_sha256={_digest(stage / 'latest.pt')}",
            f"replay_sha256={_digest(stage / 'replay.tar.gz')}",
        ]
    )
    (state_dir / "provenance.txt").write_text(provenance + "\n", encoding="utf-8")
    with tarfile.open(stage / "run-state.tar.gz", "w:gz") as tar:
        tar.add(state_dir, arcname="run-state")

    print(provenance.replace("\n", "\n  ").rjust(2))
    for name in ("latest.pt", "replay.tar.gz", "run-state.tar.gz"):
        size = (stage / name).stat().st_size / 1048576
        print(f"  {name}: {size:.1f} MB")

    if not args.publish:
        print(f"[backup] staged and validated at {stage}; nothing published")
        return 0

    tag = f"run-{args.run_id}-step{step:08d}"
    if args.tag_suffix:
        tag = f"{tag}-{args.tag_suffix}"
    subprocess.run(
        [
            "gh", "release", "create", tag,
            "--title", f"Training run {args.run_id} @ step {step}",
            "--notes", RELEASE_NOTES,
            f"{stage / 'latest.pt'}#latest.pt",
            f"{stage / 'replay.tar.gz'}#replay.tar.gz",
            f"{stage / 'run-state.tar.gz'}#run-state.tar.gz",
        ],
        check=True,
    )
    print(f"[backup] published {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
