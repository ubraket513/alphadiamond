"""Move a bucket checkpoint along archival -> candidate -> promoted.

The manifest already carries a `state`; nothing drove it, so "which checkpoint
is the release" lived in someone's head. This is the transition, and it refuses
the ways it can go wrong:

* the manifest must match its checkpoint. A digest that has drifted means the
  file was replaced under a path that was supposed to be immutable, and every
  measurement recorded against it is no longer comparable.
* states move forward one step at a time. Promoting straight from archival
  skips the conversion that `candidate` exists to gate.
* promotion converts. A `promoted` state whose artifact was never exported is
  the exact bookkeeping this is meant to remove.

Usage::

    python tools/promote_checkpoint.py TrainAlphaDiamond/checkpoints/soo/<run>/step-00044250 \\
        --to candidate
    python tools/promote_checkpoint.py <same path> --to promoted --artifacts artifacts/soo-release
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

STATES = ("archival", "candidate", "promoted")
"""In order. A transition may move one step forward, or stay where it is."""

CHECKPOINT_NAME = "checkpoint.pt"
MANIFEST_NAME = "manifest.json"


class PromotionError(RuntimeError):
    """The checkpoint cannot move to the requested state."""


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(directory: Path) -> dict:
    path = directory / MANIFEST_NAME
    if not path.is_file():
        raise PromotionError(f"no manifest at {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != 1:
        raise PromotionError(f"unsupported manifest version: {manifest.get('manifest_version')!r}")
    if manifest.get("state") not in STATES:
        raise PromotionError(f"unknown state: {manifest.get('state')!r}")
    return manifest


def verify(directory: Path, manifest: dict) -> None:
    """The manifest must still describe the file next to it."""
    checkpoint = directory / CHECKPOINT_NAME
    if not checkpoint.is_file():
        raise PromotionError(f"no checkpoint at {checkpoint}")
    actual = _digest(checkpoint)
    if actual != manifest["checkpoint_sha256"]:
        raise PromotionError(
            f"{checkpoint} hashes to {actual}, manifest says {manifest['checkpoint_sha256']}; "
            "an immutable path was overwritten and measurements against it no longer compare"
        )


def next_state(current: str, target: str) -> str:
    if target not in STATES:
        raise PromotionError(f"unknown target state: {target!r}")
    if current == target:
        return target
    if STATES.index(target) != STATES.index(current) + 1:
        raise PromotionError(
            f"cannot go from {current} to {target}: states move one step forward, and "
            "candidate is where conversion is gated"
        )
    return target


def promote(
    directory: Path,
    target: str,
    *,
    artifacts: Path | None = None,
    export: bool = True,
) -> dict:
    manifest = load_manifest(directory)
    verify(directory, manifest)
    resolved = next_state(manifest["state"], target)

    if resolved == "promoted" and export:
        if artifacts is None:
            raise PromotionError("promotion needs --artifacts: a promoted checkpoint is converted")
        # Imported here so the state machine above stays testable without torch.
        from export_deployment import export as export_artifact

        metadata = export_artifact(
            artifacts,
            family=manifest["model_family"],
            checkpoint=directory / CHECKPOINT_NAME,
            training_commit=manifest.get("training_commit"),
            training_step=manifest.get("training_step"),
        )
        manifest["deployment"] = {
            "artifact_path": artifacts.as_posix(),
            "model_sha256": metadata["model_sha256"],
            "runtime_sha256": metadata["runtime_sha256"],
        }

    manifest["state"] = resolved
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="an immutable step-<n> directory")
    parser.add_argument("--to", required=True, choices=STATES)
    parser.add_argument("--artifacts", type=Path, help="where to write the deployment artifact")
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="record the state without converting (for recovering a botched run)",
    )
    arguments = parser.parse_args()

    try:
        manifest = promote(
            arguments.checkpoint,
            arguments.to,
            artifacts=arguments.artifacts,
            export=not arguments.no_export,
        )
    except PromotionError as error:
        raise SystemExit(str(error)) from error

    print(f"{arguments.checkpoint}: {manifest['state']}")
    if "deployment" in manifest:
        print(f"  artifact: {manifest['deployment']['artifact_path']}")
        print(f"  runtime_sha256: {manifest['deployment']['runtime_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
