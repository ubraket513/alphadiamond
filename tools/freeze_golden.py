"""Freeze the golden corpus as the normative game contract.

Writes `tests/golden/MANIFEST.json`: the format version, the game-contract
version, a hash per payload file, a hash over the whole corpus, and the commit
that produced it. After this, CI verifies the manifest rather than regenerating
the files from the Python engine -- see docs/architecture/decisions.md.

This is not a routine step. Run it when the game contract is *deliberately*
changed, together with a new contract version; regenerating because a test went
red is the habit the freeze exists to prevent.

Usage::

    python tools/freeze_golden.py [--contract diamond73-v1] [--format 1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
MANIFEST = GOLDEN / "MANIFEST.json"

PAYLOAD = (
    "rules-v1.txt",
    "mcts-v1.txt",
    "mcts3p-v1.txt",
    "topology/topology_neighbour.i8",
    "topology/topology_camp_positions.i32",
    "topology/topology_pairwise_distance.i32",
    "topology/topology_physical_to_canonical.i32",
    "topology/topology_canonical_to_physical.i32",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus_digest(payload: dict[str, str]) -> str:
    """One hash over the whole corpus: names and digests, in a fixed order.

    Includes the names, so removing a file changes the corpus hash even though
    every remaining file still matches.
    """
    digest = hashlib.sha256()
    for name in sorted(payload):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload[name].encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _oracle_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - not a checkout
        return None
    commit = result.stdout.strip()
    return commit if len(commit) == 40 else None


def build(contract: str, format_version: int) -> dict:
    missing = [name for name in PAYLOAD if not (GOLDEN / name).is_file()]
    if missing:
        raise SystemExit(f"missing golden payload: {missing}")

    payload = {name: _digest(GOLDEN / name) for name in PAYLOAD}
    return {
        "golden_format_version": format_version,
        "game_contract_version": contract,
        "corpus_sha256": corpus_digest(payload),
        "oracle_commit": _oracle_commit(),
        "payload_sha256": payload,
        "note": (
            "Normative game contract, not a regenerated fixture. CI verifies "
            "these digests; it does not rebuild the corpus from the Python "
            "engine. A deliberate rules change raises game_contract_version and "
            "creates a new corpus on purpose. See docs/architecture/decisions.md."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default="diamond73-v1")
    parser.add_argument("--format", type=int, default=1, dest="format_version")
    arguments = parser.parse_args()

    manifest = build(arguments.contract, arguments.format_version)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{MANIFEST}: {len(manifest['payload_sha256'])} files")
    print(f"  contract:  {manifest['game_contract_version']}")
    print(f"  corpus:    {manifest['corpus_sha256']}")
    print(f"  oracle:    {manifest['oracle_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
