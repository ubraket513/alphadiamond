"""The golden corpus is the game contract, and it must not drift silently.

This replaces the regeneration gate. That gate rebuilt the corpus from the
Python engine on every CI run and compared -- which declared C++ the authority
while leaving the Python implementation holding the back door to the
specification. See docs/architecture/decisions.md, decision 2.

What is checked now is provenance, not agreement: every payload file still
hashes to what the manifest recorded, the corpus hash still covers exactly
those files, and the contract version is the one the native side implements.

Changing the corpus deliberately means raising `game_contract_version` and
re-running `tools/freeze_golden.py`. Re-running it because a test went red is
the habit the freeze exists to prevent, and a reviewer seeing a manifest change
should ask which contract change it belongs to.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
MANIFEST = GOLDEN / "MANIFEST.json"

GAME_CONTRACT_VERSION = "diamond73-v1"
"""What the native core implements: `native/src/deployment_artifact.cpp` refuses
an artifact declaring anything else, and the encoder and action space are
versioned alongside it."""

GOLDEN_FORMAT_VERSION = 1


def _manifest() -> dict:
    assert MANIFEST.is_file(), f"the corpus has no manifest: {MANIFEST}"
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_the_manifest_declares_the_contract_the_binary_implements() -> None:
    manifest = _manifest()
    assert manifest["golden_format_version"] == GOLDEN_FORMAT_VERSION
    assert manifest["game_contract_version"] == GAME_CONTRACT_VERSION, (
        "the frozen corpus declares a different game contract than the native "
        "core implements; one of them moved without the other"
    )
    commit = manifest["oracle_commit"]
    assert commit is None or len(commit) == 40, "oracle_commit must be a commit id"


def test_every_payload_file_matches_its_recorded_digest() -> None:
    manifest = _manifest()
    payload = manifest["payload_sha256"]
    assert payload, "the manifest records no payload"

    drifted = []
    for name, expected in sorted(payload.items()):
        path = GOLDEN / name
        assert path.is_file(), f"the manifest names a missing file: {name}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            drifted.append(f"{name}: {actual} != {expected}")

    assert not drifted, (
        "the frozen game contract changed:\n  "
        + "\n  ".join(drifted)
        + "\nThis is not a fixture to refresh. If the rules changed on purpose, "
        "raise game_contract_version and re-freeze; if they did not, something "
        "edited the contract."
    )


def test_the_corpus_hash_covers_exactly_those_files() -> None:
    """Removing a file must change the corpus hash, not just its own entry."""
    manifest = _manifest()
    digest = hashlib.sha256()
    for name in sorted(manifest["payload_sha256"]):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(manifest["payload_sha256"][name].encode("ascii"))
        digest.update(b"\0")
    assert digest.hexdigest() == manifest["corpus_sha256"]


def test_no_stray_files_in_the_corpus() -> None:
    """Anything the manifest does not name is not part of the contract."""
    manifest = _manifest()
    named = set(manifest["payload_sha256"]) | {"MANIFEST.json"}
    present = {
        path.relative_to(GOLDEN).as_posix()
        for path in GOLDEN.rglob("*")
        if path.is_file()
    }
    stray = sorted(present - named)
    assert not stray, (
        f"files in tests/golden that the contract does not name: {stray}. "
        "Add them to the manifest by re-freezing, or delete them -- an "
        "unnamed file is one no test is pinning."
    )
