"""Generated payloads must not re-enter Git.

Build directories and training checkpoints were tracked once already; ignore
rules do not untrack what is already tracked, so this test is the standing
guard. It shells out to ``git ls-files`` rather than walking the tree: only
what Git tracks matters here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PREFIXES = ("build/", "build-", "out/", "artifacts/", "runs/", "dist/")
FORBIDDEN_SUFFIXES = (".obj", ".o", ".lib", ".a", ".pdb", ".ilk", ".exp", ".so", ".pyd", ".dll")

# Weights are never source. Each exception is a deliberate, documented fixture.
WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors")
ALLOWED_WEIGHTS = {"runtime/runs/soo/cpu8h-soo-20260819/latest.pt"}

# Nothing tracked may exceed this. The step-80 checkpoint above is the one
# thing that does, and it is listed rather than exempted by size.
MAX_TRACKED_BYTES = 1_000_000
ALLOWED_LARGE = ALLOWED_WEIGHTS


def _tracked() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:  # pragma: no cover - not a git checkout
        pytest.skip("not a git checkout")
    return [path for path in result.stdout.split("\0") if path]


TRACKED = _tracked()


def test_no_build_output_is_tracked() -> None:
    offenders = [p for p in TRACKED if p.startswith(FORBIDDEN_PREFIXES)]
    offenders += [p for p in TRACKED if p.endswith(FORBIDDEN_SUFFIXES)]
    assert not offenders, f"generated files are tracked: {sorted(set(offenders))[:10]}"


def test_only_documented_weights_are_tracked() -> None:
    weights = {p for p in TRACKED if p.endswith(WEIGHT_SUFFIXES)}
    assert weights <= ALLOWED_WEIGHTS, (
        f"undocumented weights tracked: {sorted(weights - ALLOWED_WEIGHTS)}; "
        "put checkpoints in the bucket, not in Git"
    )


def test_no_unexpected_large_files_are_tracked() -> None:
    oversized = []
    for path in TRACKED:
        if path in ALLOWED_LARGE:
            continue
        absolute = ROOT / path
        if absolute.is_file() and absolute.stat().st_size > MAX_TRACKED_BYTES:
            oversized.append((path, absolute.stat().st_size))
    assert not oversized, f"large files tracked: {oversized}"


def test_the_bucket_root_contributes_only_documentation() -> None:
    bucket = {p for p in TRACKED if p.startswith("TrainAlphaDiamond/")}
    assert bucket, "TrainAlphaDiamond/README.md should be tracked"
    for path in bucket:
        assert path == "TrainAlphaDiamond/README.md" or path.startswith(
            "TrainAlphaDiamond/manifests/"
        ), f"bucket content is tracked: {path}"
