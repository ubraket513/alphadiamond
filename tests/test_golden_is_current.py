"""The committed golden file must still be what the Python oracle produces.

The C++ tests read ``tests/golden/`` without Python. That is only trustworthy
while the file matches the oracle, so this test regenerates it into a temporary
directory and compares. It is the link that keeps the native gates honest as
the Python side changes.

Almost everything is compared byte for byte. The exception is the bootstrap
prior's two float columns in ``rules-v1.txt``: they are a softmax summed in
Python, and the last ulp of that sum is not stable across interpreter versions
(measured: 8.4509626902822728 on 3.12, ...745 on 3.11, with every digest in the
same record identical). Demanding bit-equality there would make the gate fail
on the interpreter rather than on the port, so those two columns are compared
with the same tolerance the C++ test uses. Everything that can be compared
exactly still is.
"""

from __future__ import annotations

import filecmp
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
GENERATOR = ROOT / "tools" / "build_golden.py"

PRIOR_TOLERANCE = 1e-9
"""Matches kPriorTolerance in native/tests/rules_golden_test.cpp."""

EXACT_FILES = [
    "mcts-v1.txt",
    "topology/topology_neighbour.i8",
    "topology/topology_camp_positions.i32",
    "topology/topology_pairwise_distance.i32",
    "topology/topology_physical_to_canonical.i32",
    "topology/topology_canonical_to_physical.i32",
]
TOLERANT_FILES = ["rules-v1.txt"]
FILES = TOLERANT_FILES + EXACT_FILES


def _differences(committed: Path, regenerated: Path) -> list[str]:
    """Line-by-line, with the prior's float columns compared numerically."""
    before = committed.read_text(encoding="utf-8").splitlines()
    after = regenerated.read_text(encoding="utf-8").splitlines()
    if len(before) != len(after):
        return [f"{len(before)} lines committed, {len(after)} regenerated"]

    problems = []
    for index, (old, new) in enumerate(zip(before, after), start=1):
        if old == new:
            continue
        old_fields = old.split()
        new_fields = new.split()
        # `exp <n> <4 digests> <prior_count> <prior_max> <prior_dot>`
        if (
            old_fields[:1] == ["exp"]
            and len(old_fields) == len(new_fields) == 9
            and old_fields[:7] == new_fields[:7]
            and all(
                abs(float(a) - float(b)) <= PRIOR_TOLERANCE
                for a, b in zip(old_fields[7:], new_fields[7:])
            )
        ):
            continue
        problems.append(f"line {index}\n  committed:   {old}\n  regenerated: {new}")
        if len(problems) >= 3:
            break
    return problems


def test_golden_matches_the_python_oracle(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )

    # Regenerating into an empty directory also proves --output is honoured:
    # a generator that ignored it would leave tmp_path empty and fail here
    # rather than silently comparing the committed files with themselves.
    for name in FILES:
        assert (tmp_path / name).is_file(), f"generator did not write {name}"
        assert (GOLDEN / name).is_file(), f"missing committed golden file: {name}"

    stale = [
        name
        for name in EXACT_FILES
        if not filecmp.cmp(GOLDEN / name, tmp_path / name, shallow=False)
    ]
    detail: list[str] = [f"{name}: differs" for name in stale]
    for name in TOLERANT_FILES:
        problems = _differences(GOLDEN / name, tmp_path / name)
        if problems:
            stale.append(name)
            detail.extend(f"{name}:{problem}" for problem in problems)

    assert not stale, (
        f"golden files are stale: {stale}; the Python oracle changed. "
        "Re-run `make golden`, and treat any C++ test that now fails as the "
        "port drifting, not as a fixture to overwrite.\n" + "\n".join(detail)
    )
