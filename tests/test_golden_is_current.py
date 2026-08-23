"""The committed golden file must still be what the Python oracle produces.

The C++ tests read ``tests/golden/`` without Python. That is only trustworthy
while the file matches the oracle, so this test regenerates it into a temporary
directory and compares. It is the link that keeps the native gates honest as
the Python side changes.
"""

from __future__ import annotations

import filecmp
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
GENERATOR = ROOT / "tools" / "build_golden.py"

FILES = [
    "rules-v1.txt",
    "mcts-v1.txt",
    "topology/topology_neighbour.i8",
    "topology/topology_camp_positions.i32",
    "topology/topology_pairwise_distance.i32",
    "topology/topology_physical_to_canonical.i32",
    "topology/topology_canonical_to_physical.i32",
]


def _first_difference(name: str, committed: Path, regenerated: Path) -> str:
    """Name the first differing line.

    "The file changed" is not an actionable report when the file has thousands
    of lines and one number in one of them moved.
    """
    if not name.endswith(".txt"):
        return f"{name}: binary difference"
    before = committed.read_text(encoding="utf-8").splitlines()
    after = regenerated.read_text(encoding="utf-8").splitlines()
    if len(before) != len(after):
        return f"{name}: {len(before)} lines committed, {len(after)} regenerated"
    for index, (old, new) in enumerate(zip(before, after), start=1):
        if old != new:
            return f"{name}:{index}\n  committed:   {old}\n  regenerated: {new}"
    return f"{name}: differs only in line endings"


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
        name for name in FILES if not filecmp.cmp(GOLDEN / name, tmp_path / name, shallow=False)
    ]
    detail = "\n".join(
        _first_difference(name, GOLDEN / name, tmp_path / name) for name in stale
    )
    assert not stale, (
        f"golden files are stale: {stale}; the Python oracle changed. "
        "Re-run `make golden`, and treat any C++ test that now fails as the "
        f"port drifting, not as a fixture to overwrite.\n{detail}"
    )
