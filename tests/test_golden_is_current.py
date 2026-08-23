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
    "topology/topology_neighbour.i8",
    "topology/topology_camp_positions.i32",
    "topology/topology_pairwise_distance.i32",
    "topology/topology_physical_to_canonical.i32",
    "topology/topology_canonical_to_physical.i32",
]


def test_golden_matches_the_python_oracle(tmp_path: Path) -> None:
    for name in FILES:
        source = GOLDEN / name
        assert source.is_file(), f"missing golden file: {name}"
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())

    subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )

    stale = [name for name in FILES if not filecmp.cmp(GOLDEN / name, tmp_path / name, shallow=False)]
    assert not stale, (
        f"golden files are stale: {stale}; the Python oracle changed. "
        "Re-run `make golden`, and treat any C++ test that now fails as the "
        "port drifting, not as a fixture to overwrite."
    )
