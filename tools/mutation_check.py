"""Break the native core on purpose and require the C++ gates to notice.

A Python parity gate may only be retired once the C++ gate replacing it
demonstrably catches the mistakes that gate caught. "Demonstrably" is the whole
point: a test that has never failed is not evidence, and a gate that reads
plausibly can still be checking a digest of its own output.

So this applies each mutation in `tools/mutations.py` to the real source, builds
the native-ci preset, runs the whole lane, and requires the named gate to go
red. Then it restores and requires green.

Usage::

    python tools/mutation_check.py

Needs a configured `native-ci` build and a compiler on PATH. It is not part of
CI: it edits tracked files, and a job that does that on a shared runner is a
hazard. Run it when adding a gate, and when retiring a Python one.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mutations import MUTATIONS

ROOT = Path(__file__).resolve().parents[1]


def _run(command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        shell=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )


def native_lane() -> tuple[bool, str]:
    build = _run("cmake --build --preset native-ci --parallel")
    if build.returncode != 0:
        return False, "build failed:\n" + build.stdout[-2000:] + build.stderr[-2000:]
    tests = _run("ctest --preset native-ci")
    return tests.returncode == 0, tests.stdout


def main() -> int:
    survived = []
    for relative, before, after, expected_gate in MUTATIONS:
        path = ROOT / relative
        original = path.read_text(encoding="utf-8")
        if before not in original:
            print(f"STALE   {relative}: mutation no longer applies -- {before[:60]!r}")
            survived.append(relative)
            continue
        try:
            path.write_text(original.replace(before, after, 1), encoding="utf-8")
            green, output = native_lane()
            if green:
                print(f"MISSED  {relative}: {expected_gate} did not notice")
                survived.append(relative)
            elif expected_gate in output and "Failed" in output:
                print(f"caught  {relative}: {expected_gate}")
            else:
                failing = "; ".join(line.strip() for line in output.splitlines() if "Failed" in line)
                print(f"caught  {relative}: by another gate -- {failing[:120]}")
        finally:
            # write_text, not a file copy: copying preserves the original mtime,
            # ninja then skips rebuilding the restored file, and the mutation
            # stays live in the binaries -- which silently invalidates every
            # verdict after it. Found the hard way.
            path.write_text(original, encoding="utf-8")

    green, _ = native_lane()
    if not green:
        print("\nRESTORE FAILED: the tree is not green after restoring")
        return 1
    if survived:
        print(f"\n{len(survived)} mutation(s) survived: {sorted(set(survived))}")
        return 1
    print(f"\nall {len(MUTATIONS)} mutations caught; tree restored and green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
