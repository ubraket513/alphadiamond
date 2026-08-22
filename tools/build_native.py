"""Build the optional native self-play extension in place.

    python tools/build_native.py            # build into src/diamond/...
    DIAMOND_NATIVE_ARCH=native python tools/build_native.py

The extension lands next to its Python guard so an editable checkout picks it
up without reinstalling.  It is optional by design: if this fails, the Python
backend -- the default and the parity oracle -- is unaffected.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        return result.returncode

    built = sorted((ROOT / "src/diamond/alphazero/native").glob("_diamond_native*"))
    if not built:
        print("no extension produced; is pybind11 installed?", file=sys.stderr)
        return 1
    for path in built:
        print(f"built {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
