"""Build the `models/index.json` a packaged application reads at startup.

Models ship *beside* the executable, not compiled into it, so a model can be
replaced without relinking and different releases can carry different defaults.
The index names one default per family and records each artifact's digests, so
the application can verify what it is about to load.

Usage::

    python tools/build_model_index.py dist/AlphaDiamond/models \\
        --artifact soo=artifacts/soo-spike --default soo
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from diamond.alphazero.deployment import load_metadata

INDEX_VERSION = 1


def _entry(family: str, source: Path) -> dict:
    metadata = load_metadata(source / "metadata.json")
    if metadata["model_family"] != family:
        raise SystemExit(
            f"{source}: declared family {metadata['model_family']!r}, expected {family!r}"
        )
    return {
        "family": family,
        "version": metadata["model_version"],
        "path": f"{family}/{metadata['model_version']}",
        "architecture": metadata["architecture"],
        "model_sha256": metadata["model_sha256"],
        "runtime_sha256": metadata["runtime_sha256"],
        "source": metadata["source"],
    }


def build(output: Path, artifacts: dict[str, Path], defaults: list[str]) -> dict:
    if not artifacts:
        raise SystemExit("at least one --artifact is required")
    unknown = [family for family in defaults if family not in artifacts]
    if unknown:
        raise SystemExit(f"--default names a family with no artifact: {unknown}")

    entries = []
    output.mkdir(parents=True, exist_ok=True)
    for family, source in sorted(artifacts.items()):
        entry = _entry(family, source)
        target = output / entry["path"]
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        entries.append(entry)

    index = {
        "index_version": INDEX_VERSION,
        "defaults": {family: f"{family}/{_entry(family, artifacts[family])['version']}"
                     for family in sorted(defaults)},
        "models": entries,
    }
    (output / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="the models/ directory to write")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="FAMILY=PATH",
        help="a validated deployment artifact to include",
    )
    parser.add_argument(
        "--default",
        action="append",
        default=[],
        metavar="FAMILY",
        help="family whose bundled model is the default (repeatable)",
    )
    arguments = parser.parse_args()

    artifacts: dict[str, Path] = {}
    for item in arguments.artifact:
        family, separator, path = item.partition("=")
        if not separator:
            raise SystemExit(f"--artifact expects FAMILY=PATH, got {item!r}")
        artifacts[family] = Path(path)

    index = build(arguments.output, artifacts, arguments.default or sorted(artifacts))
    print(f"{arguments.output / 'index.json'}: {len(index['models'])} model(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
