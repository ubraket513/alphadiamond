"""Generate benchmark-only config copies for the RTX 5060 Ti scaling sweep.

The canonical ``runtime/configs/soo-rtx3060.json`` stays a stable production
reference and is never edited in place for a sweep (that hand-editing is what
broke two tests during the previous GPU pass).  Each point instead gets its own
frozen copy under ``az-bench/configs/``.

Only two fields vary across the worker sweep -- ``workers.games_per_iteration``
and, for the later timer sweep, ``inference.max_wait_ms``.  ``worker_count``
stays ``null`` because ``az_train.py`` takes it from ``--workers``; everything
else is inherited byte-for-byte from the canonical config so a row can differ
from another only in the dimension under test.

``games_per_iteration`` has to move with the worker count: the pool computes
``lane_count = min(worker_count, len(jobs))``, so ``--workers 64`` against 32
games would silently measure 32 lanes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CANONICAL = REPO / "runtime" / "configs" / "soo-rtx3060.json"
OUT_DIR = REPO / "az-bench" / "configs"

#: Worker-scaling points. games == workers so every job gets its own lane.
WORKER_POINTS = (30, 36, 48, 64, 80, 96)

#: Timer points, applied at the winning worker count only.
WAIT_POINTS = (1, 2, 4, 8)


def base_config() -> dict:
    return json.loads(CANONICAL.read_text(encoding="utf-8"))


def write(name: str, config: dict) -> Path:
    path = OUT_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-games",
        type=int,
        default=32,
        help="games for the 30-worker baseline row, carried over from the RTX 3060.",
    )
    args = parser.parse_args()

    written: list[Path] = []
    for workers in WORKER_POINTS:
        config = base_config()
        # The 30-worker row keeps 32 games so it reproduces the inherited
        # baseline exactly; every other row gives each job its own lane.
        games = args.baseline_games if workers == 30 else workers
        config["workers"]["games_per_iteration"] = games
        written.append(write(f"soo-rtx5060-w{workers}-g{games}", config))

    for wait in WAIT_POINTS:
        config = base_config()
        config["inference"]["max_wait_ms"] = wait
        # games_per_iteration is filled in per-run by the driver once the
        # winning worker count is known; keep the canonical value until then.
        written.append(write(f"soo-rtx5060-wait{wait}", config))

    for path in written:
        print(path.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
