"""Derive the CPU session's runtime configs from the checked-in bootstrap configs.

Blueprint section 5: only the runtime copy is tuned; the canonical reference
configs under ``configs/alphazero/`` are never mutated for one experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Calibration outcome.  Eight simulations wins on an untrained network but
# collapses to 17% completion once the value head has been trained even briefly;
# 32 stays at 100% and is faster in wall-clock.  See blueprint/cpu_train_runbook.md.
SIMULATIONS = 32
WORKER_COUNT = 4  # one process per physical core
GAMES_PER_ITERATION = 16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    out_dir = args.root / "runtime" / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in ("soo", "min"):
        config = json.loads(
            (args.root / "configs" / "alphazero" / f"{model}-bootstrap.json").read_text(
                encoding="utf-8"
            )
        )
        config["mcts"]["simulations"] = SIMULATIONS
        config["workers"]["worker_count"] = WORKER_COUNT
        config["workers"]["games_per_iteration"] = GAMES_PER_ITERATION
        config["training"]["device"] = "cpu"
        target = out_dir / f"{model}-cpu8h.json"
        target.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
