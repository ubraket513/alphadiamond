"""Gate C.2/C.3: multi-game scheduler throughput, and the latency sweep.

Native only -- no Python in the timed region, no PyTorch, no GIL.  The dummy
evaluator has a configurable per-batch cost so the scheduler can be judged at
the inference latency it will actually face.

Three modes:

    --mode starve    find where lane starvation stops (games vs threads)
    --mode batch     batch-cap sweep at a fixed operating point
    --mode latency   artificial evaluator latency sweep

The point of the latency sweep: value-only PyTorch measured ~3.8 ms per batch of
32 (section 0.4).  If the scheduler still fills B32/B64 at a 3-4 ms dummy
latency, Gate D has a real chance; if it cannot, no amount of GPU speed helps.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diamond.alphazero.game_adapter import AlphaZeroGameAdapter
from diamond.alphazero.native import native_game, require_native
from diamond.contract.state import build_players


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


class Bench:
    def __init__(self) -> None:
        self.module = require_native()
        self.players = build_players(2)
        self.native = native_game(self.players)
        state = AlphaZeroGameAdapter(self.players).initial_state()
        self.opening = self.module.State(
            occupancy=list(state.occupancy),
            current_player=state.current_player_id,
            turn_number=state.turn_number,
            status=0,
            finish_order=[],
        )

    def run(self, **kwargs) -> dict:
        config = self.module.SchedulerConfig(**kwargs)
        raw = self.native.schedule(self.opening, config)
        batches = list(raw["batch_sizes"])
        waits_ms = [ns / 1e6 for ns in raw["wait_ns"]]
        wall = raw["wall_seconds"]
        threads = kwargs.get("threads", 4)
        return {
            "evals_s": raw["evaluations"] / wall,
            "moves_s": raw["moves"] / wall,
            "batches": raw["batches"],
            "batch_mean": statistics.mean(batches) if batches else 0.0,
            "batch_p50": percentile([float(b) for b in batches], 0.50),
            "batch_p90": percentile([float(b) for b in batches], 0.90),
            "runnable": statistics.mean(raw["ready_depth"]) if raw["ready_depth"] else 0.0,
            "waiting": statistics.mean(raw["waiting"]) if raw["waiting"] else 0.0,
            "cpu": raw["worker_busy_seconds"] / (threads * wall) * 100.0,
            "wait_p50": percentile(waits_ms, 0.50),
            "wait_p90": percentile(waits_ms, 0.90),
            "wakeups_s": raw["batcher_wakeups"] / wall,
            "evaluator_share": raw["evaluator_seconds"] / wall * 100.0,
        }


HEADER = (
    f"{'games':>6} {'thr':>4} {'cap':>4} {'lat_ms':>7} "
    f"{'evals/s':>10} {'batch_mean':>11} {'p50':>5} {'p90':>5} "
    f"{'runnable':>9} {'waiting':>8} {'cpu%':>6} {'wait_p50':>9} {'wait_p90':>9}"
)


def show(games: int, threads: int, cap: int, latency: float, row: dict) -> None:
    print(
        f"{games:>6} {threads:>4} {cap:>4} {latency:>7.1f} "
        f"{row['evals_s']:>10,.0f} {row['batch_mean']:>11.1f} {row['batch_p50']:>5.0f} "
        f"{row['batch_p90']:>5.0f} {row['runnable']:>9.1f} {row['waiting']:>8.1f} "
        f"{row['cpu']:>6.0f} {row['wait_p50']:>9.2f} {row['wait_p90']:>9.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("starve", "batch", "latency"), default="starve")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--max-wait-us", type=int, default=2000)
    args = parser.parse_args()

    bench = Bench()
    common = {
        "simulations": args.simulations,
        "max_wait_us": args.max_wait_us,
        "seconds": args.seconds,
    }
    print(HEADER)

    if args.mode == "starve":
        # Where does lane starvation stop? Sweep games against threads at a
        # realistic evaluator latency; do not sweep the full grid.
        for threads in (1, 2, 4, 8):
            for games in (32, 64, 128, 256, 512):
                row = bench.run(
                    games=games, threads=threads, max_batch=32, eval_latency_ms=3.0, **common
                )
                show(games, threads, 32, 3.0, row)
            print()

    elif args.mode == "batch":
        for cap in (16, 32, 64, 128):
            for games in (128, 256, 512):
                row = bench.run(
                    games=games, threads=4, max_batch=cap, eval_latency_ms=3.0, **common
                )
                show(games, 4, cap, 3.0, row)
            print()

    else:
        for latency in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
            for cap in (32, 64):
                row = bench.run(
                    games=256, threads=4, max_batch=cap, eval_latency_ms=latency, **common
                )
                show(256, 4, cap, latency, row)
            print()


if __name__ == "__main__":
    main()
