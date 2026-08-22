"""Gate D on CPU: what the real Python callback costs the scheduler.

The GPU host is needed for Gate D's A/B throughput table. This is the part that
does not need it: the callback boundary itself -- GIL crossings per second,
evaluations per crossing, and where the single-evaluator-thread ceiling lands
once a real PyTorch forward sits inside it (risk 5).

Absolute evals/s here are CPU-bound and will not transfer to the GPU host. The
transferable quantities are the *ratios*: ValueOnly against PolicyValue, and
callback cost against everything the native side does.

Usage::

    python az-bench/profiles/bench_native_callback.py [--seconds N]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter
from diamond.alphazero.native import native_game, require_native
from diamond.alphazero.native.backend import policy_value_callback, value_only_callback
from diamond.alphazero.network.soo import SooModel
from diamond.game.state import build_players

CHECKPOINT = ROOT / "runtime" / "runs" / "soo" / "cpu8h-soo-20260819" / "latest.pt"


def load_model() -> SooModel:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = SooModel(NetworkConfig())
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def forward_cost(
    model: SooModel, batch: int, device: str = "cpu", repeats: int = 20
) -> tuple[float, float]:
    """Isolated per-batch cost of the two paths, in milliseconds.

    On CUDA the launch is asynchronous, so every timed region is bracketed by a
    synchronize. Without it this measures queueing, not the forward.
    """
    torch_device = torch.device(device)
    features = torch.zeros(batch, 73, 4, device=torch_device)

    def sync() -> None:
        if torch_device.type == "cuda":
            torch.cuda.synchronize()

    timings = []
    for value_only in (True, False):
        for _ in range(10):  # warm up; CUDA needs more than CPU does
            with torch.inference_mode():
                if value_only:
                    model.value_head(model.trunk(features).mean(dim=1))
                else:
                    model(features)
        sync()
        start = time.perf_counter()
        for _ in range(repeats):
            with torch.inference_mode():
                if value_only:
                    model.value_head(model.trunk(features).mean(dim=1))
                else:
                    model(features)
        sync()
        timings.append((time.perf_counter() - start) / repeats * 1000.0)
    return timings[0], timings[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu", help="torch device for the model, e.g. cuda:0")
    args = parser.parse_args()

    torch.set_num_threads(1)  # the evaluator is one thread; keep it that way
    module = require_native()
    players = build_players(2)
    native = native_game(players)
    state = AlphaZeroGameAdapter(players).initial_state()
    opening = module.State(
        occupancy=list(state.occupancy),
        current_player=state.current_player_id,
        turn_number=state.turn_number,
        status=0,
        finish_order=[],
    )
    model = load_model().to(torch.device(args.device))

    print(
        f"checkpoint: {CHECKPOINT.name}   torch threads: {torch.get_num_threads()}   "
        f"device: {args.device}"
    )
    print("\n--- model forward, isolated (ms/batch) ---")
    print(f"{'batch':>6} {'value_only':>11} {'full':>8} {'tail cost':>10}")
    for batch in (16, 32, 64, 128):
        value_only, full = forward_cost(model, batch, args.device)
        print(f"{batch:>6} {value_only:>11.3f} {full:>8.3f} {full / value_only:>9.2f}x")

    # The transferable measurement. The CPU forward here is ~30x the GPU host's
    # (87 ms vs 3.1 ms at batch 32), so absolute evals/s below say nothing about
    # Gate D. What does transfer is the cost of the boundary itself: marshalling
    # plus one GIL acquisition per batch, with no model inside.
    print("\n--- boundary cost, null callback (no model) ---")
    print(f"{'cap':>4} {'evals/s':>10} {'crossings/s':>12} {'us/crossing':>12} {'us/eval':>9} {'cpu%':>6}")
    for cap in (16, 32, 64, 128):
        calls = {"n": 0}

        def null(features, _calls=calls):
            _calls["n"] += 1
            return np.zeros(features.shape[0], dtype=np.float32)

        config = module.SchedulerConfig(
            games=2 * cap, threads=args.threads, max_batch=cap, max_wait_us=2000,
            simulations=args.simulations, seconds=args.seconds,
        )
        result = native.schedule_with_callback(opening, config, null, "value_only")
        wall = result["wall_seconds"]
        crossings = max(1, calls["n"])
        print(
            f"{cap:>4} {result['evaluations'] / wall:>10,.0f} {crossings / wall:>12,.0f} "
            f"{result['evaluator_seconds'] / crossings * 1e6:>12.1f} "
            f"{result['evaluator_seconds'] / max(1, result['evaluations']) * 1e6:>9.2f} "
            f"{result['worker_busy_seconds'] / (args.threads * wall) * 100:>6.1f}"
        )

    print("\n--- scheduler with the real callback ---")
    print(
        f"{'mode':>13} {'cap':>4} {'evals/s':>9} {'batch':>6} {'crossings/s':>12} "
        f"{'evals/crossing':>15} {'cpu%':>6} {'callback share':>15}"
    )
    for mode, factory in (("value_only", value_only_callback), ("policy_value", policy_value_callback)):
        for cap in (32, 64):
            calls = {"n": 0}
            inner = factory(model, device=args.device)

            if mode == "value_only":
                def wrapped(features, _inner=inner, _calls=calls):
                    _calls["n"] += 1
                    return _inner(features)
            else:
                def wrapped(features, actions, offsets, _inner=inner, _calls=calls):
                    _calls["n"] += 1
                    return _inner(features, actions, offsets)

            config = module.SchedulerConfig(
                games=2 * cap,
                threads=args.threads,
                max_batch=cap,
                max_wait_us=2000,
                simulations=args.simulations,
                seconds=args.seconds,
            )
            result = native.schedule_with_callback(opening, config, wrapped, mode)
            wall = result["wall_seconds"]
            batches = list(result["batch_sizes"])
            print(
                f"{mode:>13} {cap:>4} {result['evaluations'] / wall:>9,.0f} "
                f"{statistics.mean(batches):>6.1f} {calls['n'] / wall:>12,.0f} "
                f"{result['evaluations'] / max(1, calls['n']):>15.1f} "
                f"{result['worker_busy_seconds'] / (args.threads * wall) * 100:>6.1f} "
                f"{result['evaluator_seconds'] / wall * 100:>14.1f}%"
            )


if __name__ == "__main__":
    main()
