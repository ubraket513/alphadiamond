"""Gate D on the GPU host: the A/B throughput table against the Python backend.

This is the half of Gate D that ``bench_native_callback.py`` explicitly could
not answer on a laptop -- what the native backend does when a *real* forward
runs at GPU speed behind the callback, and how that compares with the process
-pool Python backend on the same machine, same checkpoint, same seed.

Three quantities, and the difference between them is the point:

``roofline``
    ``max_batch / forward_seconds``.  What the evaluator thread could achieve if
    batch formation were free and always full.

``real model``
    The native scheduler driving the actual Soo checkpoint through the Python
    callback on the GPU.  **Lanes lock-step**: with no Dirichlet noise and no
    temperature, a real evaluator is a pure function of the position, so every
    lane from the same opening plays the same game (see
    ``test_a_real_evaluator_makes_every_lane_play_the_same_game``).  A dense
    forward does not care that its rows are equal, so this number is close to
    honest -- but it is measured over one trajectory, not many.

``diverse``
    The Gate C dummy evaluator -- per-lane salted, provably 2*cap distinct games
    -- with ``eval_latency_ms`` pinned to *this host's measured forward*.  This
    is the number to quote.  It is what the scheduler achieves at real GPU
    latency with real lane diversity, and it is the pessimistic one.

The Python side is not run here.  It is a full training run; drive it with
``run_point.sh`` and pass the resulting seconds in with ``--python-seconds``.

Usage::

    python az-bench/profiles/bench_native_gpu_ab.py --device cuda:0 \
        --python-seconds 79 --python-samples 4410
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter
from diamond.alphazero.native import native_game, require_native
from diamond.alphazero.native.backend import policy_value_callback, value_only_callback
from diamond.alphazero.network.soo import SooModel
from diamond.game.state import build_players

CHECKPOINT = ROOT / "runtime" / "runs" / "soo" / "cpu8h-soo-20260819" / "latest.pt"
CHECKPOINT_SHA = "1634b901e213b065c107eea734b8c172c14babb1c2565352203961e86ea165af"


def load_model() -> SooModel:
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA:
        raise SystemExit(f"checkpoint digest {digest} != expected {CHECKPOINT_SHA}")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = SooModel(NetworkConfig())
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def forward_ms(
    model: SooModel, batch: int, device: str, mode: str = "value_only", repeats: int = 50
) -> float:
    """Isolated cost of one batch on the mode's own forward path, in milliseconds.

    The roofline must be built from the forward the mode actually runs.  Timing
    ValueOnly and dividing a PolicyValue run by it reports a roofline the mode
    was never able to reach.

    CUDA launches are asynchronous; without the synchronize this times the
    enqueue and reports a forward roughly two orders of magnitude too fast.
    """
    torch_device = torch.device(device)
    features = torch.zeros(batch, 73, 4, device=torch_device)

    def sync() -> None:
        if torch_device.type == "cuda":
            torch.cuda.synchronize()

    def once() -> None:
        with torch.inference_mode():
            if mode == "value_only":
                model.value_head(model.trunk(features).mean(dim=1))
            else:
                model(features)

    for _ in range(10):
        once()
    sync()
    start = time.perf_counter()
    for _ in range(repeats):
        once()
    sync()
    return (time.perf_counter() - start) / repeats * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--caps", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--mode", default="value_only", choices=("value_only", "policy_value"))
    parser.add_argument(
        "--python-seconds",
        type=float,
        default=None,
        help="self-play seconds from the Python backend reference run (run_point.sh)",
    )
    parser.add_argument("--python-samples", type=int, default=None)
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

    python_rate = None
    if args.python_seconds and args.python_samples:
        # One sample is one move; each move costs `simulations` evaluations.
        python_rate = args.python_samples * args.simulations / args.python_seconds

    print(f"device: {args.device}   torch: {torch.__version__}   mode: {args.mode}")
    print(f"checkpoint: {CHECKPOINT.name} @ {CHECKPOINT_SHA[:12]}")
    if python_rate:
        print(f"python backend reference: {python_rate:,.0f} evals/s")
    print()
    header = (
        f"{'cap':>4} {'fwd ms':>7} {'roofline':>9} {'real':>9} {'diverse':>9} "
        f"{'%roof':>6} {'calls/s':>8} {'ev/call':>8} {'evaluator%':>11} {'vs python':>10}"
    )
    print(header)
    print("-" * len(header))

    factory = value_only_callback if args.mode == "value_only" else policy_value_callback
    for cap in args.caps:
        latency = forward_ms(model, cap, args.device, args.mode)
        roofline = cap / (latency / 1000.0)

        calls = {"n": 0}
        inner = factory(model, device=args.device)
        if args.mode == "value_only":
            def wrapped(features, _inner=inner, _calls=calls):
                _calls["n"] += 1
                return _inner(features)
        else:
            def wrapped(features, actions, offsets, _inner=inner, _calls=calls):
                _calls["n"] += 1
                return _inner(features, actions, offsets)

        common = dict(
            games=2 * cap,
            threads=args.threads,
            max_batch=cap,
            max_wait_us=2000,
            simulations=args.simulations,
            seconds=args.seconds,
        )
        real = native.schedule_with_callback(opening, module.SchedulerConfig(**common), wrapped, args.mode)
        real_rate = real["evaluations"] / real["wall_seconds"]

        # Same latency, but the salted dummy evaluator, so lanes truly diverge.
        diverse = native.schedule(
            opening, module.SchedulerConfig(eval_latency_ms=latency, **common)
        )
        diverse_rate = diverse["evaluations"] / diverse["wall_seconds"]

        crossings = max(1, calls["n"])
        speedup = f"{diverse_rate / python_rate:>9.1f}x" if python_rate else f"{'-':>10}"
        print(
            f"{cap:>4} {latency:>7.3f} {roofline:>9,.0f} {real_rate:>9,.0f} "
            f"{diverse_rate:>9,.0f} {diverse_rate / roofline * 100:>5.0f}% "
            f"{crossings / real['wall_seconds']:>8,.0f} "
            f"{real['evaluations'] / crossings:>8.1f} "
            f"{real['evaluator_seconds'] / real['wall_seconds'] * 100:>10.1f}% {speedup}"
        )
        mean_batch = statistics.mean(list(real["batch_sizes"]))
        if mean_batch < cap:  # a partial batch means the lanes could not keep up
            print(f"     note: mean batch {mean_batch:.1f} < cap {cap}")


if __name__ == "__main__":
    main()
