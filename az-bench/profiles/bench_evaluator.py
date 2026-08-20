"""Isolated CUDA TorchEvaluator batch-scaling benchmark.

Answers: how much of `TorchEvaluator.evaluate` is the network forward, and how
much is the per-row GPU->CPU synchronization loop that follows it?

No coordinator, no multiprocessing -- one process, one evaluator, synthetic
requests with the real Soo feature shape and the real trained weights.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, "/workspace/alphadiamond/src")

from diamond.alphazero.config import NetworkConfig  # noqa: E402
from diamond.alphazero.evaluator.base import EvalRequest  # noqa: E402
from diamond.alphazero.evaluator.torch import TorchEvaluator  # noqa: E402
from diamond.alphazero.network import SooModel  # noqa: E402

DEVICE = "cuda:0"
NODES = 73
FEATURES = 4
LEGAL = 50          # legal actions per request
REPEATS = 60
WARMUP = 10


def make_requests(batch: int) -> tuple[EvalRequest, ...]:
    torch.manual_seed(7)
    node_features = tuple(
        tuple(float(v) for v in row)
        for row in torch.rand(NODES, FEATURES).tolist()
    )
    legal = tuple(range(LEGAL))
    return tuple(
        EvalRequest(
            node_features=node_features,
            legal_action_ids=legal,
            canonical_player_ids=(0, 1),
        )
        for _ in range(batch)
    )


def timed(fn, repeats: int = REPEATS) -> tuple[float, float]:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.mean(samples), statistics.median(samples)


def main() -> int:
    config = NetworkConfig(residual_blocks=6, width=128)
    model = SooModel(config, model_version="2.0.0")

    checkpoint = torch.load(
        "/workspace/alphadiamond/runtime/runs/soo/cpu8h-soo-20260819/latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    for key in ("model", "model_state", "model_state_dict", "state_dict"):
        if key in checkpoint:
            model.load_state_dict(checkpoint[key])
            print(f"loaded weights from checkpoint['{key}']")
            break
    else:
        print("WARNING: using random weights; keys =", sorted(checkpoint)[:12])

    evaluator = TorchEvaluator(model, value_size=1, device=DEVICE, precision="fp32")

    print(f"\ndevice={torch.cuda.get_device_name(0)}  legal_actions={LEGAL}  "
          f"nodes={NODES}x{FEATURES}\n")
    header = (f"{'batch':>6} {'evaluate':>10} {'forward':>10} {'postproc':>10} "
              f"{'post/row':>9} {'per_req':>8} {'speedup':>8}")
    print(header)
    print("-" * len(header))

    base_per_request = None
    rows = []
    for batch in (1, 2, 4, 8, 16, 30, 32, 64):
        requests = make_requests(batch)

        # Full path, exactly as the coordinator calls it.
        mean_eval, _ = timed(lambda: evaluator.evaluate(requests))

        # Forward only: the irreducible GPU work.
        features = torch.tensor(
            [r.node_features for r in requests], dtype=torch.float32, device=DEVICE
        )

        def forward_only():
            with torch.inference_mode():
                return evaluator.model(features)

        mean_fwd, _ = timed(forward_only)

        postproc = mean_eval - mean_fwd
        per_row = postproc / batch
        per_request = mean_eval / batch
        if base_per_request is None:
            base_per_request = per_request
        speedup = base_per_request / per_request

        rows.append((batch, mean_eval, mean_fwd, postproc, per_row, per_request, speedup))
        print(f"{batch:>6} {mean_eval:>9.3f}m {mean_fwd:>9.3f}m {postproc:>9.3f}m "
              f"{per_row:>8.3f}m {per_request:>7.3f}m {speedup:>7.2f}x")

    print("\n(all times milliseconds; 'postproc' = evaluate - forward = the "
          "per-row sync loop)\n")

    # What fraction of a realistic production batch is postprocessing?
    for batch, mean_eval, mean_fwd, postproc, *_ in rows:
        if batch in (1, 8, 30):
            share = 100.0 * postproc / mean_eval if mean_eval else 0.0
            print(f"batch {batch:>2}: forward {mean_fwd:.2f} ms, "
                  f"per-row sync loop {postproc:.2f} ms ({share:.0f}% of evaluate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
