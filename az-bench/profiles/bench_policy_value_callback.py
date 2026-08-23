"""Why A0 self-play evaluates slower than B0, and which fixes actually pay.

A0 has no bootstrap heuristic, so every native evaluation goes through
``policy_value_callback`` where B0 uses ``value_only``.  The production ledger,
compared at matched batch size, puts that at 1.25-1.45x per batch.  This locates
the cost and measures the candidate fixes -- including the one that does not
work, which is the more useful half.

Measurements are **interleaved**: one round times every variant back to back and
the result is the median of paired ratios.  Timing variants in sequence on a
shared GPU is unreliable; an earlier version did that and reported a 3x
difference that was mostly the training run's load drifting underneath it.

Variants:

``value_only``
    What B0 pays.  The floor.
``device rows``
    The callback as it shipped before this benchmark existed.  Builds row
    membership with ``repeat_interleave(arange(n, device=cuda), counts_cuda)``.
``output_size`` (now shipped)
    Identical output, with the length ``repeat_interleave`` would otherwise sync
    to discover passed in -- ``offsets`` already knows it on the host.
``host rows``
    Identical output, ``rows`` built with ``np.repeat`` and transferred.  Ties
    with ``output_size`` on every shape measured, and was tried first, but it
    moves a 41 KB buffer and takes row construction off the device.
``legal only``
    Scores just the legal actions instead of building ``[B, 5329]`` and gathering
    ~1 % of it back.  Reads like the obvious win and **is not one** -- the dense
    head is a single well-shaped matmul and the sparse version trades it for two
    advanced-indexing gathers.  Kept here so the measurement outlives the idea.

Usage::

    python az-bench/profiles/bench_policy_value_callback.py --device cuda:0
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.native.backend import _segmented_softmax
from diamond.alphazero.network import SooModel


def _variants(model, torch, device, feats, actions, counts):
    rows_host = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    board = feats.shape[1]

    def head():
        batch = torch.from_numpy(feats).to(device)
        with torch.inference_mode():
            nodes = model.trunk(batch)
            values = model.value_head(nodes.mean(dim=1))
        return nodes, values

    def finish(priors, values):
        return (
            priors.to(torch.float32).cpu().numpy(),
            values.reshape(-1).to(torch.float32).cpu().numpy(),
        )

    def value_only():
        batch = torch.from_numpy(feats).to(device)
        with torch.inference_mode():
            values = model.value_head(model.trunk(batch).mean(dim=1))
        return values.reshape(-1).to(torch.float32).cpu().numpy()

    def device_rows():
        nodes, values = head()
        with torch.inference_mode():
            logits = model.policy_head(nodes)
        counts_device = torch.from_numpy(counts).to(device)
        if int(counts_device.min()) <= 0:  # the second sync this variant pays
            raise ValueError("empty row")
        rows = torch.repeat_interleave(
            torch.arange(counts_device.numel(), device=device), counts_device
        )
        columns = torch.from_numpy(actions).to(device)
        return finish(_segmented_softmax(logits[rows, columns], rows, counts.size), values)

    def output_size():
        nodes, values = head()
        with torch.inference_mode():
            logits = model.policy_head(nodes)
        counts_device = torch.from_numpy(counts).to(device)
        rows = torch.repeat_interleave(
            torch.arange(counts_device.numel(), device=device),
            counts_device,
            output_size=int(actions.size),
        )
        columns = torch.from_numpy(actions).to(device)
        return finish(_segmented_softmax(logits[rows, columns], rows, counts.size), values)

    def host_rows():
        nodes, values = head()
        with torch.inference_mode():
            logits = model.policy_head(nodes)
        rows = torch.from_numpy(rows_host).to(device)
        columns = torch.from_numpy(actions).to(device)
        return finish(_segmented_softmax(logits[rows, columns], rows, counts.size), values)

    def legal_only():
        nodes, values = head()
        rows = torch.from_numpy(rows_host).to(device)
        columns = torch.from_numpy(actions).to(device)
        sources = torch.div(columns, board, rounding_mode="floor")
        destinations = columns - sources * board
        with torch.inference_mode():
            source = model.policy_head.source(nodes[rows, sources])
            destination = model.policy_head.destination(nodes[rows, destinations])
            selected = (source * destination).sum(-1) / model.policy_head.scale
        return finish(_segmented_softmax(selected, rows, counts.size), values)

    return [
        ("value_only", value_only),
        ("device rows (was shipped)", device_rows),
        ("output_size (now shipped)", output_size),
        ("host rows", host_rows),
        ("legal only", legal_only),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, nargs="+", default=[6, 12])
    parser.add_argument("--batches", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--legal", type=int, default=40, help="legal actions per row")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--calls", type=int, default=40)
    args = parser.parse_args()

    import torch

    device = torch.device(args.device)

    def timed(fn, calls):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(calls):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - started) / calls * 1e3

    print(
        f"device={args.device} width={args.width} legal/row={args.legal} "
        f"rounds={args.rounds} calls={args.calls}   (ratios are the result, "
        f"absolutes reflect whatever else the GPU was doing)"
    )
    for blocks in args.blocks:
        model = SooModel(
            NetworkConfig(width=args.width, residual_blocks=blocks),
            model_version="2.0.0",
        ).to(device).eval()
        board = len(model.trunk.board)
        print(f"\n  {args.width}x{blocks}")
        print(f"    {'batch':>6}" + "".join(f"{name:>28}" for name, _ in
                                            _variants(model, torch,
                                                      device,
                                                      np.zeros((1, board, 4), np.float32),
                                                      np.zeros(1, np.int64),
                                                      np.ones(1, np.int64))))
        for batch in args.batches:
            feats = np.random.rand(batch, board, 4).astype(np.float32)
            counts = np.full(batch, args.legal, dtype=np.int64)
            actions = np.random.randint(0, board * board, batch * args.legal).astype(np.int64)
            variants = _variants(model, torch, device, feats, actions, counts)

            reference = variants[1][1]()
            for name, fn in variants[2:]:
                got = fn()
                if not (np.allclose(got[0], reference[0], atol=1e-6)
                        and np.allclose(got[1], reference[1], atol=1e-6)):
                    raise SystemExit(f"{name} does not reproduce the shipped priors")

            for _, fn in variants:
                timed(fn, 20)
            rounds = [[timed(fn, args.calls) for _, fn in variants] for _ in range(args.rounds)]
            medians = [statistics.median(r[i] for r in rounds) for i in range(len(variants))]
            # Ratios are medians of per-round pairs, not a ratio of medians: the
            # point of interleaving is that each round shares one load level.
            cells = "".join(
                f"{t:>18.3f} x{statistics.median(r[i] / r[0] for r in rounds):>7.3f}"
                for i, t in enumerate(medians)
            )
            print(f"    {batch:>6}{cells}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
