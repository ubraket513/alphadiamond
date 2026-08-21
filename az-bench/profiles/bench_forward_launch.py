"""Separate CPU kernel-launch cost from GPU execution cost for the Soo forward.

The stage decomposition showed the forward's *CPU* time (6.92 ms/batch) almost
exactly matching its CUDA-event device span (6.97 ms/batch).  Two very different
situations produce that coincidence:

  * **GPU-bound** -- the device genuinely takes ~7 ms, and something in the path
    forces the CPU to wait for it.
  * **Launch-bound** -- the device work is trivial, but the CPU cannot issue
    kernels fast enough, so the device span is mostly idle gaps between kernels
    and the CPU is the thing actually consuming 7 ms.

They are distinguished by two tests this script runs:

  1. **Batch scaling.**  Kernel *count* is independent of batch size, so a
     launch-bound forward costs the same at batch 1 as at batch 32, while a
     GPU-bound one scales with work.
  2. **Pipelined submission.**  Issuing N forwards back-to-back without
     synchronising lets the CPU run ahead.  If the CPU is the constraint, N
     forwards cost N x the single-forward CPU time and the GPU never gets ahead;
     if the GPU is the constraint, CPU submission is much cheaper than the
     device time it represents.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch

from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.inference.model_pool import InferenceModelPool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    compatibility = CheckpointCompatibilitySpec.soo(
        model_version="2.0.0", network_config=NetworkConfig(residual_blocks=6, width=128)
    )
    pool = InferenceModelPool(device=args.device)
    key = pool.activate_checkpoint(args.checkpoint, expected=compatibility)
    model = pool.evaluator(key).model

    print(f"[env] {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    modules = sum(1 for _ in model.modules())
    print(f"[model] {modules} modules, {sum(p.numel() for p in model.parameters()):,} parameters")
    print()

    print("--- test 1: batch scaling (kernel count is constant; work is not) ---")
    print(f"{'batch':>6} {'cpu_ms':>9} {'device_ms':>10} {'cpu/device':>11} {'per_row_us':>11}")
    for batch in (1, 2, 4, 8, 12, 16, 24, 32, 64, 128, 256):
        features = torch.zeros((batch, 73, 4), dtype=torch.float32, device=device)
        for _ in range(10):  # warm up autotuning and allocator for this shape
            with torch.inference_mode():
                model(features)
        torch.cuda.synchronize(device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_event.record()
        cpu_start = perf_counter()
        for _ in range(args.repeats):
            with torch.inference_mode():
                model(features)
        cpu_elapsed = perf_counter() - cpu_start
        end_event.record()
        torch.cuda.synchronize(device)
        device_elapsed = start_event.elapsed_time(end_event) / 1000.0

        cpu_ms = cpu_elapsed / args.repeats * 1000.0
        device_ms = device_elapsed / args.repeats * 1000.0
        print(
            f"{batch:>6} {cpu_ms:>9.3f} {device_ms:>10.3f} "
            f"{cpu_ms / device_ms:>11.2f} {device_ms * 1000.0 / batch:>11.1f}"
        )

    print()
    print("--- test 2: pipelined submission at batch 12 (can the CPU run ahead?) ---")
    features = torch.zeros((12, 73, 4), dtype=torch.float32, device=device)
    for _ in range(10):
        with torch.inference_mode():
            model(features)
    torch.cuda.synchronize(device)

    for depth in (1, 2, 4, 8, 16):
        torch.cuda.synchronize(device)
        cpu_start = perf_counter()
        for _ in range(depth):
            with torch.inference_mode():
                model(features)
        submit_s = perf_counter() - cpu_start
        torch.cuda.synchronize(device)
        total_s = perf_counter() - cpu_start
        print(
            f"depth {depth:>3}: cpu submit {submit_s * 1000.0:>8.3f} ms  "
            f"total incl. drain {total_s * 1000.0:>8.3f} ms  "
            f"drain {(total_s - submit_s) * 1000.0:>7.3f} ms  "
            f"cpu/forward {submit_s / depth * 1000.0:>6.3f} ms"
        )

    print()
    print("--- test 3: where the launches go (module-level kernel counts) ---")
    counts: dict[str, int] = {}
    for name, module in model.named_modules():
        kind = type(module).__name__
        counts[kind] = counts.get(kind, 0) + 1
    for kind, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {kind:<24} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
