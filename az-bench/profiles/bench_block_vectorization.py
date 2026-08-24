"""Prototype: can the directional message-passing loop be issued in fewer kernels?

The forward is CPU-launch-bound -- its cost is flat from batch 1 to batch 256 and
the GPU drains in 22 us after 16 queued forwards -- so the only way to make it
cheaper is to issue fewer operations.

``DirectionalResidualBlock.forward`` currently runs a Python loop over the 6
board directions, and each iteration issues an index, a matmul, a Linear and an
add.  That is ~24 operations per block and ~144 across the 6-block trunk.

This prototype computes the identical quantity with two einsums:

    message = self_projection(nodes) + sum_d (adjacency[d] @ nodes) @ W_d^T

The sum over directions is exactly what an einsum contracts, so the result is
mathematically identical -- this is a reassociation, not an approximation, and
the parameters are untouched (same ``nn.Linear`` modules, same ``state_dict``).

Measures numerical parity first, then launch cost.  If parity fails or the cost
does not drop materially, the idea is dead and nothing gets implemented.
"""

from __future__ import annotations

import argparse
from time import perf_counter

import torch
from torch import Tensor

from diamond.alphazero.network.trunk import DirectionalResidualBlock, directional_adjacency
from diamond.contract.board import standard_board


def current_forward(block: DirectionalResidualBlock, nodes: Tensor, adjacency: Tensor) -> Tensor:
    message = block.self_projection(nodes)
    for direction, projection in enumerate(block.direction_projections):
        neighbours = torch.matmul(adjacency[direction], nodes)
        message = message + projection(neighbours)
    return nodes + block.activation(block.norm(message))


def vectorized_forward(block: DirectionalResidualBlock, nodes: Tensor, adjacency: Tensor) -> Tensor:
    # [directions, node, neighbour] x [batch, neighbour, width] -> [batch, directions, node, width]
    neighbours = torch.einsum("dij,bjw->bdiw", adjacency, nodes)
    # Stack the per-direction weights once, then contract width in one call.
    stacked = torch.stack([p.weight for p in block.direction_projections])  # [d, out, in]
    message = block.self_projection(nodes) + torch.einsum("bdiw,dvw->biv", neighbours, stacked)
    return nodes + block.activation(block.norm(message))


def timed(function, *, repeats: int, device: torch.device) -> tuple[float, float]:
    for _ in range(10):
        function()
    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    start = perf_counter()
    for _ in range(repeats):
        function()
    cpu_ms = (perf_counter() - start) / repeats * 1000.0
    end_event.record()
    torch.cuda.synchronize(device)
    return cpu_ms, start_event.elapsed_time(end_event) / repeats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--width", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(7)
    board = standard_board()
    adjacency = directional_adjacency(board).to(device)
    block = DirectionalResidualBlock(args.width).to(device).eval()

    print(f"[env] {torch.cuda.get_device_name(0)}, adjacency {tuple(adjacency.shape)}")
    print()

    print("--- numerical parity (fp32, same parameters) ---")
    worst = 0.0
    for batch in (1, 12, 32):
        nodes = torch.randn(batch, len(board), args.width, device=device)
        with torch.inference_mode():
            reference = current_forward(block, nodes, adjacency)
            candidate = vectorized_forward(block, nodes, adjacency)
        delta = (reference - candidate).abs().max().item()
        scale = reference.abs().max().item()
        worst = max(worst, delta)
        print(f"  batch {batch:>3}: max abs diff {delta:.3e}  (max |ref| {scale:.3f})")
    print(f"  worst {worst:.3e}")
    print()

    print("--- single-block launch cost ---")
    print(f"{'batch':>6} {'current_ms':>11} {'vector_ms':>10} {'speedup':>8}")
    for batch in (1, 12, 32, 64):
        nodes = torch.randn(batch, len(board), args.width, device=device)

        def run_current():
            with torch.inference_mode():
                current_forward(block, nodes, adjacency)

        def run_vector():
            with torch.inference_mode():
                vectorized_forward(block, nodes, adjacency)

        current_ms, _ = timed(run_current, repeats=args.repeats, device=device)
        vector_ms, _ = timed(run_vector, repeats=args.repeats, device=device)
        print(f"{batch:>6} {current_ms:>11.3f} {vector_ms:>10.3f} {current_ms / vector_ms:>8.2f}x")

    print()
    print("--- projected full-trunk effect (6 blocks) ---")
    nodes = torch.randn(12, len(board), args.width, device=device)

    def run_current():
        with torch.inference_mode():
            current_forward(block, nodes, adjacency)

    def run_vector():
        with torch.inference_mode():
            vectorized_forward(block, nodes, adjacency)

    current_ms, _ = timed(run_current, repeats=args.repeats, device=device)
    vector_ms, _ = timed(run_vector, repeats=args.repeats, device=device)
    saved = (current_ms - vector_ms) * 6
    print(f"  per block saved {current_ms - vector_ms:.3f} ms -> 6 blocks saved {saved:.3f} ms")
    print(f"  measured full forward is ~6.92 ms/batch; projected ~{6.92 - saved:.2f} ms/batch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
