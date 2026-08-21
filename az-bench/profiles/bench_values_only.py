"""Is the policy head dead work during the B0 bootstrap phase?

``VacancyPriorEvaluator`` keeps the network's *value* and replaces the policy
prior entirely with a heuristic. So in B0 self-play every neural policy logit is
computed, gathered, softmaxed, copied to host, turned into a dict, validated and
pickled -- and then thrown away.

That matters for the native callback ABI. If value-only inference is materially
cheaper, the B0 hot path can cross the Python boundary as

    features[B, 73, 4]  ->  values[B]

with no legal-action ids, offsets or priors, and the whole policy postprocessing
tail disappears. If it is not cheaper, the ABI should carry policy from the start.

Four paths, same trained checkpoint, same inputs:

  A  model.forward()                  trunk + policy head + value head
  B  trunk + value head only          the B0-relevant compute
  C  TorchEvaluator.evaluate()        today's full path, incl. postprocessing
  D  value-only evaluator path        tensor build + B + values D2H

D is the one that would sit behind the native callback; C is what it replaces.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch

from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.inference.model_pool import InferenceModelPool
from diamond.game.state import build_players, initial_state


def sample_eval_requests(count: int, seed: int = 11):
    players = build_players(2)
    state = initial_state(players)
    game = DiamondSearchAdapter(AlphaZeroGameAdapter(players, initial=state))
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        if game.is_terminal(state):
            state = initial_state(players)
        out.append(game.evaluation_request(state))
        actions = game.legal_action_ids(state)
        if not actions:
            state = initial_state(players)
            continue
        state = game.apply_action(state, rng.choice(actions))
    return out


def timed(function, *, repeats: int, device: torch.device) -> tuple[float, float]:
    """(CPU ms per call, device-span ms per call)."""
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


def count_aten_ops(function) -> int:
    """Count dispatched aten operations for one call, as a launch proxy."""
    from torch.utils._python_dispatch import TorchDispatchMode

    counter = Counter()

    class Counting(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            counter[str(func)] += 1
            return func(*args, **(kwargs or {}))

    with Counting():
        function()
    return sum(counter.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--batches", type=int, nargs="*", default=[1, 8, 16, 32, 64, 128])
    args = parser.parse_args()

    device = torch.device(args.device)
    compatibility = CheckpointCompatibilitySpec.soo(
        model_version="2.0.0", network_config=NetworkConfig(residual_blocks=6, width=128)
    )
    pool = InferenceModelPool(device=args.device)
    key = pool.activate_checkpoint(args.checkpoint, expected=compatibility)
    evaluator = pool.evaluator(key)
    model = evaluator.model

    print(f"[env] {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"[env] checkpoint {key.checkpoint_sha256[:16]}…")
    print()

    requests = sample_eval_requests(max(args.batches) * 2)

    print(f"{'batch':>6} | {'A full fwd':>19} | {'B trunk+value':>19} | "
          f"{'C TorchEvaluator':>19} | {'D value-only path':>19}")
    print(f"{'':>6} | {'cpu_ms':>9}{'dev_ms':>10} | {'cpu_ms':>9}{'dev_ms':>10} | "
          f"{'cpu_ms':>9}{'dev_ms':>10} | {'cpu_ms':>9}{'dev_ms':>10}")
    print("-" * 96)

    results = {}
    for batch in args.batches:
        rows = requests[:batch]
        features = torch.tensor(
            [r.node_features for r in rows], dtype=torch.float32, device=device
        )

        def path_a(features=features):
            with torch.inference_mode():
                model(features)

        def path_b(features=features):
            with torch.inference_mode():
                nodes = model.trunk(features)
                model.value_head(nodes.mean(dim=1))

        def path_c(rows=rows):
            evaluator.evaluate(tuple(rows))

        def path_d(rows=rows):
            # Exactly what a value-only native callback would run: build the
            # batch tensor from the wire buffer, trunk + value head, one D2H.
            with torch.inference_mode():
                x = torch.tensor(
                    [r.node_features for r in rows], dtype=torch.float32, device=device
                )
                nodes = model.trunk(x)
                values = model.value_head(nodes.mean(dim=1))
                values.cpu().tolist()

        a = timed(path_a, repeats=args.repeats, device=device)
        b = timed(path_b, repeats=args.repeats, device=device)
        c = timed(path_c, repeats=args.repeats, device=device)
        d = timed(path_d, repeats=args.repeats, device=device)
        results[batch] = (a, b, c, d)
        print(
            f"{batch:>6} | {a[0]:>9.3f}{a[1]:>10.3f} | {b[0]:>9.3f}{b[1]:>10.3f} | "
            f"{c[0]:>9.3f}{c[1]:>10.3f} | {d[0]:>9.3f}{d[1]:>10.3f}"
        )

    print()
    print("--- aten operation count per call (kernel-launch proxy) ---")
    rows = requests[:32]
    features = torch.tensor([r.node_features for r in rows], dtype=torch.float32, device=device)
    with torch.inference_mode():
        ops_a = count_aten_ops(lambda: model(features))
        ops_b = count_aten_ops(
            lambda: model.value_head(model.trunk(features).mean(dim=1))
        )
    ops_c = count_aten_ops(lambda: evaluator.evaluate(tuple(rows)))
    print(f"  A full forward          {ops_a:>5}")
    print(f"  B trunk + value head    {ops_b:>5}   ({ops_a - ops_b} fewer)")
    print(f"  C TorchEvaluator        {ops_c:>5}")

    print()
    print("--- roofline: single evaluator thread, batches/s x batch ---")
    print(f"{'batch':>6}{'C evals/s':>12}{'D evals/s':>12}{'gain':>8}")
    for batch in args.batches:
        (_, _), (_, _), (c_cpu, _), (d_cpu, _) = results[batch]
        c_rate = 1000.0 / c_cpu * batch
        d_rate = 1000.0 / d_cpu * batch
        print(f"{batch:>6}{c_rate:>12.0f}{d_rate:>12.0f}{d_rate / c_rate:>7.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
