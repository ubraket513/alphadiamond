"""Account for the gap between isolated worker MCTS cost and measured worker CPU.

`bench_worker_mcts.py` measures ~0.57 ms of MCTS+game CPU per evaluation, while
the production runs show ~5.3 ms of *worker* CPU per evaluation
(``(tree_cpu - parent_cpu) / evals_per_second``). A 9x gap that large changes
what a native port is worth, so it has to be attributed before the port is
designed around either number.

Two candidate explanations are tested:

  A. **Worker-side inference protocol.** Each evaluation builds an
     ``InferenceRequest`` (which validates all 292 feature floats), pickles it
     onto a *shared* request queue, and unpickles a response. None of that
     appears in the isolated search profile.

  B. **Oversubscription.** Production runs 48 worker processes on 18 physical
     cores. SMT siblings and memory contention inflate the CPU *time* charged
     for identical work, which psutil reports faithfully.

Mode B re-runs the identical isolated workload at N concurrent processes and
reports the inflation factor, so the two effects can be separated rather than
argued about.
"""

from __future__ import annotations

import argparse
import multiprocessing
import pickle
import random
from time import perf_counter

from diamond.alphazero.evaluator.base import EvalResult
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.inference.protocol import (
    InferenceRequest,
    InferenceResponse,
    ModelKey,
)
from diamond.contract.state import build_players, initial_state

MODEL_KEY = ModelKey(
    model_name="Soo", model_version="2.0.0", checkpoint_sha256="0" * 64
)


def sample_requests(count: int, seed: int = 11):
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


def measure_protocol(repeats: int) -> None:
    """Cost A: the per-evaluation Python protocol a worker pays around inference."""
    eval_requests = sample_requests(repeats)

    start = perf_counter()
    envelopes = [
        InferenceRequest.from_eval_request(
            client_id="game-x", request_id=f"game-x:{i}", model_key=MODEL_KEY, request=r
        )
        for i, r in enumerate(eval_requests)
    ]
    build_s = perf_counter() - start

    start = perf_counter()
    blobs = [pickle.dumps(e, protocol=pickle.HIGHEST_PROTOCOL) for e in envelopes]
    dumps_s = perf_counter() - start

    start = perf_counter()
    for blob in blobs:
        pickle.loads(blob)
    loads_s = perf_counter() - start

    responses = [
        InferenceResponse.from_eval_result(
            envelope,
            EvalResult(
                priors={a: 1.0 / len(envelope.legal_action_ids) for a in envelope.legal_action_ids},
                value=0.25,
            ),
        )
        for envelope in envelopes
    ]
    rblobs = [pickle.dumps(r, protocol=pickle.HIGHEST_PROTOCOL) for r in responses]

    start = perf_counter()
    decoded = [pickle.loads(b) for b in rblobs]
    rloads_s = perf_counter() - start

    start = perf_counter()
    for response in decoded:
        response.to_eval_result()
    to_result_s = perf_counter() - start

    print("Cost A -- worker-side inference protocol, per evaluation")
    print(f"{'step':<38}{'ms/eval':>10}")
    print("-" * 48)
    rows = [
        ("InferenceRequest.from_eval_request", build_s),
        ("pickle.dumps(request)  [feeder thread]", dumps_s),
        ("pickle.loads(request)  [parent side]", loads_s),
        ("pickle.loads(response) [pump thread]", rloads_s),
        ("InferenceResponse.to_eval_result", to_result_s),
    ]
    worker_side = build_s + dumps_s + rloads_s + to_result_s
    for name, seconds in rows:
        print(f"{name:<38}{seconds / repeats * 1000:>10.4f}")
    print("-" * 48)
    print(f"{'worker-side subtotal':<38}{worker_side / repeats * 1000:>10.4f}")
    print(f"{'(parent-side unpickle, for reference)':<38}{loads_s / repeats * 1000:>10.4f}")


def _busy(simulations: int, moves: int, queue) -> None:
    """Run the isolated search workload and report the CPU time it consumed."""
    import os
    import resource
    import sys

    sys.argv = ["x"]
    from bench_worker_mcts import DeterministicEvaluator, build_search, play

    evaluator = DeterministicEvaluator()
    inner, _game, search, state = build_search(simulations, None, evaluator)
    rng = random.Random(11)
    state, _ = play(search, inner, state, moves=3, rng=rng)

    before = resource.getrusage(resource.RUSAGE_SELF)
    cpu0 = before.ru_utime + before.ru_stime
    wall0 = perf_counter()
    evaluator.rows = 0
    play(search, inner, state, moves=moves, rng=rng)
    wall = perf_counter() - wall0
    after = resource.getrusage(resource.RUSAGE_SELF)
    cpu = after.ru_utime + after.ru_stime - cpu0
    queue.put((os.getpid(), evaluator.rows, cpu, wall))


def measure_contention(processes: int, simulations: int, moves: int) -> None:
    """Cost B: how much CPU time identical work costs at N-way oversubscription."""
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    workers = [
        context.Process(target=_busy, args=(simulations, moves, queue))
        for _ in range(processes)
    ]
    wall0 = perf_counter()
    for worker in workers:
        worker.start()
    results = [queue.get() for _ in workers]
    for worker in workers:
        worker.join()
    wall = perf_counter() - wall0

    evaluations = sum(r[1] for r in results)
    cpu = sum(r[2] for r in results)
    print(
        f"{processes:>4} procs: {evaluations:>7} evals  "
        f"cpu {cpu:>7.2f} s  wall {wall:>6.2f} s  "
        f"{cpu / evaluations * 1000:>7.3f} ms cpu/eval  "
        f"{evaluations / wall:>7.0f} evals/s"
    )
    return cpu / evaluations * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3000)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--moves", type=int, default=20)
    parser.add_argument("--procs", type=int, nargs="*", default=[1, 18, 36, 48])
    args = parser.parse_args()

    measure_protocol(args.repeats)
    print()
    print("Cost B -- same workload, N concurrent processes (18 physical cores)")
    baseline = None
    for processes in args.procs:
        value = measure_contention(processes, args.simulations, args.moves)
        if baseline is None:
            baseline = value
        else:
            print(f"      inflation vs 1 proc: {value / baseline:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
