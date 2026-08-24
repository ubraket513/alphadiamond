"""Stage-level worker CPU profile of a real trained-NN production self-play run.

This is the primary measurement for deciding native scope. Earlier component
profiles used an instant evaluator, which is suspect: a constant value changes
PUCT's Q, which changes which branches the search re-descends, which changes how
many ``_select``/``apply_action``/``legal_action_ids`` calls each evaluation
costs. Only the real network produces the real trajectory.

``time.thread_time_ns()`` is used rather than wall clock because a self-play
worker spends most of its life blocked on inference; thread CPU time excludes
that wait, so what remains is the CPU the worker actually burns.

Nothing in ``src/`` is modified. multiprocessing's spawn start method re-imports
``__main__`` in every child, so patching the classes here -- at import, before
the pool hands work out -- instruments the workers too. Each worker writes its
own counters at exit; the parent aggregates them.

Also recorded, because they are what distinguish the two hypotheses:

    select steps / eval      apply_action calls / eval
    legal_action_ids / eval  evaluations / move

If these are far above what the instant-evaluator runs showed, the trajectory
really did differ and the earlier numbers understate production.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path

OUT_DIR = Path(os.environ.get("AZ_STAGE_OUT", "/tmp/az-stages"))

NS = Counter()
CALLS = Counter()
_WALL = Counter()


def _install_patches() -> None:
    """Wrap the worker-side hot path in thread-CPU accounting, in place."""
    from diamond.alphazero.bootstrap.evaluator import VacancyPriorEvaluator
    from diamond.alphazero.game_adapter import DiamondSearchAdapter
    from diamond.alphazero.inference.remote import RemoteEvaluator
    from diamond.alphazero.mcts.search_2p import MCTS2P

    thread_time = time.thread_time_ns
    perf = time.perf_counter_ns

    # Exclusive timing. `evaluation_request` calls `legal_action_ids`, and
    # `_expand` calls both the evaluator and `legal_action_ids`, so naive
    # inclusive timers double-count: the first version of this profiler summed
    # to 112% of the search total. Each frame subtracts the time its nested
    # instrumented callees consumed, and reports that to its own caller.
    local = threading.local()

    def wrap(owner, name, key):
        original = getattr(owner, name)

        def instrumented(*args, **kwargs):
            stack = getattr(local, "stack", None)
            if stack is None:
                stack = local.stack = []
            stack.append(0)
            start = thread_time()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = thread_time() - start
                nested = stack.pop()
                NS[key] += elapsed - nested
                CALLS[key] += 1
                if stack:
                    stack[-1] += elapsed

        setattr(owner, name, instrumented)

    for name, key in (
        ("legal_action_ids", "legal_action_ids"),
        ("apply_action", "apply_action"),
        ("evaluation_request", "encode"),
        ("is_terminal", "is_terminal"),
        ("current_player_id", "current_player_id"),
    ):
        wrap(DiamondSearchAdapter, name, key)

    wrap(MCTS2P, "_select", "select")
    wrap(VacancyPriorEvaluator, "_priors", "bootstrap_prior")

    # The remote call is the one place a worker blocks: track CPU and wall
    # separately so the wait is visible without being charged as work.
    remote_evaluate = RemoteEvaluator.evaluate

    def instrumented_remote(self, requests):
        stack = getattr(local, "stack", None)
        if stack is None:
            stack = local.stack = []
        stack.append(0)
        cpu0, wall0 = thread_time(), perf()
        try:
            return remote_evaluate(self, requests)
        finally:
            elapsed = thread_time() - cpu0
            nested = stack.pop()
            NS["remote_cpu"] += elapsed - nested
            _WALL["remote_wall"] += perf() - wall0
            CALLS["remote_cpu"] += 1
            CALLS["evaluations"] += len(requests)
            if stack:
                stack[-1] += elapsed

    RemoteEvaluator.evaluate = instrumented_remote

    # ``run`` is the outermost frame: it must not subtract its callees, because
    # its total is the denominator every share is expressed against.
    run = MCTS2P.run

    def instrumented_run(self, state, *, temperature=0.0):
        start = thread_time()
        try:
            return run(self, state, temperature=temperature)
        finally:
            NS["search_total"] += thread_time() - start
            CALLS["search_total"] += 1

    MCTS2P.run = instrumented_run

    # ``_expand`` is instrumented only to be a *transparent* nesting frame, so
    # the evaluator and prior time it contains is attributed to them and its own
    # remainder (the legal/prior set comparison, edge dict build) is visible.
    expand = MCTS2P._expand

    def instrumented_expand(self, node, *, root_noise=False):
        stack = getattr(local, "stack", None)
        if stack is None:
            stack = local.stack = []
        stack.append(0)
        start = thread_time()
        try:
            return expand(self, node, root_noise=root_noise)
        finally:
            elapsed = thread_time() - start
            nested = stack.pop()
            NS["expand_own"] += elapsed - nested
            CALLS["expand_own"] += 1
            if stack:
                stack[-1] += elapsed

    MCTS2P._expand = instrumented_expand


def _dump() -> None:
    if not CALLS:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        process_cpu_s = usage.ru_utime + usage.ru_stime
    except Exception:  # noqa: BLE001 - telemetry must not break a worker
        process_cpu_s = None
    payload = {
        "pid": os.getpid(),
        "ns": dict(NS),
        "calls": dict(CALLS),
        "wall_ns": dict(_WALL),
        "main_thread_cpu_s": time.thread_time(),
        "process_cpu_s": process_cpu_s,
    }
    (OUT_DIR / f"stage-{os.getpid()}.json").write_text(json.dumps(payload), encoding="utf-8")


_install_patches()
atexit.register(_dump)


# --------------------------------------------------------------------------
# parent side
# --------------------------------------------------------------------------


def main() -> int:
    import psutil
    import torch

    from diamond.alphazero.config import MCTSConfig, NetworkConfig, SelfPlayConfig
    from diamond.alphazero.identity import CheckpointCompatibilitySpec
    from diamond.alphazero.inference.coordinator import InferenceConfig, InferenceCoordinator
    from diamond.alphazero.inference.model_pool import InferenceModelPool
    from diamond.alphazero.inference.summary import summarize_metrics
    from diamond.alphazero.orchestration.selfplay_workers import SelfPlayJob, SelfPlayWorkerPool
    from diamond.contract.state import build_players, initial_state

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--games", type=int, default=48)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--max-wait-ms", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(4)
    for stale in OUT_DIR.glob("stage-*.json"):
        stale.unlink()

    compatibility = CheckpointCompatibilitySpec.soo(
        model_version="2.0.0", network_config=NetworkConfig(residual_blocks=6, width=128)
    )
    pool = InferenceModelPool(device="cuda:0")
    model_key = pool.activate_checkpoint(args.checkpoint, expected=compatibility)
    print(f"[env] checkpoint {model_key.checkpoint_sha256[:16]}…  gpu {torch.cuda.get_device_name(0)}")

    players = build_players(2)
    start_state = initial_state(players)
    mcts = MCTSConfig(
        c_puct=1.5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25, seed=7,
        simulations=args.simulations,
    )
    selfplay = SelfPlayConfig(
        bootstrap_prior="canonical-target-vacancy-distance-v2",
        max_game_seconds=900.0, max_moves=2000, seed=7,
        temperature=1.0, temperature_moves=20,
    )
    jobs = tuple(
        SelfPlayJob(
            run_seed=7, iteration=0, game_index=index, retry_id="attempt-0",
            model_key=model_key, compatibility=compatibility, players=players,
            initial_state=start_state, mcts_config=mcts, selfplay_config=selfplay,
        )
        for index in range(args.games)
    )

    coordinator = InferenceCoordinator(
        pool,
        InferenceConfig(
            max_batch_size=32, max_wait_ms=args.max_wait_ms,
            request_queue_capacity=1024, response_timeout_s=600.0,
        ),
    )
    coordinator.start()

    me = psutil.Process(os.getpid())
    child_cpu: dict[int, float] = {}
    stop = threading.Event()

    def poll():
        while not stop.wait(0.25):
            for child in me.children(recursive=True):
                try:
                    child_cpu[child.pid] = sum(child.cpu_times()[:2])
                except psutil.Error:
                    continue

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    parent0 = sum(me.cpu_times()[:2])
    wall0 = time.perf_counter()
    try:
        episodes = SelfPlayWorkerPool(
            coordinator, worker_count=args.workers, per_game_timeout_s=900.0
        ).run(jobs)
    finally:
        wall = time.perf_counter() - wall0
        parent_cpu = sum(me.cpu_times()[:2]) - parent0
        stop.set()
        poller.join(timeout=2.0)
        metrics = coordinator.metrics
        coordinator.stop()

    worker_cpu = sum(child_cpu.values())
    summary = summarize_metrics(metrics, elapsed_s=wall)
    evaluations = metrics.requests_completed
    moves = sum(e.move_count for e in episodes)
    completed = sum(1 for e in episodes if e.completed)

    print()
    print(f"games {completed}/{len(episodes)}  moves {moves}  evals {evaluations}  wall {wall:.1f}s")
    print(f"throughput {evaluations / wall:.0f} evals/s   evals/move {evaluations / moves:.1f}")
    print(
        f"cpu ms/eval: parent {parent_cpu / evaluations * 1000:.3f}  "
        f"workers {worker_cpu / evaluations * 1000:.3f}  "
        f"total {(parent_cpu + worker_cpu) / evaluations * 1000:.3f}"
    )
    print(
        f"batches/s {summary['batches_per_second']:.1f}  mean batch {summary['mean_batch_size']:.2f}"
        f"  q->disp p50 {summary['queue_to_dispatch_p50_ms']:.2f} ms"
        f"  resp p50 {summary['response_p50_ms']:.2f} ms"
    )

    # ---- aggregate worker stage counters ---------------------------------
    files = sorted(OUT_DIR.glob("stage-*.json"))
    ns, calls, wall_ns = Counter(), Counter(), Counter()
    main_cpu = other_cpu = 0.0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ns.update(payload["ns"])
        calls.update(payload["calls"])
        wall_ns.update(payload["wall_ns"])
        main_cpu += payload["main_thread_cpu_s"]
        if payload["process_cpu_s"] is not None:
            other_cpu += payload["process_cpu_s"] - payload["main_thread_cpu_s"]

    evals = calls.get("evaluations", 0)
    if not evals:
        print("\n[warn] no worker stage data collected")
        return 1

    print()
    print(f"--- worker stage CPU (thread_time), {len(files)} workers, {evals} evaluations ---")
    search_total = ns.get("search_total", 0)
    attributed = sum(
        ns.get(k, 0)
        for k in ("legal_action_ids", "apply_action", "encode", "is_terminal",
                  "current_player_id", "select", "bootstrap_prior", "remote_cpu")
    )
    attributed += ns.get("expand_own", 0)
    rows = [(k, ns[k], calls.get(k, 0)) for k in ns if k != "search_total"]
    rows.append(("MCTS residual (tree/backup)", max(0, search_total - attributed), 0))
    print(f"{'stage':<28}{'ms/eval':>10}{'share':>8}{'calls/eval':>12}{'us/call':>10}")
    print("-" * 68)
    for name, nanos, count in sorted(rows, key=lambda r: -r[1]):
        share = nanos / search_total * 100 if search_total else 0.0
        per_call = f"{nanos / count / 1000:>10.2f}" if count else f"{'-':>10}"
        per_eval_calls = f"{count / evals:>12.2f}" if count else f"{'-':>12}"
        print(f"{name:<28}{nanos / evals / 1e6:>10.4f}{share:>7.1f}%{per_eval_calls}{per_call}")
    print("-" * 68)
    print(f"{'search_total (main thread)':<28}{search_total / evals / 1e6:>10.4f}{100.0:>7.1f}%")
    print()
    print(f"main-thread CPU  {main_cpu / evals * 1000:.3f} ms/eval   "
          f"other worker threads (pump/feeder)  {other_cpu / evals * 1000:.3f} ms/eval")
    print(f"remote evaluate WALL (blocked)      {wall_ns.get('remote_wall', 0) / evals / 1e6:.3f} ms/eval")
    print()
    print("trajectory:")
    for key in ("select", "apply_action", "legal_action_ids", "encode", "bootstrap_prior"):
        print(f"  {key:<20} {calls.get(key, 0) / evals:>7.2f} calls/eval")
    print(f"  {'evaluations/move':<20} {evaluations / moves:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
