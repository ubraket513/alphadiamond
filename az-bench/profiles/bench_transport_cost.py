"""Measure what the multiprocessing transport itself costs, directly.

Component profiling accounts for only ~1.28 ms of the ~5.25 ms of worker CPU
that production burns per evaluation:

    MCTS + game            0.566 ms/eval   (bench_worker_mcts.py)
    bootstrap prior        0.576 ms/eval   (VacancyPriorEvaluator)
    inference protocol     0.135 ms/eval   (bench_worker_gap.py, Cost A)

The remainder cannot be explained by SMT oversubscription either: production
runs at ~12 % system CPU, so the 2x inflation measured at 48 saturating
processes does not apply.

Rather than attribute the difference by subtraction, this runs the **real**
``SelfPlayWorkerPool`` -- real spawn workers, real shared request queue, real
per-lane response queues, real bridge and coordinator -- against an evaluator
that answers instantly in the parent. No GPU, no model, no inference latency.
Whatever CPU that consumes above the component total is transport.

The same workload is then run single-process with no pool at all, which is the
useful-work baseline and, not coincidentally, roughly what a native threaded
engine would have to beat.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import replace
from typing import Self

import psutil

from diamond.alphazero.bootstrap.evaluator import bootstrap_evaluator
from diamond.alphazero.config import MCTSConfig, NetworkConfig, SelfPlayConfig
from diamond.alphazero.evaluator.base import EvalResult
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.inference.coordinator import InferenceConfig, InferenceCoordinator
from diamond.alphazero.inference.protocol import InferenceResponse, ModelKey
from diamond.alphazero.inference.summary import summarize_metrics
from diamond.alphazero.orchestration.selfplay_workers import SelfPlayJob, SelfPlayWorkerPool
from diamond.alphazero.selfplay.runner_2p import SooSelfPlayRunner
from diamond.contract.state import build_players, initial_state

MODEL_KEY = ModelKey(model_name="Soo", model_version="2.0.0", checkpoint_sha256="0" * 64)
PRIOR = "canonical-target-vacancy-distance-v2"


class InstantBatchEvaluator:
    """Answer a whole coordinator batch with no work, so only transport remains."""

    def __init__(self) -> None:
        self.requests = 0
        self.batches = 0

    def evaluate(self, requests):
        self.batches += 1
        self.requests += len(requests)
        out = []
        for request in requests:
            actions = request.legal_action_ids
            share = 1.0 / len(actions)
            out.append(
                InferenceResponse(
                    client_id=request.client_id,
                    request_id=request.request_id,
                    model_key=request.model_key,
                    priors=tuple((action, share) for action in actions),
                    value=(0.125,),
                )
            )
        return tuple(out)


class InstantLocalEvaluator:
    """The same answers, delivered in-process with no transport at all."""

    def __init__(self) -> None:
        self.rows = 0

    def evaluate(self, requests):
        self.rows += len(requests)
        return tuple(
            EvalResult(
                priors={a: 1.0 / len(r.legal_action_ids) for a in r.legal_action_ids},
                value=0.125,
            )
            for r in requests
        )


def configs(simulations: int, max_moves: int):
    compatibility = CheckpointCompatibilitySpec.soo(
        model_version="2.0.0", network_config=NetworkConfig(residual_blocks=6, width=128)
    )
    mcts = MCTSConfig(
        c_puct=1.5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25, seed=7, simulations=simulations
    )
    selfplay = SelfPlayConfig(
        bootstrap_prior=PRIOR,
        max_game_seconds=900.0,
        max_moves=max_moves,
        seed=7,
        temperature=1.0,
        temperature_moves=20,
    )
    return compatibility, mcts, selfplay


class ChildCpuAccountant:
    """Accumulate child CPU while the children are still alive.

    Sampling children after ``SelfPlayWorkerPool.run`` returns reports zero:
    its ``finally`` joins, terminates and closes every worker first, so the
    processes are gone before they can be read. (That bug made an earlier
    version of this benchmark report parent-only CPU as a tree total, which in
    turn made transport look free.)

    A poller therefore records the last CPU time seen for each PID and keeps it
    after the process exits; the sum over PIDs is the tree's child CPU.
    """

    def __init__(self, parent: psutil.Process, interval: float = 0.25) -> None:
        self._parent = parent
        self._interval = interval
        self._last: dict[int, float] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        for child in self._parent.children(recursive=True):
            try:
                self._last[child.pid] = sum(child.cpu_times()[:2])
            except psutil.Error:
                continue

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    @property
    def seconds(self) -> float:
        return sum(self._last.values())


def run_pool(workers: int, games: int, simulations: int, max_moves: int) -> None:
    compatibility, mcts, selfplay = configs(simulations, max_moves)
    players = build_players(2)
    start_state = initial_state(players)
    jobs = tuple(
        SelfPlayJob(
            run_seed=7,
            iteration=0,
            game_index=index,
            retry_id="attempt-0",
            model_key=MODEL_KEY,
            compatibility=compatibility,
            players=players,
            initial_state=start_state,
            mcts_config=mcts,
            selfplay_config=selfplay,
        )
        for index in range(games)
    )

    evaluator = InstantBatchEvaluator()
    coordinator = InferenceCoordinator(
        evaluator,
        InferenceConfig(
            max_batch_size=32, max_wait_ms=2, request_queue_capacity=1024, response_timeout_s=600.0
        ),
    )
    coordinator.start()
    me = psutil.Process(os.getpid())
    parent0 = sum(me.cpu_times()[:2])
    wall0 = time.perf_counter()
    with ChildCpuAccountant(me) as accountant:
        try:
            episodes = SelfPlayWorkerPool(
                coordinator, worker_count=workers, per_game_timeout_s=900.0
            ).run(jobs)
        finally:
            wall = time.perf_counter() - wall0
            parent1 = sum(me.cpu_times()[:2])
            metrics = coordinator.metrics
            coordinator.stop()

    parent_cpu = parent1 - parent0
    worker_cpu = accountant.seconds
    total_cpu = parent_cpu + worker_cpu
    moves = sum(e.move_count for e in episodes)
    evaluations = evaluator.requests
    summary = summarize_metrics(metrics, elapsed_s=wall)
    print(
        f"POOL workers={workers:<3} games={games:<3} evals={evaluations:<7} moves={moves:<5} "
        f"wall={wall:>6.2f}s  {evaluations / wall:>7.0f} evals/s"
    )
    print(
        f"     cpu ms/eval  parent {parent_cpu / evaluations * 1000:>6.3f}  "
        f"workers {worker_cpu / evaluations * 1000:>6.3f}  "
        f"total {total_cpu / evaluations * 1000:>6.3f}   "
        f"(parent {parent_cpu:>6.2f}s = {parent_cpu / wall * 100:>5.1f}% of one core)"
    )
    print(
        f"     batches/s {summary['batches_per_second']:>6.1f}  "
        f"mean batch {summary['mean_batch_size']:>5.2f}  "
        f"max {summary['max_batch_size']:>3}  "
        f"q->disp p50 {summary['queue_to_dispatch_p50_ms']:>6.2f} ms  "
        f"resp p50 {summary['response_p50_ms']:>6.2f} ms"
    )
    return total_cpu / evaluations * 1000


def run_local(games: int, simulations: int, max_moves: int) -> None:
    compatibility, mcts, selfplay = configs(simulations, max_moves)
    players = build_players(2)
    start_state = initial_state(players)
    evaluator = InstantLocalEvaluator()
    wrapped = bootstrap_evaluator(evaluator, PRIOR)

    me = psutil.Process(os.getpid())
    cpu0 = sum(me.cpu_times()[:2])
    wall0 = time.perf_counter()
    moves = 0
    for index in range(games):
        game = DiamondSearchAdapter(AlphaZeroGameAdapter(players, initial=start_state))
        episode = SooSelfPlayRunner(
            game,
            wrapped,
            replace(mcts, seed=7 + index),
            replace(selfplay, seed=7 + index),
            compatibility,
        ).run()
        moves += episode.move_count
    wall = time.perf_counter() - wall0
    cpu = sum(me.cpu_times()[:2]) - cpu0
    evaluations = evaluator.rows
    print(
        f"LOCAL  1 process   games={games:<3} "
        f"evals={evaluations:<7} moves={moves:<5} "
        f"wall={wall:>6.2f}s cpu={cpu:>7.2f}s "
        f"=> {cpu / evaluations * 1000:>6.3f} ms cpu/eval  "
        f"{evaluations / wall:>7.0f} evals/s"
    )
    return cpu / evaluations * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--max-moves", type=int, default=25)
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--workers", type=int, nargs="*", default=[12, 24, 48])
    args = parser.parse_args()

    print("Identical Soo self-play work, with and without the multiprocessing transport.")
    print("Evaluator answers instantly in both arms, so any difference is transport.\n")

    local = run_local(args.games, args.simulations, args.max_moves)
    print()
    for workers in args.workers:
        pool = run_pool(workers, max(workers, args.games), args.simulations, args.max_moves)
        print(f"       transport overhead vs local: {pool - local:+.3f} ms/eval "
              f"({pool / local:.1f}x total CPU)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
